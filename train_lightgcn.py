import gin
import os
import torch
import wandb

from accelerate import Accelerator
from data.collaborative import load_collaborative_split
from data.collaborative import sample_bpr_batch
from data.processed import RecDataset
from modules.lightgcn import LightGCN
from modules.lightgcn import build_normalized_adjacency
from modules.utils import parse_config
from torch.optim import Adam
from tqdm import tqdm


@gin.configurable
def train(
    iterations: int = 10000,
    batch_size: int = 2048,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    l2_reg_weight: float = 1e-4,
    dataset_folder: str = "dataset/amazon",
    dataset: RecDataset = RecDataset.AMAZON,
    dataset_split: str = "beauty",
    save_dir_root: str = "out/lightgcn/amazon_beauty/",
    save_model_every: int = 5000,
    eval_every: int = 5000,
    embedding_dim: int = 64,
    n_layers: int = 3,
    force_dataset_process: bool = False,
    wandb_logging: bool = False,
    split_batches: bool = True,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
):
    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    collab = load_collaborative_split(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
    )

    model = LightGCN(
        num_users=collab.num_users,
        num_items=collab.num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)
    adjacency = build_normalized_adjacency(
        num_users=collab.num_users,
        num_items=collab.num_items,
        edge_index=collab.train_edge_index.to(device),
        device=device,
    )

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="lightgcn-training", config=params)

    running_losses = []
    with tqdm(total=iterations, disable=not accelerator.is_main_process) as pbar:
        for iteration in range(iterations):
            model.train()
            users, pos_items, neg_items = sample_bpr_batch(
                train_user_items=collab.train_user_items,
                num_items=collab.num_items,
                batch_size=batch_size,
                device=device,
            )

            with accelerator.autocast():
                ranking_loss, reg_loss = model.bpr_loss(adjacency, users, pos_items, neg_items)
                loss = ranking_loss + l2_reg_weight * reg_loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_losses.append(loss.detach().cpu().item())
            running_losses = running_losses[-200:]
            pbar.set_description(f"loss: {sum(running_losses) / len(running_losses):.4f}")
            pbar.update(1)

            if accelerator.is_main_process and wandb_logging:
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "ranking_loss": ranking_loss.detach().cpu().item(),
                        "reg_loss": reg_loss.detach().cpu().item(),
                    },
                    step=iteration,
                )

            if accelerator.is_main_process and ((iteration + 1) % save_model_every == 0 or iteration + 1 == iterations):
                state = {
                    "iter": iteration,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "model_config": {
                        "num_users": collab.num_users,
                        "num_items": collab.num_items,
                        "embedding_dim": embedding_dim,
                        "n_layers": n_layers,
                    },
                    "optimizer": optimizer.state_dict(),
                    "dataset_meta": {
                        "dataset_folder": dataset_folder,
                        "dataset": dataset.name,
                        "dataset_split": dataset_split,
                    },
                    "train_item_mask": collab.train_item_mask,
                    "eval_targets": collab.eval_targets,
                    "test_targets": collab.test_targets,
                }

                if not os.path.exists(save_dir_root):
                    os.makedirs(save_dir_root)
                torch.save(state, os.path.join(save_dir_root, f"checkpoint_{iteration}.pt"))

            if accelerator.is_main_process and wandb_logging and ((iteration + 1) % eval_every == 0 or iteration + 1 == iterations):
                with torch.no_grad():
                    embeddings = accelerator.unwrap_model(model).compute_embeddings(adjacency)
                    wandb.log(
                        {
                            "item_embedding_norm": embeddings.item_embeddings.norm(dim=-1).mean().cpu().item(),
                            "user_embedding_norm": embeddings.user_embeddings.norm(dim=-1).mean().cpu().item(),
                        },
                        step=iteration,
                    )

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
