import gin
import os
import random
import torch
import wandb

from accelerate import Accelerator
from data.collaborative import build_sasrec_training_tensors
from data.collaborative import load_collaborative_split
from data.processed import RecDataset
from modules.sasrec import SasRec
from modules.utils import parse_config
from torch.optim import Adam
from tqdm import tqdm


@gin.configurable
def train(
    iterations: int = 20000,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    dataset_folder: str = "dataset/amazon",
    dataset: RecDataset = RecDataset.AMAZON,
    dataset_split: str = "beauty",
    save_dir_root: str = "out/sasrec/amazon_beauty/",
    save_model_every: int = 5000,
    max_seq_len: int = 50,
    hidden_size: int = 64,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
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

    model = SasRec(
        num_items=collab.num_items,
        max_len=max_seq_len,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="sasrec-training", config=params)

    running_losses = []
    user_indices = list(range(collab.num_users))
    with tqdm(total=iterations, disable=not accelerator.is_main_process) as pbar:
        for iteration in range(iterations):
            model.train()
            sampled_users = random.sample(user_indices, k=min(batch_size, len(user_indices)))
            batch_sequences = [collab.train_sequences[idx] for idx in sampled_users]
            seq, pos, neg = build_sasrec_training_tensors(
                train_sequences=batch_sequences,
                num_items=collab.num_items,
                max_len=max_seq_len,
                device=device,
            )

            with accelerator.autocast():
                model_output = model(seq=seq, pos=pos, neg=neg)
                loss = model_output.loss

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
                        "item_embedding_norm": accelerator.unwrap_model(model).get_item_embeddings().norm(dim=-1).mean().cpu().item(),
                    },
                    step=iteration,
                )

            if accelerator.is_main_process and ((iteration + 1) % save_model_every == 0 or iteration + 1 == iterations):
                state = {
                    "iter": iteration,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "model_config": {
                        "num_items": collab.num_items,
                        "max_len": max_seq_len,
                        "hidden_size": hidden_size,
                        "num_blocks": num_blocks,
                        "num_heads": num_heads,
                        "dropout": dropout,
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

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
