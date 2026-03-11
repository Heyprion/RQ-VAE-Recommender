import gin
import os
import torch
import numpy as np
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from modules.tokenizer.collaborative_trainer import CollaborativeTokenizerTrainer
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/amazon",
    dataset=RecDataset.AMAZON,
    dataset_split="beauty",
    pretrained_rqvae_path=None,
    resume_checkpoint_path=None,
    collab_embeddings_path=None,
    save_dir_root="out/rqvae/amazon_letter/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=5000,
    eval_every=5000,
    commitment_weight=0.25,
    vae_n_cat_feats=0,
    vae_input_dim=768,
    vae_embed_dim=32,
    vae_hidden_dims=[512, 256, 128],
    vae_codebook_size=256,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.ROTATION_TRICK,
    vae_sim_vq=False,
    vae_n_layers=3,
    collab_loss_weight=1.0,
    diversity_loss_weight=0.0,
    collab_temperature=0.1,
    collab_proj_dim=64,
    collab_embedding_dim=None,
    lightgcn_embed_dim=None,
    train_on_all_items=False,
):
    if collab_embeddings_path is None:
        raise ValueError("collab_embeddings_path must be set for train_tokenizer_with_cf.py.")
    if collab_embedding_dim is None and lightgcn_embed_dim is None:
        raise ValueError("Set collab_embedding_dim for the collaborative item embedding table.")
    if collab_embedding_dim is None:
        collab_embedding_dim = lightgcn_embed_dim
    elif lightgcn_embed_dim is not None and collab_embedding_dim != lightgcn_embed_dim:
        raise ValueError("collab_embedding_dim and lightgcn_embed_dim disagree; keep only one value or make them equal.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device

    train_split = "all" if train_on_all_items else "train"
    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split=train_split,
        split=dataset_split,
    )
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="eval",
            split=dataset_split,
        )
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    index_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )

    train_dataloader = accelerator.prepare(train_dataloader)

    rqvae = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None and resume_checkpoint_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
    )
    if pretrained_rqvae_path is not None:
        rqvae.load_pretrained(pretrained_rqvae_path)

    trainer = CollaborativeTokenizerTrainer(
        rq_vae=rqvae,
        collab_embeddings_path=collab_embeddings_path,
        collab_embedding_dim=collab_embedding_dim,
        collab_proj_dim=collab_proj_dim,
        collab_temperature=collab_temperature,
        collab_loss_weight=collab_loss_weight,
        diversity_loss_weight=diversity_loss_weight,
    )

    optimizer = AdamW(trainer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_iter = 0
    if resume_checkpoint_path is not None:
        state = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        trainer.load_state_dict(state["trainer"])
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    trainer, optimizer = accelerator.prepare(trainer, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=None,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = accelerator.unwrap_model(trainer).rq_vae

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        wandb.init(project="rq-vae-letter-training", config=params)

    with tqdm(initial=start_iter, total=start_iter + iterations, disable=not accelerator.is_main_process) as pbar:
        running_total, running_semantic, running_collab = [], [], []
        for iteration in range(start_iter, start_iter + iterations):
            trainer.train()
            total_loss = 0
            t = 0.2
            if iteration == 0 and use_kmeans_init and pretrained_rqvae_path is None and resume_checkpoint_path is None:
                kmeans_init_data = batch_to(train_dataset[torch.arange(min(20000, len(train_dataset)))], device)
                accelerator.unwrap_model(trainer).rq_vae(kmeans_init_data, t)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                with accelerator.autocast():
                    model_output = trainer(data, gumbel_t=t)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)
            optimizer.step()

            running_total.append(total_loss.detach().cpu().item())
            running_semantic.append(model_output.semantic_loss.detach().cpu().item())
            running_collab.append(model_output.collaborative_loss.detach().cpu().item())
            running_total = running_total[-200:]
            running_semantic = running_semantic[-200:]
            running_collab = running_collab[-200:]

            pbar.set_description(
                "loss: "
                f"{np.mean(running_total):.4f}, "
                f"sl: {np.mean(running_semantic):.4f}, "
                f"cl: {np.mean(running_collab):.4f}"
            )
            pbar.update(1)

            id_diversity_log = {}
            if accelerator.is_main_process and wandb_logging:
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                p_unique_ids = (
                    model_output.p_unique_ids.detach().cpu().item()
                    if isinstance(model_output.p_unique_ids, torch.Tensor)
                    else float(model_output.p_unique_ids)
                )
                wandb.log(
                    {
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.detach().cpu().item(),
                        "semantic_loss": model_output.semantic_loss.detach().cpu().item(),
                        "collab_loss": model_output.collaborative_loss.detach().cpu().item(),
                        "diversity_loss": model_output.diversity_loss.detach().cpu().item(),
                        "reconstruction_loss": model_output.reconstruction_loss.detach().cpu().item(),
                        "rqvae_loss": model_output.rqvae_loss.detach().cpu().item(),
                        "temperature": t,
                        "p_unique_ids": p_unique_ids,
                        **{f"emb_avg_norm_{i}": emb_norms_avg[i].detach().cpu().item() for i in range(vae_n_layers)},
                    },
                    step=iteration,
                )

            if do_eval and ((iteration + 1) % eval_every == 0 or iteration + 1 == start_iter + iterations):
                trainer.eval()
                eval_losses = {"total": [], "semantic": [], "collab": [], "diversity": []}
                with tqdm(eval_dataloader, desc=f"Eval {iteration + 1}", disable=True) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_output = trainer(data, gumbel_t=t)
                        eval_losses["total"].append(eval_output.loss.detach().cpu().item())
                        eval_losses["semantic"].append(eval_output.semantic_loss.detach().cpu().item())
                        eval_losses["collab"].append(eval_output.collaborative_loss.detach().cpu().item())
                        eval_losses["diversity"].append(eval_output.diversity_loss.detach().cpu().item())

                id_diversity_log["eval_total_loss"] = float(np.mean(eval_losses["total"]))
                id_diversity_log["eval_semantic_loss"] = float(np.mean(eval_losses["semantic"]))
                id_diversity_log["eval_collab_loss"] = float(np.mean(eval_losses["collab"]))
                id_diversity_log["eval_diversity_loss"] = float(np.mean(eval_losses["diversity"]))

            if accelerator.is_main_process:
                if (iteration + 1) % save_model_every == 0 or iteration + 1 == start_iter + iterations:
                    state = {
                        "iter": iteration,
                        "model": accelerator.unwrap_model(trainer).rq_vae.state_dict(),
                        "trainer": accelerator.unwrap_model(trainer).state_dict(),
                        "model_config": accelerator.unwrap_model(trainer).rq_vae.config,
                        "optimizer": optimizer.state_dict(),
                    }
                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)
                    torch.save(state, os.path.join(save_dir_root, f"checkpoint_{iteration}.pt"))

                if (iteration + 1) % eval_every == 0 or iteration + 1 == start_iter + iterations:
                    tokenizer.reset()
                    trainer.eval()
                    tokenizer.rq_vae = accelerator.unwrap_model(trainer).rq_vae
                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

                    _, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p * torch.log(p)).sum()

                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:, cid], return_counts=True)
                        id_diversity_log[f"codebook_usage_{cid}"] = len(counts) / vae_codebook_size

                    id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                    id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()

                if wandb_logging:
                    wandb.log(id_diversity_log, step=iteration)

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
