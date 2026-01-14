"""
Evaluation-only script for decoder checkpoints.

Uses the same gin config bindings as train_decoder.py (selector: "train"),
but only runs the full-eval Top-K metrics (no training).

Example:
  python eval_decoder.py --config configs/decoder_amazon_beauty_context2.gin \\
    --decoder-ckpt out/decoder/amazon_beauty_context2/beauty_context2_decoder_49999.pt \\
    --dataset-split beauty
"""

import argparse
import gin
import torch

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer


@gin.configurable("train")
def evaluate(
    # 训练步数与优化器超参数（评测不使用，但保持一致性）
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    # 数据路径与数据集选择
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    # 预训练模型路径
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    # 训练策略与加速相关
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    # 保存与评估频率（评测不使用）
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    # RQ‑VAE tokenizer 相关超参
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    # Decoder Transformer 超参
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    # 其他训练开关
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    save_name_prefix="checkpoint",
    # 方案1/2 上下文设置
    context1_enabled=False,
    context1_num_buckets=256,
    context1_source="user",
    context2_enabled=False,
    context2_codebook_layer=0
):
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if pretrained_decoder_path is None:
        raise ValueError("pretrained_decoder_path is required for evaluation.")

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    device = accelerator.device

    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split=dataset_split
    )

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    base_sem_id_dim = tokenizer.sem_ids_dim
    sem_id_dim = base_sem_id_dim + (1 if context2_enabled else 0)
    inference_verifier_fn = (
        (lambda x: tokenizer.exists_prefix(x[..., :base_sem_id_dim]))
        if context2_enabled else (lambda x: tokenizer.exists_prefix(x))
    )

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=inference_verifier_fn,
        sem_id_dim=sem_id_dim,
        base_sem_id_dim=base_sem_id_dim,
        context1_enabled=context1_enabled,
        context1_num_buckets=context1_num_buckets,
        context1_source=context1_source,
        context2_enabled=context2_enabled,
        context2_codebook=tokenizer.rq_vae.layers[context2_codebook_layer].embedding.weight if context2_enabled else None,
        context2_embed_dim=vae_embed_dim if context2_enabled else None,
        context2_rqvae_encoder=tokenizer.rq_vae.encoder if context2_enabled else None,
        context2_rqvae_input_dim=vae_input_dim if context2_enabled else None,
        max_pos=eval_dataset.max_seq_len * sem_id_dim,
        jagged_mode=model_jagged_mode
    )

    checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model = accelerator.prepare(model)
    model.eval()
    model.enable_generation = True

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    with torch.no_grad():
        for batch in eval_dataloader:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)

            generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
            actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
            if context2_enabled:
                actual = actual[:, :base_sem_id_dim]
                top_k = top_k[..., :base_sem_id_dim]

            metrics_accumulator.accumulate(actual=actual, top_k=top_k)

    eval_metrics = metrics_accumulator.reduce()
    print(eval_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to gin config file.")
    parser.add_argument("--decoder-ckpt", help="Path to decoder checkpoint.")
    parser.add_argument("--rqvae-ckpt", help="Path to RQ-VAE checkpoint.")
    parser.add_argument("--dataset-split", help="Dataset split override (e.g., beauty/beauty_small).")
    parser.add_argument("--dataset-folder", help="Dataset folder override.")
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    overrides = {}
    if args.decoder_ckpt:
        overrides["pretrained_decoder_path"] = args.decoder_ckpt
    if args.rqvae_ckpt:
        overrides["pretrained_rqvae_path"] = args.rqvae_ckpt
    if args.dataset_split:
        overrides["dataset_split"] = args.dataset_split
    if args.dataset_folder:
        overrides["dataset_folder"] = args.dataset_folder

    evaluate(**overrides)


if __name__ == "__main__":
    main()
