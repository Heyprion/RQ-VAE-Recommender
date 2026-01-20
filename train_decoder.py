"""
训练解码器（生成式检索模型）的主脚本。

核心流程概览：
1) 构建数据集：物品语料 + 用户序列。
2) 加载冻结的 RQ‑VAE tokenizer，把物品/序列映射为语义 ID。
3) 训练 Transformer 解码器，预测“下一步语义 ID”序列。
4) 周期性评估：部分评估（loss），完整评估（Top‑K 命中率）。
5) 保存 checkpoint，可选记录到 W&B。
"""

import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    # 训练步数与优化器超参数
    iterations=500000,  # 总训练迭代步数
    batch_size=64,  # 每步的 batch 大小
    learning_rate=0.001,  # AdamW 学习率
    weight_decay=0.01,  # AdamW 权重衰减
    # 数据路径与数据集选择
    dataset_folder="dataset/ml-1m",  # 数据集根目录（会读取/写入 processed）
    save_dir_root="out/",  # checkpoint 保存目录
    dataset=RecDataset.ML_1M,  # 数据集枚举（当前仅支持 AMAZON）
    # 预训练模型路径
    pretrained_rqvae_path=None,  # 预训练 RQ‑VAE 权重路径（用于 tokenizer）
    pretrained_decoder_path=None,  # 预训练 decoder 权重路径（用于续训/微调）
    # 训练策略与加速相关
    split_batches=True,  # accelerate 是否拆分 batch（多卡时）
    amp=False,  # 是否启用混合精度
    wandb_logging=False,  # 是否启用 W&B 记录
    force_dataset_process=False,  # 是否强制重新处理数据
    mixed_precision_type="fp16",  # 混精类型（fp16/bf16）
    gradient_accumulate_every=1,  # 梯度累积步数
    # 保存与评估频率
    save_model_every=1000000,  # checkpoint 保存频率（步）
    partial_eval_every=1000,  # 仅算 loss 的评估频率
    full_eval_every=10000,  # 开启生成的完整评估频率
    # RQ‑VAE tokenizer 相关超参（需与预训练模型一致）
    vae_input_dim=18,  # 物品特征输入维度
    vae_embed_dim=16,  # RQ‑VAE 隐空间维度
    vae_hidden_dims=[18, 18],  # RQ‑VAE 编码器/解码器隐藏层
    vae_codebook_size=32,  # 每层 codebook 大小
    vae_codebook_normalize=False,  # 是否对 codebook 向量归一化
    vae_sim_vq=False,  # 是否启用 SimVQ 变体
    vae_n_cat_feats=18,  # 类别特征数量（0 表示纯连续）
    vae_n_layers=3,  # 量化层数（语义 ID 维度 = n_layers + 1）
    # Decoder Transformer 超参
    decoder_embed_dim=64,  # 语义 ID embedding 维度
    dropout_p=0.1,  # Dropout 概率
    attn_heads=8,  # 注意力头数
    attn_embed_dim=64,  # 注意力内部维度
    attn_layers=4,  # Transformer 层数（编码器/解码器各一半）
    # 其他训练开关
    dataset_split="beauty",  # Amazon 数据集分支（beauty/sports/toys）
    push_vae_to_hf=False,  # 是否将 RQ‑VAE 推送到 HuggingFace
    train_data_subsample=True,  # 是否对训练序列做随机子序列采样
    model_jagged_mode=True,  # 是否使用 jagged（变长）注意力实现
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",  # 推送到 HF 的模型名
    save_name_prefix="checkpoint",  # 保存权重文件名前缀
    context1_enabled=False,  # 是否启用方案1：上下文作为额外条件 token
    context1_num_buckets=256,  # 方案1上下文桶数量
    context1_source="user",  # 方案1上下文来源：user/seq_len
    context2_enabled=False,  # 是否启用方案2：主SID + 动态上下文ID
    context2_codebook_layer=0,  # 方案2上下文ID使用的 RQ‑VAE codebook 层索引
    text_encoder="t5",  # item文本向量化方法：t5/tfidf
    tfidf_max_features=50000,  # TF-IDF词表大小上限
    tfidf_svd_dim=768  # TF-IDF降维维度
):  
    # 当前解码器训练仅支持 Amazon 数据集（其他数据集路径尚未接通）
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        # 记录当前函数全部超参，用于 W&B 配置页显示
        params = locals()

    # Accelerator 负责混合精度与多卡封装
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        # 仅主进程登录 W&B，避免多进程重复初始化
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    # item_dataset: 用于构建全量语义 ID 缓存（语料库）
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
        text_encoder=text_encoder,
        tfidf_max_features=tfidf_max_features,
        tfidf_svd_dim=tfidf_svd_dim
    )
    # train_dataset: 训练用用户序列
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
        split=dataset_split,
        text_encoder=text_encoder,
        tfidf_max_features=tfidf_max_features,
        tfidf_svd_dim=tfidf_svd_dim
    )
    # eval_dataset: 评估用用户序列
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split,
        text_encoder=text_encoder,
        tfidf_max_features=tfidf_max_features,
        tfidf_svd_dim=tfidf_svd_dim
    )

    # 普通 DataLoader（batch_size 由 gin 配置）
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 训练用无限迭代器，便于按 step 训练
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    # 交给 accelerate 进行设备/多卡封装
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # RQ‑VAE tokenizer：将物品特征序列 -> 语义 ID 序列
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
    # tokenizer 也放到 accelerator 管理的设备上
    tokenizer = accelerator.prepare(tokenizer)
    # 预计算语料库中的语义 ID，用于快速匹配与前缀校验
    tokenizer.precompute_corpus_ids(item_dataset)
    
    if push_vae_to_hf:
        # 可选：把 RQ‑VAE 权重推送到 HuggingFace Hub
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    # Transformer 编解码模型：预测下一个语义 ID token
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
        max_pos=train_dataset.max_seq_len*sem_id_dim,
        jagged_mode=model_jagged_mode
    )

    # 解码器参数优化器
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Transformer 常用的逆平方根学习率调度
    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        # 从已有 decoder checkpoint 恢复训练
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    # 模型与优化器交给 accelerator 管理（支持多卡/混精度）
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    # Top‑K 评价指标累积器
    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                # 取一批用户序列数据
                data = next_batch(train_dataloader, device)
                # 通过 tokenizer 转换为语义 ID 序列
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    # 前向：预测下一步语义 ID
                    model_output = model(tokenized_data)
                    # 梯度累积时做平均
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    # 记录每个语义 ID 位置的损失
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(total_loss)
                # 确保语义 ID embedding 有梯度（用于排查训练异常）
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            # 参数更新 + 学习率调度
            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            saved_final = False
            if accelerator.is_main_process and (iter+1) == iterations:
                # Save final checkpoint before eval to avoid losing it if eval crashes.
                state = {
                    "iter": iter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict()
                }

                if not os.path.exists(save_dir_root):
                    os.makedirs(save_dir_root)

                torch.save(state, save_dir_root + f"{save_name_prefix}_{iter}.pt")
                saved_final = True

            if (iter+1) % partial_eval_every == 0:
                # 部分评估：只算 loss，不做生成
                model.eval()
                model.enable_generation = False
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        # 记录 eval loss 与分位置损失
                        eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                        wandb.log(eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                # 完整评估：开启自回归生成，统计 Top‑K 命中
                model.eval()
                model.enable_generation = True
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                        if context2_enabled:
                            actual = actual[:, :base_sem_id_dim]
                            top_k = top_k[..., :base_sem_id_dim]

                        # 累积 Top‑K 指标
                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                        if accelerator.is_main_process and wandb_logging:
                            wandb.log(eval_debug_metrics)
                
                # 汇总并打印 Top‑K 指标
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    # 将指标写入 W&B
                    wandb.log(eval_metrics)
                
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if ((iter+1) % save_model_every == 0 or iter+1 == iterations) and not saved_final:
                    # 保存训练状态（便于续训）
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"{save_name_prefix}_{iter}.pt")
                
                if wandb_logging:
                    # 记录训练步的学习率与 loss
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    # 通过 gin 配置文件注入 train() 参数
    parse_config()
    train()
