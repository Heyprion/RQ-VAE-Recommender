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
from modules.context_routed_model import ContextRoutedEncoderDecoderRetrievalModel
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
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    use_multi_aspect_sid=False,
    use_context_router=True,
    num_aspects=4,
    router_hidden_dim=128,
    branch_hidden_dim=0,
    history_max_items=20,
    history_branch_warmup=1,
    freeze_base_quantizer=True,
    loss_ntp_weight=1.0,
    loss_div_weight=0.0,
    loss_orth_weight=0.0,
    loss_router_weight=0.0,
    stage1_rqvae_path=None,
    stage2_resume_path=None,
):  
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    if stage1_rqvae_path is not None:
        pretrained_rqvae_path = stage1_rqvae_path
    if stage2_resume_path is not None:
        pretrained_decoder_path = stage2_resume_path
    if use_multi_aspect_sid and not use_context_router:
        loss_router_weight = 0.0

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
        split=dataset_split
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
        use_multi_aspect_sid=use_multi_aspect_sid,
    )
    tokenizer = tokenizer.to(device)
    tokenizer.precompute_corpus_ids(item_dataset)
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    if use_multi_aspect_sid:
        model = ContextRoutedEncoderDecoderRetrievalModel(
            embedding_dim=decoder_embed_dim,
            attn_dim=attn_embed_dim,
            dropout=dropout_p,
            num_heads=attn_heads,
            n_layers=attn_layers,
            num_embeddings=vae_codebook_size,
            inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
            sem_id_dim=tokenizer.sem_ids_dim,
            max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim,
            rqvae=tokenizer.rq_vae,
            num_aspects=num_aspects,
            router_hidden_dim=router_hidden_dim,
            branch_hidden_dim=branch_hidden_dim,
            history_max_items=history_max_items,
            history_branch_warmup=history_branch_warmup,
            loss_ntp_weight=loss_ntp_weight,
            loss_div_weight=loss_div_weight,
            loss_orth_weight=loss_orth_weight,
            loss_router_weight=loss_router_weight,
            freeze_base_quantizer=freeze_base_quantizer,
            use_context_router=use_context_router,
        )
    else:
        model = EncoderDecoderRetrievalModel(
            embedding_dim=decoder_embed_dim,
            attn_dim=attn_embed_dim,
            dropout=dropout_p,
            num_heads=attn_heads,
            n_layers=attn_layers,
            num_embeddings=vae_codebook_size,
            inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
            sem_id_dim=tokenizer.sem_ids_dim,
            max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
            jagged_mode=model_jagged_mode
        )

    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

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
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    if use_multi_aspect_sid:
                        valid_history = model_output.teacher_branch_idx >= 0
                        teacher_branch_idx = model_output.teacher_branch_idx[valid_history]
                        selected_branch_idx = model_output.selected_branch_idx[valid_history]
                        train_debug_metrics = {
                            "loss_ntp": model_output.loss_ntp.detach().cpu().item(),
                            "loss_div": model_output.loss_div.detach().cpu().item(),
                            "loss_orth": model_output.loss_orth.detach().cpu().item(),
                            "loss_router": model_output.loss_router.detach().cpu().item(),
                            "history_teacher_branch_idx_mean": teacher_branch_idx.to(torch.float32).mean().detach().cpu().item() if teacher_branch_idx.numel() > 0 else -1.0,
                            "history_selected_branch_idx_mean": selected_branch_idx.to(torch.float32).mean().detach().cpu().item() if selected_branch_idx.numel() > 0 else -1.0,
                            "target_branch_idx_mean": model_output.target_branch_idx.to(torch.float32).mean().detach().cpu().item(),
                        }
                    else:
                        train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(total_loss)
                assert accelerator.unwrap_model(model).sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter+1) % partial_eval_every == 0:
                model.eval()
                accelerator.unwrap_model(model).enable_generation = False
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        if use_multi_aspect_sid:
                            valid_history = model_output_eval.teacher_branch_idx >= 0
                            teacher_branch_idx = model_output_eval.teacher_branch_idx[valid_history]
                            selected_branch_idx = model_output_eval.selected_branch_idx[valid_history]
                            eval_debug_metrics = {
                                "eval_loss": model_output_eval.loss.detach().cpu().item(),
                                "eval_loss_ntp": model_output_eval.loss_ntp.detach().cpu().item(),
                                "eval_loss_div": model_output_eval.loss_div.detach().cpu().item(),
                                "eval_loss_orth": model_output_eval.loss_orth.detach().cpu().item(),
                                "eval_loss_router": model_output_eval.loss_router.detach().cpu().item(),
                                "eval_history_teacher_branch_idx_mean": teacher_branch_idx.to(torch.float32).mean().detach().cpu().item() if teacher_branch_idx.numel() > 0 else -1.0,
                                "eval_history_selected_branch_idx_mean": selected_branch_idx.to(torch.float32).mean().detach().cpu().item() if selected_branch_idx.numel() > 0 else -1.0,
                                "eval_target_branch_idx_mean": model_output_eval.target_branch_idx.to(torch.float32).mean().detach().cpu().item(),
                            }
                        else:
                            eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                            eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                        wandb.log(eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                model.eval()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.enable_generation = True
                if use_multi_aspect_sid:
                    tokenizer.precompute_multi_aspect_corpus_ids(item_dataset, unwrapped_model.branch_encoder)
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        if use_multi_aspect_sid:
                            history_batch = unwrapped_model.build_generation_history_batch(tokenized_data)
                            generated = unwrapped_model.generate_next_sem_id(history_batch, top_k=True, temperature=1)
                            actual, top_k = unwrapped_model.select_target_full_ids(tokenized_data), generated.sem_ids
                        else:
                            generated = unwrapped_model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                            actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                        if accelerator.is_main_process and wandb_logging and not use_multi_aspect_sid:
                            wandb.log(eval_debug_metrics)
                
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if wandb_logging:
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
