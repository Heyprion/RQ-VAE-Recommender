import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        base_sem_id_dim=None,
        context1_enabled: bool = False,
        context1_num_buckets: int = 0,
        context1_source: str = "user",
        context2_enabled: bool = False,
        context2_codebook: Tensor | None = None,
        context2_embed_dim: int | None = None,
        context2_rqvae_encoder: nn.Module | None = None,
        context2_rqvae_input_dim: int | None = None,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.base_sem_id_dim = base_sem_id_dim or sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False
        self.context1_enabled = context1_enabled
        self.context2_enabled = context2_enabled
        if self.context1_enabled:
            if context1_num_buckets <= 0:
                raise ValueError("context1_num_buckets must be > 0 when context1_enabled is True.")
            self.context1_num_buckets = context1_num_buckets
            self.context1_source = context1_source
            self.context1_embedder = nn.Embedding(context1_num_buckets, embedding_dim)

        if self.context2_enabled:
            if context2_codebook is None or context2_embed_dim is None:
                raise ValueError("context2_codebook and context2_embed_dim are required when context2_enabled is True.")
            # Freeze codebook weights for context2 ID quantization.
            self.register_buffer("context2_codebook", context2_codebook.detach().clone())
            if context2_rqvae_encoder is not None:
                if context2_rqvae_input_dim is None:
                    raise ValueError("context2_rqvae_input_dim is required when context2_rqvae_encoder is provided.")
                self.context2_adapter = nn.Linear(attn_dim, context2_rqvae_input_dim, bias=False)
                self.context2_encoder = context2_rqvae_encoder
                for p in self.context2_encoder.parameters():
                    p.requires_grad = False
                self.context2_proj = None
            else:
                self.context2_proj = nn.Linear(attn_dim, context2_embed_dim, bias=False)
                self.context2_adapter = None
                self.context2_encoder = None

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,
            decoder_layers=n_layers // 2
        ) if self.jagged_mode else nn.Transformer(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)
    
    def _context2_quantize(self, x: Tensor) -> Tensor:
        # Nearest-neighbor lookup in the shared RQ-VAE codebook.
        codebook = self.context2_codebook
        dist = (
            (x**2).sum(axis=-1, keepdim=True) +
            (codebook.T**2).sum(axis=0, keepdim=True) -
            2 * x @ codebook.T
        )
        return dist.min(axis=-1).indices

    @torch.no_grad
    def _compute_context2_ids(self, sem_ids: Tensor, seq_mask_item: Tensor, token_type_ids: Tensor, user_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute per-item and next-item context2 IDs from a transformer encoder pass.
        Returns:
          context_ids: (B, N) per-item IDs for history
          context_id_fut: (B,) ID for next-item context
        """
        # Build embeddings from stable semantic IDs (no context ID).
        base_batch = TokenizedSeqBatch(
            user_ids=user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=None,
            seq_mask=seq_mask_item.repeat_interleave(self.base_sem_id_dim, dim=1),
            token_type_ids=token_type_ids,
            token_type_ids_fut=None
        )
        sem_ids_emb = self.sem_id_embedder(base_batch).seq

        B, N, _ = sem_ids_emb.shape
        seq_lengths_tokens = seq_mask_item.sum(axis=1) * self.base_sem_id_dim
        pos = torch.arange(sem_ids_emb.shape[1], device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)
        user_emb = self.user_id_embedder(base_batch.user_ids)
        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding = self.in_proj_context(self.do(self.norm(input_embedding)))

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(
                input_embedding, lengths=seq_lengths_tokens + 1, max_len=input_embedding.shape[1]
            )
            enc_out = self.transformer.encoder(
                x=input_embedding, padding_mask=seq_mask_item.repeat_interleave(self.base_sem_id_dim, dim=1), is_causal=False, context=None, jagged=True
            )
        else:
            # Unjagged attention not supported by current implementation.
            raise Exception("Context ID computation requires jagged_mode=True.")

        values = enc_out.values()
        offsets = enc_out.offsets()

        context_ids = torch.zeros(seq_mask_item.shape, device=sem_ids.device, dtype=torch.long)
        context_id_fut = torch.zeros(sem_ids.shape[0], device=sem_ids.device, dtype=torch.long)

        for b in range(B):
            seq_len_items = int(seq_mask_item[b].sum().item())
            if seq_len_items == 0:
                continue
            start = int(offsets[b].item())
            end = int(offsets[b + 1].item())
            token_len = seq_len_items * self.base_sem_id_dim
            # Drop user token at position 0.
            seq_tokens = values[start + 1:start + 1 + token_len]
            seq_tokens = seq_tokens.reshape(seq_len_items, self.base_sem_id_dim, self.attn_dim)
            item_ctx = seq_tokens.mean(dim=1)
            if self.context2_encoder is not None:
                proj = self.context2_encoder(self.context2_adapter(item_ctx))
            else:
                proj = self.context2_proj(item_ctx)
            ids = self._context2_quantize(proj)
            context_ids[b, :seq_len_items] = ids

            fut_ctx = item_ctx.mean(dim=0, keepdim=True)
            if self.context2_encoder is not None:
                fut_proj = self.context2_encoder(self.context2_adapter(fut_ctx))
            else:
                fut_proj = self.context2_proj(fut_ctx)
            context_id_fut[b] = self._context2_quantize(fut_proj).squeeze(0)

        return context_ids, context_id_fut

    def _augment_with_context2_ids(self, batch: TokenizedSeqBatch) -> TokenizedSeqBatch:
        # Build item-level mask from token-level mask.
        seq_mask_item = batch.seq_mask[:, ::self.base_sem_id_dim]
        N = seq_mask_item.shape[1]
        sem_ids = rearrange(batch.sem_ids, "b (n d) -> b n d", d=self.base_sem_id_dim)

        token_type_ids = torch.arange(self.base_sem_id_dim, device=sem_ids.device).repeat(sem_ids.shape[0], N)
        context_ids, context_id_fut = self._compute_context2_ids(
            sem_ids=rearrange(sem_ids, "b n d -> b (n d)"),
            seq_mask_item=seq_mask_item,
            token_type_ids=token_type_ids,
            user_ids=batch.user_ids
        )

        sem_ids = torch.cat([sem_ids, context_ids.unsqueeze(-1)], dim=2)
        sem_ids_fut = batch.sem_ids_fut
        if sem_ids_fut is not None and (self.training or not self.enable_generation):
            sem_ids_fut = torch.cat([sem_ids_fut, context_id_fut.unsqueeze(1)], dim=1)

        sem_ids = rearrange(sem_ids, "b n d -> b (n d)")

        token_type_ids = torch.arange(self.base_sem_id_dim + 1, device=sem_ids.device).repeat(sem_ids.shape[0], N)
        token_type_ids_fut = (
            torch.arange(sem_ids_fut.shape[1], device=sem_ids.device).repeat(sem_ids.shape[0], 1)
            if sem_ids_fut is not None else None
        )
        seq_mask = seq_mask_item.repeat_interleave(self.base_sem_id_dim + 1, dim=1)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )

    def _compute_context1_ids(self, batch: TokenizedSeqBatch) -> Tensor:
        if self.context1_source == "user":
            ids = batch.user_ids % self.context1_num_buckets
        elif self.context1_source == "seq_len":
            seq_mask_item = batch.seq_mask[:, ::self.base_sem_id_dim]
            seq_len = seq_mask_item.sum(axis=1).to(torch.long)
            ids = torch.minimum(
                seq_len,
                torch.tensor(self.context1_num_buckets - 1, device=seq_len.device)
            )
        else:
            raise ValueError(f"Unsupported context1_source: {self.context1_source}")
        return ids

    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)
          
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        prefix_tokens = [user_emb]
        if self.context1_enabled:
            context1_ids = self._compute_context1_ids(batch)
            context1_emb = self.context1_embedder(context1_ids)
            if context1_emb.dim() == 2:
                context1_emb = context1_emb.unsqueeze(1)
            prefix_tokens.append(context1_emb)
        input_embedding = torch.cat(prefix_tokens + [wpe + sem_ids_emb], axis=1)
        prefix_len = len(prefix_tokens)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+prefix_len, max_len=input_embedding.shape[1])

            seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        else:
            mem_mask = torch.cat([
                torch.ones(B, prefix_len, dtype=torch.bool, device=batch.seq_mask.device),
                batch.seq_mask
            ], axis=1)
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")
        
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        
        if self.jagged_mode:
            transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input.shape[1])
            transformer_output = self.transformer(src=transformer_context, tgt=transformer_input, tgt_is_causal=True, tgt_mask=causal_mask, src_key_padding_mask=f_mask, memory_key_padding_mask=f_mask)

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 64 if top_k else 1
        n_top_k_candidates = 256 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)
            
            if generated is not None:
                parent_id = torch.gather(generated, 1, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1,-1,i))
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)
                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions 
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache_batch = input_batch
                    if self.context2_enabled:
                        cache_batch = self._augment_with_context2_ids(input_batch)

                    prefix_len = 1 + (1 if self.context1_enabled else 0)
                    cache = torch.zeros(
                        cache_batch.sem_ids.shape[0],
                        cache_batch.sem_ids.shape[1] + prefix_len,
                        self.attn_dim,
                        device=cache_batch.sem_ids.device
                    )
                    cache_mask = torch.cat(
                        [torch.ones(cache_batch.sem_ids.shape[0], prefix_len, dtype=bool, device=cache_batch.seq_mask.device),
                         cache_batch.seq_mask],
                        axis=1
                    )
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = self.transformer.cached_enc_output.offsets().diff().repeat_interleave(k)
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        if self.context2_enabled:
            batch = self._augment_with_context2_ids(batch)
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(out, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
