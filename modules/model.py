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
        max_pos=2048,
        jagged_mode: bool = True,
        use_multi_sid: bool = False,
        use_sid_hard_selection: bool = False,
        sid_selection_context_len: int = 20,
        sid_selection_min_history: int = 1,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False
        self.use_multi_sid = use_multi_sid
        self.use_sid_hard_selection = use_sid_hard_selection
        self.sid_selection_context_len = sid_selection_context_len
        self.sid_selection_min_history = sid_selection_min_history

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

    def _get_user_emb(self, user_ids: Tensor) -> Tensor:
        user_emb = self.user_id_embedder(user_ids)
        if user_emb.dim() == 2:
            user_emb = user_emb.unsqueeze(1)
        return user_emb

    def _get_projected_user_context(self, user_ids: Tensor) -> Tensor:
        user_emb = self._get_user_emb(user_ids)
        return self.in_proj_context(self.do(self.norm(user_emb))).squeeze(1)

    def _repeat_optional(self, x: Tensor, repeats: int) -> Tensor:
        if x is None:
            return None
        return x.repeat_interleave(repeats, dim=0)

    def _encode_history_context(
        self,
        user_ids: Tensor,
        selected_sid: Tensor,
        item_seq_mask: Tensor
    ) -> Tensor:
        if selected_sid.shape[1] == 0:
            return self._get_projected_user_context(user_ids)

        B, num_items, sem_id_dim = selected_sid.shape
        seq_mask = item_seq_mask.repeat_interleave(sem_id_dim, dim=1)
        sem_ids = rearrange(selected_sid, "b n d -> b (n d)").clone()
        sem_ids[~seq_mask] = -1
        token_type_ids = torch.arange(sem_id_dim, device=sem_ids.device).repeat(B, num_items)

        sem_ids_emb = self.sem_id_embedder.embed_sem_ids(sem_ids, token_type_ids)
        user_emb = self._get_user_emb(user_ids)

        pos = torch.arange(sem_ids_emb.shape[1], device=sem_ids_emb.device).unsqueeze(0)
        input_embedding = torch.cat([user_emb, self.wpe(pos) + sem_ids_emb], axis=1)

        if self.jagged_mode:
            seq_lengths = seq_mask.sum(axis=1) + 1
            input_embedding = padded_to_jagged_tensor(
                input_embedding,
                lengths=seq_lengths,
                max_len=input_embedding.shape[1]
            )
            transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
            encoder_output = self.transformer.encoder(
                transformer_context,
                padding_mask=seq_mask,
                is_causal=False,
                context=None,
                jagged=True
            )
            flattened_encoder_output = jagged_to_flattened_tensor(encoder_output)
            offsets = encoder_output.offsets().to(torch.long)
            return flattened_encoder_output[offsets[1:] - 1]

        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        mem_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=item_seq_mask.device),
            seq_mask
        ], axis=1)
        encoder_output = self.transformer.encoder(
            transformer_context,
            src_key_padding_mask=~mem_mask
        )
        row_idx = torch.arange(B, device=encoder_output.device)
        seq_lengths = mem_mask.sum(dim=1).to(torch.long) - 1
        return encoder_output[row_idx, seq_lengths, :]

    @torch.no_grad()
    def _apply_history_sid_selection(self, batch: TokenizedSeqBatch) -> TokenizedSeqBatch:
        if not (
            self.use_multi_sid and
            self.use_sid_hard_selection and
            batch.multi_sids is not None and
            batch.item_seq_mask is not None
        ):
            return batch

        B, num_items, _, sem_id_dim = batch.multi_sids.shape
        device = batch.sem_ids.device
        row_idx = torch.arange(B, device=device)

        selected_sid = batch.multi_sids[:, :, 0, :].clone()
        selected_sid_idx = torch.zeros(B, num_items, dtype=torch.long, device=device)

        sid_embs = self.sem_id_embedder.embed_sem_ids(batch.multi_sids)[..., :-1, :].sum(dim=-2)
        selected_sid_emb = torch.zeros(
            B,
            num_items,
            sid_embs.shape[-1],
            dtype=sid_embs.dtype,
            device=sid_embs.device
        )

        for item_pos in range(num_items):
            context_start = max(0, item_pos - self.sid_selection_context_len)
            context_mask = batch.item_seq_mask[:, context_start:item_pos]
            valid_history_count = context_mask.sum(dim=1)
            use_canonical = valid_history_count < self.sid_selection_min_history
            current_selected_sid_idx = torch.zeros(B, dtype=torch.long, device=device)

            if not use_canonical.all():
                context_repr = self._encode_history_context(
                    user_ids=batch.user_ids,
                    selected_sid=selected_sid[:, context_start:item_pos, :],
                    item_seq_mask=context_mask
                )
                current_sid_embs = sid_embs[:, item_pos, :, :]
                current_sid_repr = self.in_proj_context(self.do(self.norm(current_sid_embs)))
                attention_scores = (context_repr.unsqueeze(1) * current_sid_repr).sum(dim=-1)
                attention_probs = torch.softmax(attention_scores, dim=-1)
                current_selected_sid_idx = attention_probs.argmax(dim=-1)
                current_selected_sid_idx = torch.where(
                    use_canonical,
                    torch.zeros_like(current_selected_sid_idx),
                    current_selected_sid_idx
                )
            else:
                current_sid_embs = sid_embs[:, item_pos, :, :]

            selected_sid_idx[:, item_pos] = current_selected_sid_idx
            selected_sid[:, item_pos, :] = batch.multi_sids[row_idx, item_pos, current_selected_sid_idx, :]
            selected_sid_emb[:, item_pos, :] = current_sid_embs[row_idx, current_selected_sid_idx, :]

            invalid_rows = ~batch.item_seq_mask[:, item_pos]
            if invalid_rows.any():
                selected_sid_idx[invalid_rows, item_pos] = 0
                selected_sid[invalid_rows, item_pos, :] = -1
                selected_sid_emb[invalid_rows, item_pos, :] = 0

        selected_sem_ids = rearrange(selected_sid, "b n d -> b (n d)")
        selected_seq_mask = batch.item_seq_mask.repeat_interleave(sem_id_dim, dim=1)
        selected_sem_ids[~selected_seq_mask] = -1
        selected_token_type_ids = torch.arange(sem_id_dim, device=device).repeat(B, num_items)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=selected_sem_ids,
            seq_mask=selected_seq_mask,
            token_type_ids=selected_token_type_ids,
            sem_ids_fut=batch.sem_ids_fut,
            token_type_ids_fut=batch.token_type_ids_fut,
            item_ids=batch.item_ids,
            item_ids_fut=batch.item_ids_fut,
            item_seq_mask=batch.item_seq_mask,
            multi_sids=None,
            multi_sids_fut=batch.multi_sids_fut
        )

    def _expand_multi_target_batch(self, batch: TokenizedSeqBatch) -> tuple[TokenizedSeqBatch, int]:
        num_targets = batch.multi_sids_fut.shape[1]
        expanded_sem_ids_fut = rearrange(batch.multi_sids_fut, "b m d -> (b m) d")
        expanded_token_type_ids_fut = torch.arange(
            expanded_sem_ids_fut.shape[1],
            device=expanded_sem_ids_fut.device
        ).repeat(expanded_sem_ids_fut.shape[0], 1)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids.repeat_interleave(num_targets, dim=0),
            sem_ids=batch.sem_ids.repeat_interleave(num_targets, dim=0),
            seq_mask=batch.seq_mask.repeat_interleave(num_targets, dim=0),
            token_type_ids=batch.token_type_ids.repeat_interleave(num_targets, dim=0),
            sem_ids_fut=expanded_sem_ids_fut,
            token_type_ids_fut=expanded_token_type_ids_fut,
            item_ids=self._repeat_optional(batch.item_ids, num_targets),
            item_ids_fut=self._repeat_optional(batch.item_ids_fut, num_targets),
            item_seq_mask=self._repeat_optional(batch.item_seq_mask, num_targets),
            multi_sids=None,
            multi_sids_fut=None
        ), num_targets
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self._get_user_emb(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)
          
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

            seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        else:
            mem_mask = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
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
        batch = self._apply_history_sid_selection(batch)

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 64 if top_k else 1
        n_top_k_candidates = self.num_embeddings if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            sem_ids_fut=None,
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
                    cache = torch.zeros(input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim, device=input_batch.sem_ids.device)
                    cache_mask = torch.cat([torch.ones(input_batch.sem_ids.shape[0], 1, dtype=bool, device=input_batch.seq_mask.device), input_batch.seq_mask], axis=1)
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
        
        sem_ids_out = generated[:, 0, :] if generated.dim() == 3 and generated.shape[1] == 1 else generated
        log_probas_out = log_probas[:, 0] if log_probas.dim() == 2 and log_probas.shape[1] == 1 else log_probas

        return GenerationOutput(
            sem_ids=sem_ids_out,
            log_probas=log_probas_out
        )
            
    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        batch = self._apply_history_sid_selection(batch)
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape
        use_multi_target_loss = (
            self.use_multi_sid and
            batch.multi_sids_fut is not None and
            batch.multi_sids_fut.shape[1] > 1 and
            batch.sem_ids_fut is not None
        )

        predict_batch = batch
        predict_batch_size = B
        num_targets = 1
        if (self.training or not self.enable_generation) and use_multi_target_loss:
            predict_batch, num_targets = self._expand_multi_target_batch(batch)
            predict_batch_size = predict_batch.seq_mask.shape[0]

        trnsf_out = self._predict(predict_batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits_by_pos = rearrange(
                    jagged_to_flattened_tensor(predict_out),
                    "(b n) d -> b n d",
                    b=predict_batch_size
                )[:, :-1, :]
                logits = logits_by_pos.flatten(end_dim=1)
                target = predict_batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(
                    F.cross_entropy(logits, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=predict_batch_size
                )
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = predict_batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(
                    F.cross_entropy(out, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=predict_batch_size
                )

            if use_multi_target_loss:
                candidate_nll = unred_loss.sum(axis=1).reshape(B, num_targets)
                loss = -torch.logsumexp(-candidate_nll, dim=1).mean()
                loss_d = unred_loss.reshape(B, num_targets, -1).mean(axis=1).mean(axis=0)
                if self.jagged_mode:
                    logits = logits_by_pos[:, :, :].reshape(B, num_targets, -1, logits_by_pos.shape[-1])[:, 0, :, :].flatten(end_dim=1)
            else:
                loss = unred_loss.sum(axis=1).mean()
                loss_d = unred_loss.mean(axis=0)

            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
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
