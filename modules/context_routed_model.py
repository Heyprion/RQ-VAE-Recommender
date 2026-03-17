import torch

from data.schemas import MultiAspectTokenizedSeqBatch
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.model import GenerationOutput
from modules.model import ModelOutput
from modules.multi_aspect_sid import ContextRouter
from modules.multi_aspect_sid import MultiAspectBranchEncoder
from modules.multi_aspect_sid import compute_diversity_loss
from modules.multi_aspect_sid import compute_orthogonality_loss
from modules.normalize import RMSNorm
from modules.rqvae import RqVae
from modules.transformer.model import TransformerEncoderDecoder
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class ContextRoutedModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor
    loss_ntp: Tensor
    loss_div: Tensor
    loss_orth: Tensor
    loss_router: Tensor
    router_scores: Tensor
    teacher_branch_idx: Tensor
    selected_branch_idx: Tensor
    target_branch_idx: Tensor


class TokenizedTrainingOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor
    loss_per_example: Tensor


class HistoryRoutingOutput(NamedTuple):
    history_full_ids: Tensor
    history_seq_mask: Tensor
    loss_router: Tensor
    loss_div: Tensor
    loss_orth: Tensor
    router_scores: Tensor
    teacher_branch_idx: Tensor
    selected_branch_idx: Tensor


class ContextRoutedEncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attn_dim: int,
        dropout: float,
        num_heads: int,
        n_layers: int,
        num_embeddings: int,
        sem_id_dim: int,
        inference_verifier_fn,
        rqvae: RqVae,
        num_aspects: int,
        router_hidden_dim: int,
        branch_hidden_dim: int = 0,
        loss_ntp_weight: float = 1.0,
        loss_div_weight: float = 0.0,
        loss_orth_weight: float = 0.0,
        loss_router_weight: float = 0.0,
        history_branch_index: int = 0,
        history_max_items: int = 20,
        history_branch_warmup: int = 1,
        max_pos: int = 2048,
        freeze_base_quantizer: bool = True,
        use_context_router: bool = True,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.base_sem_id_dim = sem_id_dim - 1
        self.num_aspects = num_aspects
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False
        self.history_branch_index = history_branch_index
        self.history_max_items = history_max_items
        self.history_branch_warmup = history_branch_warmup

        self.loss_ntp_weight = loss_ntp_weight
        self.loss_div_weight = loss_div_weight
        self.loss_orth_weight = loss_orth_weight
        self.loss_router_weight = loss_router_weight
        self.use_context_router = use_context_router

        self.rq_vae = rqvae
        self.rq_vae.eval()
        if freeze_base_quantizer:
            self.rq_vae.freeze_quantizer()
        self.rq_vae.requires_grad_(False)

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
            decoder_layers=n_layers // 2,
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

        self.branch_encoder = MultiAspectBranchEncoder(
            input_dim=rqvae.embed_dim,
            output_dim=rqvae.embed_dim,
            num_aspects=num_aspects,
            hidden_dim=branch_hidden_dim,
        )
        self.router = ContextRouter(
            history_dim=attn_dim,
            item_dim=rqvae.embed_dim,
            branch_dim=rqvae.embed_dim,
            hidden_dim=router_hidden_dim,
        )

    def _repeat_token_type_ids(self, batch_size: int, num_items: int, device: torch.device) -> Tensor:
        return torch.arange(self.sem_id_dim, device=device).repeat(batch_size, num_items)

    def _build_tokenized_batch(
        self,
        user_ids: Tensor,
        history_full_ids: Tensor,
        seq_mask: Tensor,
        target_full_ids: Tensor | None = None,
    ) -> TokenizedSeqBatch:
        batch_size, num_items, _ = history_full_ids.shape
        sem_ids = rearrange(history_full_ids, "b n d -> b (n d)")
        seq_mask_tokens = rearrange(history_full_ids >= 0, "b n d -> b (n d)")
        sem_ids = sem_ids.clone()
        sem_ids[~seq_mask_tokens] = -1

        token_type_ids = self._repeat_token_type_ids(batch_size, num_items, history_full_ids.device)
        if target_full_ids is None:
            sem_ids_fut = None
            token_type_ids_fut = None
        else:
            sem_ids_fut = target_full_ids
            token_type_ids_fut = torch.arange(target_full_ids.shape[1], device=history_full_ids.device).repeat(batch_size, 1)

        return TokenizedSeqBatch(
            user_ids=user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask_tokens,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )

    def _truncate_history_batch(self, batch: MultiAspectTokenizedSeqBatch) -> MultiAspectTokenizedSeqBatch:
        batch_size, seq_len = batch.seq_mask.shape
        truncated_len = min(seq_len, self.history_max_items)

        base_sem_ids = -1 * torch.ones(
            batch_size, truncated_len, batch.base_sem_ids.shape[-1],
            device=batch.base_sem_ids.device,
            dtype=batch.base_sem_ids.dtype
        )
        history_ids = -1 * torch.ones(
            batch_size, truncated_len,
            device=batch.history_ids.device,
            dtype=batch.history_ids.dtype
        )
        history_x = -1 * torch.ones(
            batch_size, truncated_len, batch.history_x.shape[-1],
            device=batch.history_x.device,
            dtype=batch.history_x.dtype
        )
        seq_mask = torch.zeros(
            batch_size, truncated_len,
            device=batch.seq_mask.device,
            dtype=torch.bool
        )

        for row in range(batch_size):
            valid_count = int(batch.seq_mask[row].sum().item())
            keep = min(valid_count, truncated_len)
            if keep == 0:
                continue

            start = valid_count - keep
            end = valid_count
            base_sem_ids[row, :keep] = batch.base_sem_ids[row, start:end, :]
            history_ids[row, :keep] = batch.history_ids[row, start:end]
            history_x[row, :keep] = batch.history_x[row, start:end, :]
            seq_mask[row, :keep] = True

        return MultiAspectTokenizedSeqBatch(
            user_ids=batch.user_ids,
            base_sem_ids=base_sem_ids,
            base_sem_ids_fut=batch.base_sem_ids_fut,
            seq_mask=seq_mask,
            history_ids=history_ids,
            future_ids=batch.future_ids,
            history_x=history_x,
            future_x=batch.future_x,
        )

    def _prepare_context_inputs(self, batch: TokenizedSeqBatch) -> tuple[Tensor, Tensor]:
        user_emb = self.user_id_embedder(batch.user_ids).unsqueeze(1)
        sem_ids_emb = self.sem_id_embedder(batch).seq
        seq_lengths = batch.seq_mask.sum(axis=1)

        num_tokens = sem_ids_emb.shape[1]
        pos = torch.arange(num_tokens, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], dim=1).contiguous()
        input_embedding = padded_to_jagged_tensor(
            input_embedding,
            lengths=seq_lengths + 1,
            max_len=input_embedding.shape[1]
        )
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        return transformer_context, seq_lengths

    def _prepare_future_inputs(self, batch: TokenizedSeqBatch) -> Tensor:
        batch_size = batch.user_ids.shape[0]
        input_embedding_fut = self.bos_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)

        if batch.sem_ids_fut is not None:
            sem_ids_emb_fut = self.sem_id_embedder(batch).fut
            tte_fut = self.tte_fut(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat([input_embedding_fut, sem_ids_emb_fut + tte_fut], dim=1)

        seq_lengths_fut = torch.full(
            (batch_size,),
            input_embedding_fut.shape[1],
            device=input_embedding_fut.device,
            dtype=torch.int64
        )
        input_embedding_fut = padded_to_jagged_tensor(
            input_embedding_fut.contiguous(),
            lengths=seq_lengths_fut,
            max_len=input_embedding_fut.shape[1]
        )
        return self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))

    def _encode_history_context(self, batch: TokenizedSeqBatch) -> tuple[Tensor, Tensor]:
        transformer_context, _ = self._prepare_context_inputs(batch)
        context = self.transformer.encoder(
            transformer_context,
            padding_mask=batch.seq_mask,
            is_causal=False,
            context=None,
            jagged=True
        )
        context_mask = torch.cat([
            torch.ones(batch.seq_mask.shape[0], 1, dtype=torch.bool, device=batch.seq_mask.device),
            batch.seq_mask
        ], dim=1)
        context_values = jagged_to_flattened_tensor(context)
        padded_context = torch.zeros(
            context_mask.shape[0],
            context_mask.shape[1],
            context_values.shape[-1],
            device=context_values.device,
            dtype=context_values.dtype
        )
        padded_context[context_mask] = context_values
        history_hidden = padded_context[:, 1:, :]
        history_mask = batch.seq_mask
        return history_hidden, history_mask

    def _predict_tokenized(self, batch: TokenizedSeqBatch) -> Tensor:
        transformer_context, _ = self._prepare_context_inputs(batch)
        transformer_input = self._prepare_future_inputs(batch)
        context = self.transformer.encoder(
            transformer_context,
            padding_mask=batch.seq_mask,
            is_causal=False,
            context=None,
            jagged=True
        )
        return self.transformer.decoder(
            transformer_input,
            padding_mask=None,
            is_causal=True,
            context=context,
            jagged=True
        )

    def _forward_tokenized(
        self,
        batch: TokenizedSeqBatch,
        generation_mode: bool = False,
    ) -> ModelOutput | TokenizedTrainingOutput:
        batch_size = batch.user_ids.shape[0]
        trnsf_out = self._predict_tokenized(batch)

        if generation_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=batch_size)[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            return ModelOutput(loss=None, logits=logits, loss_d=None)

        if batch.sem_ids_fut is None:
            raise Exception("Target semantic ids must be provided unless generation mode is enabled.")

        predict_out = self.out_proj(trnsf_out)
        logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=batch_size)[:, :-1, :]
        flat_logits = logits.flatten(end_dim=1)
        unred_loss = rearrange(
            F.cross_entropy(flat_logits, batch.sem_ids_fut.flatten(end_dim=1), reduction="none", ignore_index=-1),
            "(b n) -> b n",
            b=batch_size
        )
        loss_per_example = unred_loss.sum(dim=1)
        return TokenizedTrainingOutput(
            loss=loss_per_example.mean(),
            logits=flat_logits,
            loss_d=unred_loss.mean(dim=0),
            loss_per_example=loss_per_example,
        )

    def _compute_branch_candidates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        base_repr = self.rq_vae.encode(x)
        aspect_embs = self.branch_encoder(base_repr)
        flat_aspects = rearrange(aspect_embs, "b m d -> (b m) d")
        quantized = self.rq_vae.quantize_first_layer(flat_aspects)
        branch_ids = rearrange(quantized.ids, "(b m) -> b m", b=x.shape[0], m=self.num_aspects)
        branch_embs = rearrange(quantized.embeddings, "(b m) d -> b m d", b=x.shape[0], m=self.num_aspects)
        div_loss = compute_diversity_loss(aspect_embs)
        orth_loss = compute_orthogonality_loss(aspect_embs)
        return base_repr, branch_ids, branch_embs, div_loss, orth_loss

    def _gather_candidate_ids(self, candidate_ids: Tensor, indices: Tensor) -> Tensor:
        gather_idx = indices.view(-1, 1, 1).expand(-1, 1, candidate_ids.shape[-1])
        return torch.gather(candidate_ids, 1, gather_idx).squeeze(1)

    def _score_history_branches(
        self,
        user_ids: Tensor,
        prefix_full_ids: Tensor,
        prefix_mask: Tensor,
        base_repr: Tensor,
        branch_embs: Tensor,
    ) -> Tensor:
        if not self.use_context_router:
            return torch.zeros(
                branch_embs.shape[0],
                branch_embs.shape[1],
                device=branch_embs.device,
                dtype=branch_embs.dtype
            )

        history_batch = self._build_tokenized_batch(
            user_ids=user_ids,
            history_full_ids=prefix_full_ids,
            seq_mask=prefix_mask,
            target_full_ids=None,
        )
        history_hidden, history_mask = self._encode_history_context(history_batch)
        return self.router(history_hidden, history_mask, base_repr, branch_embs)

    def _compute_history_teacher_losses(
        self,
        user_ids: Tensor,
        prefix_full_ids: Tensor,
        prefix_mask: Tensor,
        current_base_ids: Tensor,
        candidate_branch_ids: Tensor,
        next_base_ids: Tensor,
    ) -> Tensor:
        batch_size = user_ids.shape[0]
        current_full_ids = torch.cat([
            current_base_ids.unsqueeze(1).expand(-1, self.num_aspects, -1),
            candidate_branch_ids.unsqueeze(-1)
        ], dim=-1)

        if prefix_full_ids.shape[1] == 0:
            candidate_history = current_full_ids.unsqueeze(2)
            candidate_mask = torch.ones(batch_size, self.num_aspects, 1, dtype=torch.bool, device=current_full_ids.device)
        else:
            candidate_history = torch.cat([
                prefix_full_ids.unsqueeze(1).expand(-1, self.num_aspects, -1, -1),
                current_full_ids.unsqueeze(2)
            ], dim=2)
            candidate_mask = torch.cat([
                prefix_mask.unsqueeze(1).expand(-1, self.num_aspects, -1),
                torch.ones(batch_size, self.num_aspects, 1, dtype=torch.bool, device=current_full_ids.device)
            ], dim=2)

        tokenized = self._build_tokenized_batch(
            user_ids=user_ids.repeat_interleave(self.num_aspects, dim=0),
            history_full_ids=rearrange(candidate_history, "b m n d -> (b m) n d"),
            seq_mask=rearrange(candidate_mask, "b m n -> (b m) n"),
            target_full_ids=rearrange(
                next_base_ids.unsqueeze(1).expand(-1, self.num_aspects, -1),
                "b m d -> (b m) d"
            ),
        )
        output = self._forward_tokenized(tokenized)
        assert isinstance(output, TokenizedTrainingOutput)
        return rearrange(output.loss_per_example, "(b m) -> b m", b=batch_size, m=self.num_aspects)

    def _get_next_base_ids(
        self,
        batch: MultiAspectTokenizedSeqBatch,
        position: int,
        active: Tensor,
    ) -> Tensor:
        next_base_ids = batch.base_sem_ids_fut[active]
        if position + 1 < batch.base_sem_ids.shape[1]:
            has_next_history = batch.seq_mask[active, position + 1]
            if has_next_history.any():
                next_base_ids = next_base_ids.clone()
                next_base_ids[has_next_history] = batch.base_sem_ids[active, position + 1, :][has_next_history]
        return next_base_ids

    def _reduce_losses(self, losses: list[Tensor], device: torch.device, dtype: torch.dtype) -> Tensor:
        if len(losses) == 0:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.stack(losses).mean()

    def _route_history(
        self,
        batch: MultiAspectTokenizedSeqBatch,
        use_teacher: bool,
    ) -> HistoryRoutingOutput:
        batch_size, seq_len = batch.seq_mask.shape
        history_full_ids = torch.cat([
            batch.base_sem_ids,
            -1 * torch.ones(batch_size, seq_len, 1, device=batch.base_sem_ids.device, dtype=batch.base_sem_ids.dtype)
        ], dim=-1)

        router_scores = torch.zeros(
            batch_size, seq_len, self.num_aspects,
            device=batch.base_sem_ids.device,
            dtype=torch.float32
        )
        teacher_branch_idx = -1 * torch.ones(
            batch_size, seq_len,
            device=batch.base_sem_ids.device,
            dtype=torch.long
        )
        selected_branch_idx = -1 * torch.ones_like(teacher_branch_idx)

        router_losses, div_losses, orth_losses = [], [], []

        for position in range(seq_len):
            active = batch.seq_mask[:, position]
            if not active.any():
                continue

            if position < self.history_branch_warmup:
                continue

            current_x = batch.history_x[active, position, :]
            current_base_ids = batch.base_sem_ids[active, position, :]
            prefix_full_ids = history_full_ids[active, :position, :]
            prefix_mask = batch.seq_mask[active, :position]

            base_repr, branch_ids, branch_embs, div_loss, orth_loss = self._compute_branch_candidates(current_x)
            scores = self._score_history_branches(
                user_ids=batch.user_ids[active],
                prefix_full_ids=prefix_full_ids,
                prefix_mask=prefix_mask,
                base_repr=base_repr,
                branch_embs=branch_embs,
            )
            next_base_ids = self._get_next_base_ids(batch, position, active)
            teacher_losses = self._compute_history_teacher_losses(
                user_ids=batch.user_ids[active],
                prefix_full_ids=prefix_full_ids,
                prefix_mask=prefix_mask,
                current_base_ids=current_base_ids,
                candidate_branch_ids=branch_ids,
                next_base_ids=next_base_ids,
            )
            teacher_idx = teacher_losses.argmin(dim=-1)
            chosen_idx = teacher_idx if use_teacher else scores.argmax(dim=-1)
            chosen_branch_ids = branch_ids.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)

            history_full_ids[active, position, -1] = chosen_branch_ids
            router_scores[active, position, :] = scores
            teacher_branch_idx[active, position] = teacher_idx
            selected_branch_idx[active, position] = chosen_idx

            div_losses.append(div_loss)
            orth_losses.append(orth_loss)
            if self.use_context_router:
                router_losses.append(F.cross_entropy(scores, teacher_idx))

        return HistoryRoutingOutput(
            history_full_ids=history_full_ids,
            history_seq_mask=batch.seq_mask,
            loss_router=self._reduce_losses(router_losses, batch.base_sem_ids.device, torch.float32),
            loss_div=self._reduce_losses(div_losses, batch.base_sem_ids.device, torch.float32),
            loss_orth=self._reduce_losses(orth_losses, batch.base_sem_ids.device, torch.float32),
            router_scores=router_scores,
            teacher_branch_idx=teacher_branch_idx,
            selected_branch_idx=selected_branch_idx,
        )

    def _build_future_candidate_ids(
        self,
        batch: MultiAspectTokenizedSeqBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _, branch_ids, _, div_loss, orth_loss = self._compute_branch_candidates(batch.future_x)
        full_ids = torch.cat([
            batch.base_sem_ids_fut.unsqueeze(1).expand(-1, self.num_aspects, -1),
            branch_ids.unsqueeze(-1)
        ], dim=-1)
        return full_ids, branch_ids, div_loss, orth_loss

    def _compute_target_candidate_losses(
        self,
        batch: MultiAspectTokenizedSeqBatch,
        history_full_ids: Tensor,
        candidate_full_ids: Tensor,
    ) -> Tensor:
        batch_size = batch.user_ids.shape[0]
        tokenized = self._build_tokenized_batch(
            user_ids=batch.user_ids.repeat_interleave(self.num_aspects, dim=0),
            history_full_ids=history_full_ids.repeat_interleave(self.num_aspects, dim=0),
            seq_mask=batch.seq_mask.repeat_interleave(self.num_aspects, dim=0),
            target_full_ids=rearrange(candidate_full_ids, "b m d -> (b m) d"),
        )
        output = self._forward_tokenized(tokenized)
        assert isinstance(output, TokenizedTrainingOutput)
        return rearrange(output.loss_per_example, "(b m) -> b m", b=batch_size, m=self.num_aspects)

    def _select_target_full_ids_from_history(
        self,
        batch: MultiAspectTokenizedSeqBatch,
        history_full_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        candidate_full_ids, _, div_loss, orth_loss = self._build_future_candidate_ids(batch)
        candidate_losses = self._compute_target_candidate_losses(batch, history_full_ids, candidate_full_ids)
        target_branch_idx = candidate_losses.argmin(dim=-1)
        target_full_ids = self._gather_candidate_ids(candidate_full_ids, target_branch_idx)
        return target_full_ids, target_branch_idx, div_loss, orth_loss

    def build_generation_history_batch(self, batch: MultiAspectTokenizedSeqBatch) -> TokenizedSeqBatch:
        batch = self._truncate_history_batch(batch)
        routing_output = self._route_history(batch, use_teacher=False)
        return self._build_tokenized_batch(
            user_ids=batch.user_ids,
            history_full_ids=routing_output.history_full_ids,
            seq_mask=routing_output.history_seq_mask,
            target_full_ids=None,
        )

    @torch.no_grad
    def select_target_full_ids(self, batch: MultiAspectTokenizedSeqBatch) -> Tensor:
        batch = self._truncate_history_batch(batch)
        routing_output = self._route_history(batch, use_teacher=False)
        target_full_ids, _, _, _ = self._select_target_full_ids_from_history(
            batch=batch,
            history_full_ids=routing_output.history_full_ids,
        )
        return target_full_ids

    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        assert self.enable_generation, "Model generation is not enabled"

        batch_size = batch.sem_ids.shape[0]
        generated, log_probas = None, 0
        n_top_k_candidates = min(self.num_embeddings, 256) if top_k else 1
        k = min(64, n_top_k_candidates) if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self._forward_tokenized(input_batch, generation_mode=True).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = torch.cat([
                    generated.flatten(0, 1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1),
                    samples_batched.unsqueeze(-1)
                ], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(batch_size, -1)

            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(batch_size, -1)
            samples = samples_batched.reshape(batch_size, -1)

            sorted_log_probas, sorted_indices = (
                -10000 * (~is_valid_prefix) +
                sampled_log_probas +
                (log_probas.repeat_interleave(n_top_k_candidates, dim=1) if isinstance(log_probas, Tensor) else 0)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)

            if generated is not None:
                parent_id = torch.gather(
                    generated,
                    1,
                    (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1, -1, i)
                )
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)
                next_sem_ids = top_k_samples.flatten(end_dim=1)
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

            input_batch = TokenizedSeqBatch(
                user_ids=input_batch.user_ids.repeat_interleave(k, dim=0) if generated is None else input_batch.user_ids,
                sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0) if generated is None else input_batch.sem_ids,
                sem_ids_fut=next_sem_ids,
                token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0) if generated is None else input_batch.seq_mask,
                token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0) if generated is None else input_batch.token_type_ids
            )

            generated = top_k_samples.unsqueeze(-1) if generated is None else torch.clone(top_k_samples.detach())
            log_probas = torch.clone(top_k_log_probas.detach())

        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )

    def forward(self, batch: MultiAspectTokenizedSeqBatch) -> ContextRoutedModelOutput:
        batch = self._truncate_history_batch(batch)
        routing_output = self._route_history(batch, use_teacher=self.training)
        target_full_ids, target_branch_idx, target_div, target_orth = self._select_target_full_ids_from_history(
            batch=batch,
            history_full_ids=routing_output.history_full_ids,
        )

        selected_tokenized = self._build_tokenized_batch(
            user_ids=batch.user_ids,
            history_full_ids=routing_output.history_full_ids,
            seq_mask=routing_output.history_seq_mask,
            target_full_ids=target_full_ids,
        )
        ntp_output = self._forward_tokenized(selected_tokenized)
        assert isinstance(ntp_output, TokenizedTrainingOutput)

        loss_div = torch.stack([routing_output.loss_div, target_div]).mean()
        loss_orth = torch.stack([routing_output.loss_orth, target_orth]).mean()
        total_loss = (
            self.loss_ntp_weight * ntp_output.loss +
            self.loss_div_weight * loss_div +
            self.loss_orth_weight * loss_orth +
            self.loss_router_weight * routing_output.loss_router
        )

        return ContextRoutedModelOutput(
            loss=total_loss,
            logits=ntp_output.logits,
            loss_d=ntp_output.loss_d,
            loss_ntp=ntp_output.loss,
            loss_div=loss_div,
            loss_orth=loss_orth,
            loss_router=routing_output.loss_router,
            router_scores=routing_output.router_scores,
            teacher_branch_idx=routing_output.teacher_branch_idx,
            selected_branch_idx=routing_output.selected_branch_idx,
            target_branch_idx=target_branch_idx,
        )
