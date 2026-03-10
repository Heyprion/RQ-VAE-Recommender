import torch

from data.schemas import SeqBatch
from modules.loss import CodebookDiversityLoss
from modules.loss import ContrastiveAlignmentLoss
from modules.rqvae import RqVae
from typing import NamedTuple
from torch import Tensor
from torch import nn


class CollaborativeTokenizerOutput(NamedTuple):
    loss: Tensor
    semantic_loss: Tensor
    collaborative_loss: Tensor
    diversity_loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    quantized_embeddings: Tensor
    sem_ids: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class CollaborativeTokenizerTrainer(nn.Module):
    def __init__(
        self,
        rq_vae: RqVae,
        collab_embeddings_path: str,
        lightgcn_embed_dim: int,
        collab_proj_dim: int,
        collab_temperature: float = 0.1,
        collab_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.rq_vae = rq_vae
        self.collab_loss_weight = collab_loss_weight
        self.diversity_loss_weight = diversity_loss_weight

        payload = torch.load(collab_embeddings_path, map_location="cpu", weights_only=False)
        if payload["embedding_dim"] != lightgcn_embed_dim:
            raise ValueError(
                f"Expected LightGCN embedding dim {lightgcn_embed_dim}, found {payload['embedding_dim']} in {collab_embeddings_path}."
            )
        self.register_buffer("collab_item_embeddings", payload["item_embeddings"].to(torch.float32), persistent=False)
        self.register_buffer("collab_item_mask", payload["item_mask"].to(torch.bool), persistent=False)

        self.semantic_projector = nn.Linear(rq_vae.embed_dim, collab_proj_dim, bias=False)
        self.collab_projector = nn.Linear(lightgcn_embed_dim, collab_proj_dim, bias=False)
        self.collaborative_loss = ContrastiveAlignmentLoss(temperature=collab_temperature)
        self.diversity_loss = CodebookDiversityLoss(codebook_size=rq_vae.codebook_size)

    def _lookup_collaborative_embeddings(self, item_ids: Tensor) -> tuple[Tensor, Tensor]:
        if item_ids.max() >= self.collab_item_embeddings.shape[0]:
            raise IndexError(
                f"Found item id {item_ids.max().item()} but only {self.collab_item_embeddings.shape[0]} collaborative embeddings were exported."
            )

        item_ids = item_ids.to(torch.long)
        return self.collab_item_embeddings[item_ids], self.collab_item_mask[item_ids]

    def forward(self, batch: SeqBatch, gumbel_t: float = 0.001) -> CollaborativeTokenizerOutput:
        rqvae_output = self.rq_vae.encode_with_quantized_embeddings(batch.x, gumbel_t=gumbel_t)
        reconstruction_loss = self.rq_vae.reconstruction_loss(rqvae_output.reconstruction, batch.x)
        rqvae_loss = rqvae_output.quantize_loss
        semantic_loss = (reconstruction_loss + rqvae_loss).mean()

        collab_targets, collab_mask = self._lookup_collaborative_embeddings(batch.ids.view(-1))

        collaborative_loss = self.collaborative_loss(
            self.semantic_projector(rqvae_output.quantized_sum),
            self.collab_projector(collab_targets),
            valid_mask=collab_mask,
        )

        diversity_loss = self.diversity_loss(rqvae_output.sem_ids)
        total_loss = semantic_loss + self.collab_loss_weight * collaborative_loss
        if self.diversity_loss_weight != 0:
            total_loss = total_loss + self.diversity_loss_weight * diversity_loss
        else:
            diversity_loss = diversity_loss.detach() * 0

        with torch.no_grad():
            embs_norm = rqvae_output.embeddings.norm(dim=1)
            p_unique_ids = torch.unique(rqvae_output.sem_ids, dim=0).shape[0] / rqvae_output.sem_ids.shape[0]

        return CollaborativeTokenizerOutput(
            loss=total_loss,
            semantic_loss=semantic_loss,
            collaborative_loss=collaborative_loss,
            diversity_loss=diversity_loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            quantized_embeddings=rqvae_output.quantized_sum,
            sem_ids=rqvae_output.sem_ids,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
