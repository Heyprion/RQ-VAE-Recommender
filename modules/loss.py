import math
import torch

from torch import nn
from torch import Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats
    
    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if valid_mask is None:
            valid_mask = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)

        query = query[valid_mask]
        target = target[valid_mask]
        if query.shape[0] <= 1:
            return query.new_zeros(())

        query = nn.functional.normalize(query, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        logits = query @ target.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_q = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.T, labels)
        return 0.5 * (loss_q + loss_t)


class CodebookDiversityLoss(nn.Module):
    def __init__(self, codebook_size: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.eps = eps

    def forward(self, sem_ids: Tensor) -> Tensor:
        if sem_ids.numel() == 0:
            return sem_ids.new_zeros(())

        losses = []
        max_entropy = math.log(self.codebook_size)
        for layer_idx in range(sem_ids.shape[1]):
            ids = sem_ids[:, layer_idx]
            valid = ids >= 0
            ids = ids[valid]
            if ids.numel() == 0:
                continue

            counts = torch.bincount(ids, minlength=self.codebook_size).to(torch.float32)
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * torch.log(probs + self.eps)).sum()
            losses.append(1.0 - entropy / max(max_entropy, self.eps))

        if len(losses) == 0:
            return sem_ids.new_zeros(())
        return torch.stack(losses).mean()
