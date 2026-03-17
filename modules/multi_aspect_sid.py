import math
import torch

from torch import nn
from torch import Tensor


class MultiAspectBranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_aspects: int,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_aspects = num_aspects
        self.hidden_dim = hidden_dim

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            ) if hidden_dim > 0 else nn.Linear(input_dim, output_dim)
            for _ in range(num_aspects)
        ])

    def forward(self, x: Tensor) -> Tensor:
        aspect_embs = [head(x) for head in self.heads]
        return torch.stack(aspect_embs, dim=1)


class ContextRouter(nn.Module):
    def __init__(
        self,
        history_dim: int,
        item_dim: int,
        branch_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.history_dim = history_dim
        self.item_dim = item_dim
        self.branch_dim = branch_dim

        self.query_proj = nn.Linear(branch_dim, history_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(history_dim * 2 + item_dim + branch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_hidden: Tensor,
        history_mask: Tensor,
        base_repr: Tensor,
        branch_repr: Tensor,
    ) -> Tensor:
        num_aspects = branch_repr.shape[1]
        base_repr_exp = base_repr.unsqueeze(1).expand(-1, num_aspects, -1)

        query = self.query_proj(branch_repr)
        if history_hidden.shape[1] == 0:
            context = torch.zeros_like(query)
            score_input = torch.cat([context, query, base_repr_exp, branch_repr], dim=-1)
            return self.score_mlp(score_input).squeeze(-1)

        attn_scores = torch.einsum("bmd,btd->bmt", query, history_hidden) / math.sqrt(self.history_dim)
        attn_scores = attn_scores.masked_fill(~history_mask.unsqueeze(1), float("-inf"))

        no_history_rows = ~history_mask.any(dim=1)
        if no_history_rows.any():
            attn_scores[no_history_rows] = 0.0

        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.einsum("bmt,btd->bmd", attn, history_hidden)

        score_input = torch.cat([context, query, base_repr_exp, branch_repr], dim=-1)
        return self.score_mlp(score_input).squeeze(-1)


def compute_diversity_loss(aspect_embs: Tensor) -> Tensor:
    if aspect_embs.shape[1] <= 1:
        return torch.zeros((), device=aspect_embs.device, dtype=aspect_embs.dtype)

    norm_aspects = torch.nn.functional.normalize(aspect_embs, dim=-1)
    sim = torch.einsum("bmd,bnd->bmn", norm_aspects, norm_aspects)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).unsqueeze(0)
    penalty = (sim - eye).pow(2)
    off_diag_mask = ~eye.bool().expand_as(penalty)
    return penalty.masked_select(off_diag_mask).mean()


def compute_orthogonality_loss(aspect_embs: Tensor) -> Tensor:
    if aspect_embs.shape[1] <= 1:
        return torch.zeros((), device=aspect_embs.device, dtype=aspect_embs.dtype)

    aspect_norm = torch.nn.functional.normalize(aspect_embs, dim=-1)
    gram = torch.matmul(aspect_norm, aspect_norm.transpose(-1, -2))
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return (gram - eye).pow(2).mean()
