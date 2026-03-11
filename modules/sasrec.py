import torch

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple


class SasRecOutput(NamedTuple):
    hidden_states: Tensor
    loss: Tensor


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SasRecBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = PointWiseFeedForward(hidden_size, dropout)

    def forward(self, x: Tensor, padding_mask: Tensor, causal_mask: Tensor) -> Tensor:
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = residual + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SasRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SasRecBlock(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def encode(self, seq: Tensor) -> Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        hidden = self.item_embedding(seq) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        padding_mask = seq.eq(0)
        causal_mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, padding_mask=padding_mask, causal_mask=causal_mask)
        return self.final_norm(hidden)

    def compute_loss(self, hidden_states: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        pos_emb = self.item_embedding(pos)
        neg_emb = self.item_embedding(neg)

        pos_logits = (hidden_states * pos_emb).sum(dim=-1)
        neg_logits = (hidden_states * neg_emb).sum(dim=-1)

        valid_mask = pos.ne(0)
        if not valid_mask.any():
            return hidden_states.new_zeros(())

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits[valid_mask],
            torch.ones_like(pos_logits[valid_mask]),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits[valid_mask],
            torch.zeros_like(neg_logits[valid_mask]),
        )
        return pos_loss + neg_loss

    def forward(self, seq: Tensor, pos: Tensor, neg: Tensor) -> SasRecOutput:
        hidden_states = self.encode(seq)
        return SasRecOutput(
            hidden_states=hidden_states,
            loss=self.compute_loss(hidden_states, pos=pos, neg=neg),
        )

    def get_item_embeddings(self) -> Tensor:
        return self.item_embedding.weight[1:]
