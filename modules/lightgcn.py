import torch

from typing import NamedTuple
from torch import Tensor
from torch import nn


class LightGCNOutput(NamedTuple):
    user_embeddings: Tensor
    item_embeddings: Tensor


def build_normalized_adjacency(
    num_users: int,
    num_items: int,
    edge_index: Tensor,
    device: torch.device | None = None,
) -> Tensor:
    if edge_index.numel() == 0:
        raise ValueError("LightGCN requires at least one training interaction.")

    if device is None:
        device = edge_index.device

    users = edge_index[0].to(device=device, dtype=torch.long)
    items = edge_index[1].to(device=device, dtype=torch.long) + num_users

    src = torch.cat([users, items], dim=0)
    dst = torch.cat([items, users], dim=0)
    indices = torch.stack([src, dst], dim=0)
    values = torch.ones(indices.shape[1], device=device)

    num_nodes = num_users + num_items
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1.0)
    norm_values = adjacency.values() * degree[adjacency.indices()[0]].pow(-0.5) * degree[adjacency.indices()[1]].pow(-0.5)
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        norm_values,
        size=adjacency.shape,
        device=device,
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    @property
    def device(self) -> torch.device:
        return self.user_embedding.weight.device

    def compute_embeddings(self, adjacency: Tensor) -> LightGCNOutput:
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_per_layer = [all_embeddings]
        propagated = all_embeddings

        for _ in range(self.n_layers):
            propagated = torch.sparse.mm(adjacency, propagated)
            embeddings_per_layer.append(propagated)

        stacked = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
        user_embeddings, item_embeddings = torch.split(stacked, [self.num_users, self.num_items], dim=0)
        return LightGCNOutput(user_embeddings=user_embeddings, item_embeddings=item_embeddings)

    def score(self, user_embeddings: Tensor, item_embeddings: Tensor) -> Tensor:
        return (user_embeddings * item_embeddings).sum(dim=-1)

    def bpr_loss(
        self,
        adjacency: Tensor,
        users: Tensor,
        pos_items: Tensor,
        neg_items: Tensor,
    ) -> tuple[Tensor, Tensor]:
        embeddings = self.compute_embeddings(adjacency)
        user_emb = embeddings.user_embeddings[users]
        pos_emb = embeddings.item_embeddings[pos_items]
        neg_emb = embeddings.item_embeddings[neg_items]

        pos_scores = self.score(user_emb, pos_emb)
        neg_scores = self.score(user_emb, neg_emb)
        ranking_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (
            self.user_embedding(users).pow(2).sum(dim=-1) +
            self.item_embedding(pos_items).pow(2).sum(dim=-1) +
            self.item_embedding(neg_items).pow(2).sum(dim=-1)
        ).mean()
        return ranking_loss, reg_loss
