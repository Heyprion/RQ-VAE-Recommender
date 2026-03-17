import torch

from data.schemas import TokenizedSeqBatch
from torch import nn
from torch import Tensor
from typing import NamedTuple


class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim*num_embeddings
        
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings*self.sem_ids_dim+1,
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )

    def _expand_token_type_ids(self, sem_ids: Tensor, token_type_ids: Tensor = None) -> Tensor:
        if token_type_ids is not None:
            return token_type_ids

        sem_ids_dim = sem_ids.shape[-1]
        view_shape = [1] * sem_ids.dim()
        view_shape[-1] = sem_ids_dim
        return torch.arange(sem_ids_dim, device=sem_ids.device).view(*view_shape).expand_as(sem_ids)

    def embed_sem_ids(self, sem_ids: Tensor, token_type_ids: Tensor = None) -> Tensor:
        token_type_ids = self._expand_token_type_ids(sem_ids, token_type_ids)
        input_ids = token_type_ids * self.num_embeddings + sem_ids.clamp(min=0)
        input_ids = input_ids.masked_fill(sem_ids < 0, self.padding_idx)
        return self.emb(input_ids)
    
    def forward(self, batch: TokenizedSeqBatch) -> Tensor:
        sem_ids = batch.sem_ids.clone()
        sem_ids[~batch.seq_mask] = -1

        if batch.sem_ids_fut is not None:
            sem_ids_fut = self.embed_sem_ids(batch.sem_ids_fut, batch.token_type_ids_fut)
        else:
            sem_ids_fut = None
        return SemIdEmbeddingBatch(
            seq=self.embed_sem_ids(sem_ids, batch.token_type_ids),
            fut=sem_ids_fut
        ) 
    

class UserIdEmbedder(nn.Module):
    # TODO: Implement hashing trick embedding for user id
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        hashed_indices = x % self.num_buckets
        # hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)
