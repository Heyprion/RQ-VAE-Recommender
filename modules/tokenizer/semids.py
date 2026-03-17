import math
import torch

from data.processed import ItemData
from data.processed import SeqData
from data.schemas import MultiAspectTokenizedSeqBatch
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.rqvae import RqVae
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16

class SemanticIdTokenizer(nn.Module):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False,
        use_multi_aspect_sid: bool = False,
    ) -> None:
        super().__init__()

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=rqvae_codebook_normalize,
            codebook_sim_vq=rqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
        )
        
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.use_multi_aspect_sid = use_multi_aspect_sid
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
        self.cached_base_ids = None
    
    @property
    def sem_ids_dim(self):
        return self.n_layers + 1
    
    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        if self.use_multi_aspect_sid:
            cached_base_ids = []
            sampler = BatchSampler(
                SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
            )
            dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
            for batch in dataloader:
                batch = batch_to(batch, self.rq_vae.device)
                batch_ids = self.rq_vae.get_base_semantic_ids(batch.x)
                cached_base_ids.append(batch_ids)

            self.cached_base_ids = torch.cat(cached_base_ids, dim=0)
            return self.cached_base_ids

        cached_ids = None
        dedup_dim = []
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def precompute_multi_aspect_corpus_ids(
        self,
        movie_dataset: ItemData,
        branch_encoder: nn.Module,
        gumbel_t: float = 0.001,
    ) -> Tensor:
        if not self.use_multi_aspect_sid:
            raise Exception("Multi-aspect corpus ids requested while tokenizer is in baseline mode.")

        candidate_ids = []
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            batch = batch_to(batch, self.rq_vae.device)
            base_ids = self.rq_vae.get_base_semantic_ids(batch.x, gumbel_t=gumbel_t)
            base_repr = self.rq_vae.encode(batch.x)
            aspect_embs = branch_encoder(base_repr)
            flat_aspects = rearrange(aspect_embs, "b m d -> (b m) d")
            branch_ids = self.rq_vae.quantize_first_layer(flat_aspects, gumbel_t=gumbel_t).ids
            branch_ids = rearrange(branch_ids, "(b m) -> b m", b=base_ids.shape[0])
            full_ids = torch.cat([
                base_ids.unsqueeze(1).expand(-1, branch_ids.shape[1], -1),
                branch_ids.unsqueeze(-1)
            ], dim=-1)
            candidate_ids.append(rearrange(full_ids, "b m d -> (b m) d"))

        self.cached_ids = torch.cat(candidate_ids, dim=0)
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # Batch prefixes matching to avoid OOM. 
        batches = math.ceil(sem_id_prefix.shape[0] / BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])

    def _lookup_cached_base_ids(self, ids: Tensor) -> Tensor:
        if self.cached_base_ids is None:
            raise Exception("Base ids cache is empty.")

        out = -1 * torch.ones(*ids.shape, self.n_layers, device=ids.device, dtype=torch.long)
        valid = ids >= 0
        if valid.any():
            out[valid] = self.cached_base_ids[ids[valid]]
        return out

    def _compute_base_ids_from_features(self, x: Tensor) -> Tensor:
        x_shape = x.shape[:-1]
        x_flat = rearrange(x, "... d -> (...) d")
        valid = (x_flat != -1).all(dim=-1)

        base_ids = -1 * torch.ones(x_flat.shape[0], self.n_layers, device=x.device, dtype=torch.long)
        if valid.any():
            base_ids[valid] = self.rq_vae.get_base_semantic_ids(x_flat[valid])

        return base_ids.reshape(*x_shape, self.n_layers)

    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch | MultiAspectTokenizedSeqBatch:
        if self.use_multi_aspect_sid:
            history_ids = batch.ids
            future_ids = batch.ids_fut.squeeze(-1)
            future_x = batch.x_fut.squeeze(1)

            if self.cached_base_ids is None or history_ids.max() >= self.cached_base_ids.shape[0]:
                base_sem_ids = self._compute_base_ids_from_features(batch.x)
                base_sem_ids_fut = self._compute_base_ids_from_features(future_x)
            else:
                base_sem_ids = self._lookup_cached_base_ids(history_ids)
                base_sem_ids_fut = self._lookup_cached_base_ids(batch.ids_fut).squeeze(1)

            return MultiAspectTokenizedSeqBatch(
                user_ids=batch.user_ids,
                base_sem_ids=base_sem_ids,
                base_sem_ids_fut=base_sem_ids_fut,
                seq_mask=batch.seq_mask,
                history_ids=history_ids,
                future_ids=future_ids,
                history_x=batch.x,
                future_x=future_x
            )

        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
        
        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )

if __name__ == "__main__":
    dataset = ItemData("dataset/ml-1m-movie")
    tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
    tokenizer.precompute_corpus_ids(dataset)
    
    seq_data = SeqData("dataset/ml-1m")
    batch = seq_data[:10]
    tokenized = tokenizer(batch)
    import pdb; pdb.set_trace()
