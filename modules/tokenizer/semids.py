import math
import torch

from data.processed import ItemData
from data.processed import SeqData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
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
        use_multi_sid: bool = False,
        num_sid_heads: int = 1,
        lambda_orth: float = 0.0,
        lambda_bal: float = 0.0,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.use_multi_sid = use_multi_sid and max(1, num_sid_heads) > 1
        self.num_sid_heads = max(1, num_sid_heads) if self.use_multi_sid else 1
        self._num_embeddings = self.semantic_vocab_size

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
            use_multi_sid=self.use_multi_sid,
            num_sid_heads=self.num_sid_heads,
            lambda_orth=lambda_orth,
            lambda_bal=lambda_bal,
        )
        
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
        self.cached_multi_sids = None
        self.cached_legal_ids = None
        self.cached_tuple_keys = None
        self.cached_tuple_item_ids = None
    
    @property
    def sem_ids_dim(self):
        return self.n_layers + 1

    @property
    def semantic_vocab_size(self) -> int:
        return self.codebook_size * self.num_sid_heads

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    def _lookup_cache(self, cache: Tensor, ids: Tensor) -> Tensor:
        safe_ids = ids.clamp(min=0)
        return cache[safe_ids.reshape(-1)]

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(-1)
        cached_ids = self._lookup_cache(self.cached_ids, ids)
        return rearrange(cached_ids, "(b n) d -> b (n d)", n=ids.shape[1])

    def _tokenize_multi_sid_batch_from_cached(self, ids: Tensor) -> Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(-1)
        cached_multi_sids = self._lookup_cache(self.cached_multi_sids, ids)
        return rearrange(cached_multi_sids, "(b n) m d -> b n m d", n=ids.shape[1])

    def _tokenize_multi_sid_targets_from_cached(self, ids: Tensor) -> Tensor:
        cached_multi_sids = self._lookup_cache(self.cached_multi_sids, ids)
        if ids.dim() == 1:
            return cached_multi_sids
        return rearrange(cached_multi_sids, "(b n) m d -> b n m d", n=ids.shape[1]).squeeze(1)

    def _offset_multi_sids(self, multi_sids: Tensor) -> Tensor:
        if not self.use_multi_sid:
            return multi_sids

        head_offsets = (
            torch.arange(self.num_sid_heads, device=multi_sids.device)
            .view(1, self.num_sid_heads, 1) * self.codebook_size
        )
        offset_multi_sids = multi_sids.clone()
        offset_multi_sids = offset_multi_sids + head_offsets
        return offset_multi_sids

    def _tokenize_items_from_rqvae(self, batch: SeqBatch) -> tuple[Tensor, Tensor]:
        tokenized = self.rq_vae.get_semantic_ids(batch.x)
        multi_sids = tokenized.multi_sids
        if self.use_multi_sid:
            multi_sids = self._offset_multi_sids(multi_sids)
        return tokenized.sem_ids, multi_sids

    def _compute_tuple_keys(self, sem_ids: Tensor) -> Tensor:
        sem_ids = sem_ids.to(torch.long)
        position_bases = torch.full(
            (sem_ids.shape[-1],),
            fill_value=self.semantic_vocab_size + 1,
            device=sem_ids.device,
            dtype=torch.long
        )
        position_bases[-1] = self.num_embeddings + 1

        tuple_keys = sem_ids[..., 0]
        for pos in range(1, sem_ids.shape[-1]):
            tuple_keys = tuple_keys * position_bases[pos] + sem_ids[..., pos]
        return tuple_keys

    def _build_tuple_lookup_cache(self) -> None:
        if self.use_multi_sid:
            self.cached_legal_ids = rearrange(self.cached_multi_sids, "i m d -> (i m) d")
            item_ids = torch.arange(self.cached_multi_sids.shape[0], device=self.cached_multi_sids.device)
            tuple_item_ids = item_ids.repeat_interleave(self.num_sid_heads)
        else:
            self.cached_legal_ids = self.cached_ids
            tuple_item_ids = torch.arange(self.cached_ids.shape[0], device=self.cached_ids.device)

        tuple_keys = self._compute_tuple_keys(self.cached_legal_ids)
        sorted_keys, sort_idx = tuple_keys.sort()
        self.cached_tuple_keys = sorted_keys
        self.cached_tuple_item_ids = tuple_item_ids[sort_idx]

        if sorted_keys.shape[0] > 1:
            assert not (sorted_keys[1:] == sorted_keys[:-1]).any(), "Legal SID tuples must uniquely map to one item."

    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        cached_multi_sids = []
        dedup_dim = []
        cached_legal_ids = None
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            _, batch_multi_sids = self._tokenize_items_from_rqvae(batch_to(batch, self.rq_vae.device))
            flat_batch_multi_sids = rearrange(batch_multi_sids, "b m d -> (b m) d")

            # Detect duplicates across all legal head tuples.
            is_hit = self._get_hits(flat_batch_multi_sids, flat_batch_multi_sids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0

            if cached_legal_ids is None:
                cached_legal_ids = flat_batch_multi_sids.clone()
            else:
                is_hit = self._get_hits(flat_batch_multi_sids, cached_legal_ids)
                hits += is_hit.sum(axis=-1)
                cached_legal_ids = torch.cat([cached_legal_ids, flat_batch_multi_sids], dim=0)

            cached_multi_sids.append(batch_multi_sids)
            dedup_dim.append(rearrange(hits, "(b m) -> b m", m=self.num_sid_heads))

        cached_multi_sids = torch.cat(cached_multi_sids, dim=0)
        dedup_dim_tensor = torch.cat(dedup_dim, dim=0)
        if self.use_multi_sid:
            self._num_embeddings = max(
                self.semantic_vocab_size,
                int(dedup_dim_tensor.max().item()) + 1
            )
        else:
            self._num_embeddings = self.semantic_vocab_size

        self.cached_multi_sids = torch.cat([cached_multi_sids, dedup_dim_tensor.unsqueeze(-1)], dim=-1)
        self.cached_ids = self.cached_multi_sids[:, 0, :]
        self._build_tuple_lookup_cache()
        
        return self.cached_ids

    def sem_ids_to_item_ids(self, sem_ids: Tensor) -> Tensor:
        if self.cached_tuple_keys is None or self.cached_tuple_item_ids is None:
            raise Exception("Tuple lookup cache is empty. Call precompute_corpus_ids first.")

        tuple_keys = self._compute_tuple_keys(sem_ids)
        flat_tuple_keys = tuple_keys.reshape(-1)
        lookup_idx = torch.searchsorted(self.cached_tuple_keys, flat_tuple_keys)
        valid_lookup = lookup_idx < self.cached_tuple_keys.shape[0]

        matched_item_ids = torch.full(
            flat_tuple_keys.shape,
            fill_value=-1,
            dtype=torch.long,
            device=flat_tuple_keys.device
        )

        if valid_lookup.any():
            valid_positions = valid_lookup.nonzero().flatten()
            valid_lookup_idx = lookup_idx[valid_lookup]
            exact_match = self.cached_tuple_keys[valid_lookup_idx] == flat_tuple_keys[valid_lookup]
            if exact_match.any():
                matched_item_ids[valid_positions[exact_match]] = self.cached_tuple_item_ids[valid_lookup_idx[exact_match]]

        return matched_item_ids.reshape(tuple_keys.shape)

    def unique_item_topk(self, sem_ids: Tensor) -> Tensor:
        predicted_item_ids = self.sem_ids_to_item_ids(sem_ids)
        if predicted_item_ids.dim() == 1:
            return predicted_item_ids

        unique_item_ids = torch.full_like(predicted_item_ids, fill_value=-1)
        for row_idx in range(predicted_item_ids.shape[0]):
            seen_items = set()
            out_col = 0
            for item_id in predicted_item_ids[row_idx].tolist():
                if item_id < 0 or item_id in seen_items:
                    continue
                seen_items.add(item_id)
                unique_item_ids[row_idx, out_col] = item_id
                out_col += 1
                if out_col == unique_item_ids.shape[1]:
                    break

        return unique_item_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_legal_ids[:, :prefix_length] if self.use_multi_sid else self.cached_ids[:, :prefix_length]
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # Batch prefixes matching to avoid OOM. 
        batches = math.ceil(sem_id_prefix.shape[0] / BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            sem_ids, multi_sids = self._tokenize_items_from_rqvae(batch)
            B = sem_ids.shape[0]
            D = sem_ids.shape[-1]
            seq_mask = torch.ones_like(sem_ids, dtype=torch.bool)
            sem_ids_fut = None
            token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, 1)
            token_type_ids_fut = None
            item_ids = batch.ids.unsqueeze(-1) if batch.ids.dim() == 1 else batch.ids
            item_ids_fut = batch.ids_fut
            item_seq_mask = batch.seq_mask if batch.seq_mask.dim() > 1 else batch.seq_mask.unsqueeze(-1)
            multi_sids_fut = None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
            token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
            token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
            item_ids = batch.ids
            item_ids_fut = batch.ids_fut
            item_seq_mask = batch.seq_mask
            multi_sids = self._tokenize_multi_sid_batch_from_cached(batch.ids) if self.use_multi_sid else None
            multi_sids_fut = self._tokenize_multi_sid_targets_from_cached(batch.ids_fut) if self.use_multi_sid else None
            if multi_sids is not None:
                multi_sids[~item_seq_mask.unsqueeze(-1).unsqueeze(-1).expand_as(multi_sids)] = -1
        
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            sem_ids_fut=sem_ids_fut,
            token_type_ids_fut=token_type_ids_fut,
            item_ids=item_ids,
            item_ids_fut=item_ids_fut,
            item_seq_mask=item_seq_mask,
            multi_sids=multi_sids,
            multi_sids_fut=multi_sids_fut
        )

if __name__ == "__main__":
    dataset = ItemData("dataset/ml-1m-movie")
    tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
    tokenizer.precompute_corpus_ids(dataset)
    
    seq_data = SeqData("dataset/ml-1m")
    batch = seq_data[:10]
    tokenized = tokenizer(batch)
    import pdb; pdb.set_trace()
