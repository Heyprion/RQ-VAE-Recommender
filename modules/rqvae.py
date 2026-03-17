import torch

from data.schemas import SeqBatch
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor

torch.set_float32_matmul_precision('high')


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor
    item_repr: Tensor
    head_repr: Tensor
    multi_sids: Tensor
    orth_loss: Tensor
    balance_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    orth_loss: Tensor
    balance_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    item_repr: Tensor
    head_repr: Tensor
    multi_sids: Tensor


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        use_multi_sid: bool = False,
        num_sid_heads: int = 1,
        lambda_orth: float = 0.0,
        lambda_bal: float = 0.0,
    ) -> None:
        self._config = locals()
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features
        self.use_multi_sid = use_multi_sid and max(1, num_sid_heads) > 1
        self.num_sid_heads = max(1, num_sid_heads) if self.use_multi_sid else 1
        self.lambda_orth = lambda_orth
        self.lambda_bal = lambda_bal

        if self.use_multi_sid:
            self.projection_heads = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False)
                for _ in range(self.num_sid_heads)
            ])
            self.head_layers = nn.ModuleList([
                nn.ModuleList([
                    Quantize(
                        embed_dim=embed_dim,
                        n_embed=codebook_size,
                        forward_mode=codebook_mode,
                        do_kmeans_init=codebook_kmeans_init,
                        codebook_normalize=layer_idx == 0 and codebook_normalize,
                        sim_vq=codebook_sim_vq,
                        commitment_weight=commitment_weight
                    ) for layer_idx in range(n_layers)
                ]) for _ in range(self.num_sid_heads)
            ])
        else:
            self.layers = nn.ModuleList(modules=[
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=codebook_mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=i == 0 and codebook_normalize,
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight
                ) for i in range(n_layers)
            ])

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
    
    @cached_property
    def config(self) -> dict:
        return self._config
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def _quantize_head(
        self,
        head_repr: Tensor,
        layers: nn.ModuleList,
        gumbel_t: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual = head_repr
        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in layers:
            residuals.append(residual)
            quantized = layer(residual, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb, sem_id = quantized.embeddings, quantized.ids
            residual = residual - emb
            sem_ids.append(sem_id)
            embs.append(emb)

        return (
            torch.stack(embs, dim=-1),
            torch.stack(residuals, dim=-1),
            torch.stack(sem_ids, dim=-1),
            quantize_loss
        )

    def _compute_aux_losses(self, head_repr: Tensor) -> tuple[Tensor, Tensor]:
        if not self.use_multi_sid:
            zero = torch.zeros((), device=head_repr.device, dtype=head_repr.dtype)
            return zero, zero

        cross_head_gram = torch.einsum("bmd,bne->mnde", head_repr, head_repr)
        head_eye = torch.eye(
            self.num_sid_heads,
            device=head_repr.device,
            dtype=torch.bool
        ).view(self.num_sid_heads, self.num_sid_heads, 1, 1)
        orth_loss = cross_head_gram.masked_fill(head_eye, 0).pow(2).mean()

        head_norms = head_repr.norm(dim=-1)
        mean_head_norm = head_norms.mean(dim=1, keepdim=True)
        balance_loss = (head_norms - mean_head_norm).pow(2).mean()
        return orth_loss, balance_loss

    def get_semantic_ids(
        self,
        x: Tensor,
        gumbel_t: float = 0.001
    ) -> RqVaeOutput:
        item_repr = self.encode(x)

        if self.use_multi_sid:
            head_repr = torch.stack(
                [projection_head(item_repr) for projection_head in self.projection_heads],
                dim=1
            )

            multi_embs, multi_residuals, multi_sids = [], [], []
            quantize_loss = 0
            for head_idx, head_layers in enumerate(self.head_layers):
                head_embs, head_residuals, head_sids, head_quantize_loss = self._quantize_head(
                    head_repr=head_repr[:, head_idx, :],
                    layers=head_layers,
                    gumbel_t=gumbel_t
                )
                multi_embs.append(head_embs)
                multi_residuals.append(head_residuals)
                multi_sids.append(head_sids)
                quantize_loss += head_quantize_loss

            multi_embs = torch.stack(multi_embs, dim=1)
            multi_residuals = torch.stack(multi_residuals, dim=1)
            multi_sids = torch.stack(multi_sids, dim=1)
            embs = multi_embs.mean(dim=1)
            residuals = multi_residuals.mean(dim=1)
            sem_ids = multi_sids[:, 0, :]
            orth_loss, balance_loss = self._compute_aux_losses(head_repr)
        else:
            head_repr = item_repr.unsqueeze(1)
            embs, residuals, sem_ids, quantize_loss = self._quantize_head(
                head_repr=item_repr,
                layers=self.layers,
                gumbel_t=gumbel_t
            )
            multi_sids = sem_ids.unsqueeze(1)
            orth_loss = torch.zeros((), device=item_repr.device, dtype=item_repr.dtype)
            balance_loss = torch.zeros((), device=item_repr.device, dtype=item_repr.dtype)

        return RqVaeOutput(
            embeddings=embs,
            residuals=residuals,
            sem_ids=sem_ids,
            quantize_loss=quantize_loss,
            item_repr=item_repr,
            head_repr=head_repr,
            multi_sids=multi_sids,
            orth_loss=orth_loss,
            balance_loss=balance_loss,
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=-1))
        x_hat = torch.cat([l2norm(x_hat[...,:-self.n_cat_feats]), x_hat[...,-self.n_cat_feats:]], axis=-1)

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        original_loss = (reconstuction_loss + rqvae_loss).mean()
        loss = (
            original_loss +
            self.lambda_orth * quantized.orth_loss +
            self.lambda_bal * quantized.balance_loss
        )

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                ((quantized.sem_ids.unsqueeze(1) == quantized.sem_ids.unsqueeze(0)).all(axis=-1)), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            orth_loss=quantized.orth_loss,
            balance_loss=quantized.balance_loss,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            item_repr=quantized.item_repr,
            head_repr=quantized.head_repr,
            multi_sids=quantized.multi_sids
        )
