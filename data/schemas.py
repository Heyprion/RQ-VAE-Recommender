from typing import NamedTuple
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor

class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor


class MultiAspectTokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    base_sem_ids: Tensor
    base_sem_ids_fut: Tensor
    seq_mask: Tensor
    history_ids: Tensor
    future_ids: Tensor
    history_x: Tensor
    future_x: Tensor
