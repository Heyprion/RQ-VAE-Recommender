from typing import NamedTuple
from typing import Optional
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
    seq_mask: Tensor
    token_type_ids: Tensor
    sem_ids_fut: Optional[Tensor] = None
    token_type_ids_fut: Optional[Tensor] = None
    item_ids: Optional[Tensor] = None
    item_ids_fut: Optional[Tensor] = None
    item_seq_mask: Optional[Tensor] = None
    multi_sids: Optional[Tensor] = None
    multi_sids_fut: Optional[Tensor] = None
