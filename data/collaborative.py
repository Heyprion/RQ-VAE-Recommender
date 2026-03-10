import os
import random
import torch

from data.processed import DATASET_NAME_TO_MAX_SEQ_LEN
from data.processed import DATASET_NAME_TO_RAW_DATASET
from data.processed import RecDataset
from typing import NamedTuple


class CollaborativeSplit(NamedTuple):
    num_users: int
    num_items: int
    train_edge_index: torch.Tensor
    train_user_items: list[set[int]]
    train_item_mask: torch.Tensor
    eval_targets: torch.Tensor
    test_targets: torch.Tensor
    user_id_map: dict[int, int]


def _flatten_unique_edges(user_items: list[list[int]]) -> torch.Tensor:
    users, items = [], []
    for user_idx, item_list in enumerate(user_items):
        for item_id in sorted(set(item_list)):
            users.append(user_idx)
            items.append(item_id)

    if len(users) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([users, items], dtype=torch.long)


def load_collaborative_split(
    root: str,
    dataset: RecDataset = RecDataset.AMAZON,
    force_process: bool = False,
    split: str = "beauty",
) -> CollaborativeSplit:
    raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
    max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

    raw_data = raw_dataset_class(root=root, split=split)
    processed_data_path = raw_data.processed_paths[0]
    if not os.path.exists(processed_data_path) or force_process:
        raw_data.process(max_seq_len=max_seq_len)

    history = raw_data.data[("user", "rated", "item")]["history"]
    train_sequences = history["train"]["itemId"]
    train_users = history["train"]["userId"].squeeze(-1).tolist()
    eval_targets = history["eval"]["itemId_fut"].squeeze(-1).to(torch.long)
    test_targets = history["test"]["itemId_fut"].squeeze(-1).to(torch.long)

    user_id_map = {int(user_id): idx for idx, user_id in enumerate(train_users)}
    user_items = [list(map(int, items)) for items in train_sequences]
    train_edge_index = _flatten_unique_edges(user_items)

    num_items = raw_data.data["item"]["x"].shape[0]
    train_item_mask = torch.zeros(num_items, dtype=torch.bool)
    if train_edge_index.numel() > 0:
        train_item_mask[train_edge_index[1].unique()] = True

    return CollaborativeSplit(
        num_users=len(train_users),
        num_items=num_items,
        train_edge_index=train_edge_index,
        train_user_items=[set(items) for items in user_items],
        train_item_mask=train_item_mask,
        eval_targets=eval_targets,
        test_targets=test_targets,
        user_id_map=user_id_map,
    )


def sample_bpr_batch(
    train_user_items: list[set[int]],
    num_items: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users, pos_items, neg_items = [], [], []
    valid_users = [idx for idx, items in enumerate(train_user_items) if len(items) > 0 and len(items) < num_items]
    if len(valid_users) == 0:
        raise ValueError("No valid users available for BPR sampling.")

    for _ in range(batch_size):
        user_idx = random.choice(valid_users)
        pos_item = random.choice(tuple(train_user_items[user_idx]))
        neg_item = random.randrange(num_items)
        while neg_item in train_user_items[user_idx]:
            neg_item = random.randrange(num_items)

        users.append(user_idx)
        pos_items.append(pos_item)
        neg_items.append(neg_item)

    return (
        torch.tensor(users, device=device, dtype=torch.long),
        torch.tensor(pos_items, device=device, dtype=torch.long),
        torch.tensor(neg_items, device=device, dtype=torch.long),
    )
