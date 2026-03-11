import argparse
import os
import torch

from data.collaborative import load_collaborative_split
from data.processed import RecDataset
from modules.lightgcn import LightGCN
from modules.lightgcn import build_normalized_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--dataset-folder", default="dataset/amazon", type=str)
    parser.add_argument("--dataset-split", default="beauty", type=str)
    parser.add_argument("--dataset", default="AMAZON", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--force-dataset-process", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RecDataset[args.dataset]
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["model_config"]

    collab = load_collaborative_split(
        root=args.dataset_folder,
        dataset=dataset,
        force_process=args.force_dataset_process,
        split=args.dataset_split,
    )

    model = LightGCN(
        num_users=config["num_users"],
        num_items=config["num_items"],
        embedding_dim=config["embedding_dim"],
        n_layers=config["n_layers"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    adjacency = build_normalized_adjacency(
        num_users=collab.num_users,
        num_items=collab.num_items,
        edge_index=collab.train_edge_index.to(args.device),
        device=torch.device(args.device),
    )

    with torch.no_grad():
        item_embeddings = model.compute_embeddings(adjacency).item_embeddings.detach().cpu().clone()

    if item_embeddings.shape[0] != collab.num_items:
        raise ValueError(
            f"Exported {item_embeddings.shape[0]} item embeddings, expected {collab.num_items}."
        )

    with torch.no_grad():
        item_embeddings[~collab.train_item_mask] = 0.0

    payload = {
        "item_embeddings": item_embeddings,
        "item_mask": collab.train_item_mask,
        "num_items": collab.num_items,
        "embedding_dim": item_embeddings.shape[1],
        "dataset_meta": {
            "dataset_folder": args.dataset_folder,
            "dataset": dataset.name,
            "dataset_split": args.dataset_split,
            "checkpoint": args.checkpoint,
        },
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved item embeddings to {args.output}")


if __name__ == "__main__":
    main()
