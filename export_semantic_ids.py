import argparse
import os
import torch

from data.processed import ItemData
from data.processed import RecDataset
from modules.tokenizer.semids import SemanticIdTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae-checkpoint", required=True, type=str)
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
    checkpoint = torch.load(args.rqvae_checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["model_config"]

    item_dataset = ItemData(
        root=args.dataset_folder,
        dataset=dataset,
        force_process=args.force_dataset_process,
        train_test_split="all",
        split=args.dataset_split,
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=config["input_dim"],
        output_dim=config["embed_dim"],
        hidden_dims=config["hidden_dims"],
        codebook_size=config["codebook_size"],
        n_layers=config["n_layers"],
        n_cat_feats=config["n_cat_features"],
        commitment_weight=config["commitment_weight"],
        rqvae_weights_path=args.rqvae_checkpoint,
        rqvae_codebook_normalize=config["codebook_normalize"],
        rqvae_sim_vq=config["codebook_sim_vq"],
    ).to(args.device)

    semantic_ids = tokenizer.precompute_corpus_ids(item_dataset).cpu()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "semantic_ids": semantic_ids,
            "dataset_meta": {
                "dataset_folder": args.dataset_folder,
                "dataset": dataset.name,
                "dataset_split": args.dataset_split,
                "rqvae_checkpoint": args.rqvae_checkpoint,
            },
        },
        args.output,
    )
    print(f"Saved semantic ids to {args.output}")


if __name__ == "__main__":
    main()
