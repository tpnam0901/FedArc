import random
import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import os
import argparse
from tqdm.auto import tqdm

import data

from configs.base import Config
from models import networks
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def main(cfg: Config, best_ckpt: bool = True):

    # Preparing checkpoint output
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")
    # Build dataset
    print("Building dataset...")
    dataset = getattr(data, cfg.data_type)(cfg)
    dataset.train()
    dataloader = data.build_dataloader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=False,
        pin_memory=cfg.pin_memory,
    )

    print("Building model, loss and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(networks, cfg.model_type)(cfg)
    model.to(device)

    if best_ckpt:
        checkpoint = torch.load(weight_best_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(weight_last_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict_model"])

    dataset.train()
    model.eval()
    feats = np.zeros((0, cfg.embedding_linear_out))
    targets = np.zeros((0))
    for inputs, labels in tqdm(iter(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        targets = np.concatenate((targets, labels.cpu().numpy()), axis=0)

        with torch.no_grad():
            _, temp_feats = model(inputs)
            feats = np.concatenate((feats, temp_feats.detach().cpu().numpy()), axis=0)
    # TSNE
    X_transformed = TSNE(n_components=2).fit_transform(feats)
    fig = plt.figure(figsize=(5, 5))
    # 2D plot
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=targets)
    # remove ticks
    ax.xaxis.set(ticks=())
    ax.yaxis.set(ticks=())

    # add legend
    if False:
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="")
        legend1.get_texts()[0].set_text("Angry")
        legend1.get_texts()[1].set_text("Happy")
        legend1.get_texts()[2].set_text("Sad")
        legend1.get_texts()[3].set_text("Neutral")
        ax.add_artist(legend1)

    # remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"TSNE_{cfg.name}.png", dpi=300, bbox_inches="tight")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to resume cfg.log file if want to resume training",
    )
    parser.add_argument("--best_ckpt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg = Config()
    if args.resume:
        cfg.load(args.resume)
    main(cfg, args.best_ckpt)
