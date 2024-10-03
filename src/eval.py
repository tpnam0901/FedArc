import random
import numpy as np
import torch
import torch.nn as nn

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import os
import argparse
from tqdm.auto import tqdm
from data.cifar10 import Cifar10Dataset
from data.dataloader import build_dataloader
from configs.base import Config
from models import networks, metrics


def main(cfg: Config, best_ckpt: bool = True):

    # Preparing checkpoint output
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")

    # Build dataset
    print("Building dataset...")
    dataset = Cifar10Dataset(cfg)
    dataloader = build_dataloader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=False,
        pin_memory=cfg.pin_memory,
    )

    cfg.num_classes = len(dataset.classes)

    print("Building model, loss and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(networks, cfg.model_type)(cfg)
    model.to(device)

    acc_metric = metrics.Accuracy(cfg)
    roc_auc_metric = metrics.ROC_AUC(cfg)

    if best_ckpt:
        checkpoint = torch.load(weight_best_path)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(weight_last_path)
        model.load_state_dict(checkpoint["state_dict_model"])

    total_acc_val = []
    total_roc_auc_val = []
    dataset.test()
    model.eval()
    for inputs, labels in tqdm(iter(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            acc = acc_metric(outputs, labels)
            roc_auc = roc_auc_metric(outputs, labels)

            acc = acc.detach().cpu().numpy()
            roc_auc = roc_auc.detach().cpu().numpy()

        total_acc_val.append(acc.item())
        total_roc_auc_val.append(roc_auc.item())

    total_acc = np.mean(total_acc_val).item()
    total_roc_auc = np.mean(total_roc_auc_val).item()

    print("Test_acc: {:.4f} - Test_roc_auc: {:.4f}".format(total_acc, total_roc_auc))


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
