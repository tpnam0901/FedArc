import random
import numpy as np
import torch

SEED = 1996
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import os
import math
import data
import time
import argparse
import logging
import mlflow
import datetime
from torch import optim
from tqdm.auto import tqdm
from scipy.special import softmax
from configs.base import Config, import_config
from models import networks, metrics, losses, aggregation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def train(
    cfg,
    logger,
    weight_best_path,
    weight_last_path,
    dataset,
    dataloader,
    dataloader_shuffle,
    device,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    acc_metric,
    best_acc,
    global_train_epoch,
    global_val_epoch,
    mlflow_prefix,
    current_round,
    feature_anchors,
):
    if mlflow_prefix:
        weight_best_path = os.path.join(
            cfg.checkpoint_dir, f"{mlflow_prefix}_weight_best.pth"
        )
    for epoch in range(cfg.num_epochs):
        start = time.process_time()
        total_loss_train = []
        total_acc_train = []
        local_train_step = 0
        dataset.train()
        model.train()
        model.to(device)
        with tqdm(total=len(dataloader_shuffle), ascii=True) as pbar:
            for inputs, labels in iter(dataloader_shuffle):
                local_train_step += 1

                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                if "CosineSimilarityLoss" in cfg.loss_type:
                    anchors = torch.stack([feature_anchors] * labels.size(0), dim=0)
                    anchors = anchors[
                        range(labels.size(0)), list(labels.detach().cpu().numpy()), :
                    ]
                    outputs = list(outputs)
                    outputs.append(anchors)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                acc = acc_metric(outputs, labels)

                loss = loss.detach().cpu().numpy()
                acc = acc.detach().cpu().numpy()

                total_acc_train.append(acc.item())
                total_loss_train.append(loss.item())

                postfix = (
                    "Round {}/{}, Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(
                        current_round + 1,
                        cfg.num_rounds,
                        epoch + 1,
                        cfg.num_epochs,
                        loss.item(),
                        acc.item(),
                    )
                )
                pbar.set_description(postfix)
                pbar.update(1)

                if local_train_step % cfg.ckpt_save_fred == 0:
                    checkpoint = {
                        "global_train_epoch": global_train_epoch,
                        "local_train_step": local_train_step,
                        "global_val_epoch": global_val_epoch,
                        "best_acc": best_acc,
                        "state_dict_model": model.state_dict(),
                        "state_dict_optim_model": optimizer.state_dict(),
                        "state_dict_scheduler_model": lr_scheduler.state_dict(),
                    }
                    torch.save(checkpoint, weight_last_path)
        end = time.process_time()
        mlflow.log_metric(
            f"{mlflow_prefix}_epoch_time", end - start, step=global_train_epoch
        )

        checkpoint = {
            "global_train_epoch": global_train_epoch,
            "local_train_step": local_train_step,
            "global_val_epoch": global_val_epoch,
            "best_acc": best_acc,
            "state_dict_model": model.state_dict(),
            "state_dict_optim_model": optimizer.state_dict(),
            "state_dict_scheduler_model": lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, weight_last_path)
        logger.info(
            "Round {}/{}, Epoch {}/{} - epoch_loss: {:.4f} - epoch_acc: {:.4f}".format(
                current_round + 1,
                cfg.num_rounds,
                epoch + 1,
                cfg.num_epochs,
                np.mean(total_loss_train).item(),
                np.mean(total_acc_train).item(),
            )
        )

        mlflow.log_metric(
            f"{mlflow_prefix}_epoch_loss",
            np.mean(total_loss_train).item(),
            step=global_train_epoch,
        )
        mlflow.log_metric(
            f"{mlflow_prefix}_epoch_acc",
            np.mean(total_acc_train).item(),
            step=global_train_epoch,
        )

        if cfg.client_do_evaluation:
            global_val_epoch, best_acc = eval(
                model,
                dataset,
                dataloader,
                device,
                criterion,
                acc_metric,
                global_val_epoch,
                logger,
                mlflow_prefix,
                current_round,
                epoch,
                weight_best_path,
                best_acc,
            )
        # lr_scheduler.step()

    # -------------------------------- Feature anchor ------------------------------------- #
    feature_anchors = torch.zeros(cfg.num_classes, cfg.embedding_linear_out)
    if "CosineSimilarityLoss" in cfg.loss_type:
        dataset.train()
        features = [
            torch.zeros(0, cfg.embedding_linear_out) for _ in range(cfg.num_classes)
        ]
        with torch.no_grad():
            for inputs, labels in iter(dataloader_shuffle):
                inputs = inputs.to(device)
                outputs = model(inputs)
                for cls_index, feat in zip(labels, outputs[1].detach().cpu()):
                    cls_index = int(cls_index.item())
                    features[cls_index] = torch.concat(
                        [features[cls_index], feat.unsqueeze(0)], dim=0
                    )
            feature_anchors = torch.stack([feat.mean(0) for feat in features], dim=0)

    model.to("cpu")
    if cfg.client_best_weight and cfg.client_do_evaluation:
        model.load_state_dict(torch.load(weight_best_path, map_location="cpu"))

    return (
        global_train_epoch,
        global_val_epoch,
        best_acc,
        model.state_dict(),
        feature_anchors,
    )


def eval(
    model,
    dataset,
    dataloader,
    device,
    criterion,
    acc_metric,
    global_val_epoch,
    logger,
    mlflow_prefix,
    current_round,
    epoch,
    weight_best_path,
    best_acc,
):
    total_loss_val = []
    total_acc_val = []
    model.eval()
    dataset.val()
    global_val_epoch += 1
    for inputs, labels in tqdm(iter(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            acc = acc_metric(outputs, labels)

            loss = loss.detach().cpu().numpy()
            acc = acc.detach().cpu().numpy()

        total_acc_val.append(acc.item())
        total_loss_val.append(loss.item())

    total_loss = np.mean(total_loss_val).item()
    total_acc = np.mean(total_acc_val).item()

    mlflow.log_metric(
        f"{mlflow_prefix}_val_loss", float(total_loss), step=global_val_epoch
    )
    mlflow.log_metric(
        f"{mlflow_prefix}_val_acc", float(total_acc), step=global_val_epoch
    )
    logger.info(
        "Round {}/{}, Epoch {}/{} - val_loss: {:.4f} - val_acc: {:.4f}".format(
            current_round + 1,
            cfg.num_rounds,
            epoch + 1,
            cfg.num_epochs,
            total_loss,
            total_acc,
        )
    )
    if total_acc > best_acc and cfg.client_best_weight:
        best_acc = total_acc
        torch.save(model.state_dict(), weight_best_path)
    return global_val_epoch, best_acc


def main(cfg: Config, resume: bool):
    current_time = (
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not cfg.datetime
        else cfg.datetime
    )
    cfg.datetime = current_time
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.name, current_time)

    # Log, weight, mlflow folder
    log_dir = os.path.join(cfg.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ## Add logger to log folder
    logging.getLogger().setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(cfg.name)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    ## Add mlflow to log folder
    mlflow.set_tracking_uri(
        uri=f'file://{os.path.abspath(os.path.join(log_dir, "mlruns"))}'
    )

    # Preparing checkpoint output
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")

    # Build dataset
    logger.info("Building dataset...")
    dataset = getattr(data, cfg.data_type)(cfg)
    if cfg.test_as_val:
        dataset.val_X = dataset.test_X
        dataset.val_y = dataset.test_y
    if cfg.val_from_train:
        dataset.train_val_split(val_size=cfg.val_size, seed=1996)
    dataset.train()
    dataloader = data.build_dataloader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=False,
        pin_memory=cfg.pin_memory,
    )
    dataloader_shuffle = data.build_dataloader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
    )

    cfg.num_classes = len(dataset.classes)

    # Save configs
    logger.info("Saving config to {}".format(cfg.checkpoint_dir))
    cfg.save(cfg.checkpoint_dir)
    cfg.show()

    logger.info("Building model, loss and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(networks, cfg.model_type)(cfg)
    model.to(device)

    criterion = getattr(losses, cfg.loss_type)(cfg)
    criterion.to(device)

    acc_metric = metrics.Accuracy(cfg)
    acc_metric.to(device)

    # -------------------------- Federated training -------------------------- #
    model.to("cpu")
    global_state_dict = model.state_dict()
    global_feature_anchors = torch.zeros(cfg.num_classes, 0, cfg.embedding_linear_out)
    global_best_acc = -1.0
    global_val_step = 0

    global_train_epochs = [0] * cfg.num_clients
    global_val_epochs = [0] * cfg.num_clients
    local_best_acc = [-1.0] * cfg.num_clients

    client_ids = list(range(cfg.num_clients))
    if cfg.num_clients > 1:
        dataset.fl_split()

    with mlflow.start_run():
        for current_round in range(cfg.num_rounds):
            local_state_dicts = [{}] * cfg.num_clients
            if current_round == 0:
                criterion.set_lambda(1.0)
            else:
                criterion.set_lambda(cfg.loss_lambda)

            # Get random client
            random.shuffle(client_ids)
            client_active_ids = client_ids[
                : min(
                    int(math.ceil(cfg.num_client_prob * cfg.num_clients)),
                    cfg.num_clients,
                )
            ]
            # There is at least one client in the pool
            client_active_ids = (
                [0] if len(client_active_ids) == 0 else client_active_ids
            )

            for client_id in client_active_ids:
                logger.info("Training client: {}".format(client_id))
                dataset.set_dataset(client_id)
                model.to("cpu")
                model.load_state_dict(global_state_dict)
                model.to(device)
                optimizer = optim.SGD(
                    params=model.parameters(),
                    momentum=cfg.momentum,
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )

                lr_scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=cfg.lr_step_size,
                    gamma=cfg.gamma,
                )

                if resume:
                    raise NotImplementedError

                train_step, val_step, best_acc, state_dict, local_feature_anchor = (
                    train(
                        cfg=cfg,
                        logger=logger,
                        weight_best_path=weight_best_path,
                        weight_last_path=weight_last_path,
                        dataset=dataset,
                        dataloader=dataloader,
                        dataloader_shuffle=dataloader_shuffle,
                        device=device,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        acc_metric=acc_metric,
                        best_acc=local_best_acc[client_id],
                        global_train_epoch=global_train_epochs[client_id],
                        global_val_epoch=global_val_epochs[client_id],
                        mlflow_prefix=client_id,
                        current_round=current_round,
                        feature_anchors=global_feature_anchors,
                    )
                )
                local_state_dicts[client_id] = state_dict
                global_train_epochs[client_id] += train_step
                global_val_epochs[client_id] += val_step
                local_best_acc[client_id] = best_acc
                global_feature_anchors = torch.concat(
                    [
                        global_feature_anchors,
                        local_feature_anchor.unsqueeze(1),
                    ],
                    dim=1,
                )

            client_weights = [1] * cfg.num_clients
            new_client_weights = []
            for client_id in client_active_ids:
                new_client_weights.append(
                    len(dataset.get_dataset(client_id)["train_X"])
                )
            new_client_weights = softmax(new_client_weights)
            for client_id, new_weight in zip(client_active_ids, new_client_weights):
                client_weights[client_id] = new_weight

            global_state_dict = getattr(aggregation, cfg.aggregate_type)(
                global_state_dict, local_state_dicts, client_weights
            )
            global_feature_anchors = global_feature_anchors.mean(1).unsqueeze(1)
            model.to("cpu")
            model.load_state_dict(global_state_dict)
            model.to(device)
            logger.info(
                "Evaluating global model at round {}/{}".format(
                    current_round, cfg.num_rounds
                )
            )
            global_val_step, best_acc = eval(
                model=model,
                dataset=dataset,
                dataloader=dataloader,
                device=device,
                criterion=criterion,
                acc_metric=acc_metric,
                global_val_epoch=global_val_step,
                logger=logger,
                mlflow_prefix="global",
                current_round=current_round,
                epoch=cfg.num_epochs - 1,
                weight_best_path=weight_best_path,
                best_acc=global_best_acc,
            )

            if global_best_acc < best_acc:
                global_best_acc = best_acc
                torch.save(model.state_dict(), weight_best_path)

    checkpoint = {
        "state_dict_model": model.state_dict(),
    }
    torch.save(checkpoint, weight_last_path)
    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Training finished at {}".format(end_time))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="configs/base.py",
        help="Path to config.py file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to resume cfg.log file if want to resume training",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = import_config(args.config)
    if args.resume:
        cfg.load(args.resume)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(cfg, args.resume)
