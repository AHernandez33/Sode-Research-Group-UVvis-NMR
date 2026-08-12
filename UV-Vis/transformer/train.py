#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch


project_folder = Path(
    __file__
).resolve().parent

src_folder = (
    project_folder
    / "src"
)

if str(src_folder) not in sys.path:
    sys.path.insert(
        0,
        str(src_folder)
    )

from transformer import make_model


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Transformer on UV-Vis spectra containing "
            "181 targets from 220 through 400 nm."
        )
    )

    parser.add_argument(
        "--trainfile",
        type=str,
        default="",
        help="Training CSV containing smiles and wavelength columns 220 through 400."
    )

    parser.add_argument(
        "--testfile",
        type=str,
        default="",
        help="Validation CSV containing smiles and wavelength columns 220 through 400."
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional legacy prefix for <prefix>_train<fold>.csv and <prefix>_test<fold>.csv."
    )

    parser.add_argument(
        "--fold",
        type=str,
        default="",
        help="Optional legacy fold used with --prefix."
    )

    parser.add_argument(
        "--datadir",
        type=str,
        default="models",
        help="Directory used for model checkpoints and training outputs."
    )

    parser.add_argument(
        "--savemodel",
        action="store_true",
        default=False,
        help="Save the final trained model."
    )

    parser.add_argument(
        "--dynamic",
        default=0,
        type=int,
        help="Stop after this many epochs without validation improvement. Use 0 to disable."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs."
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="mse",
        choices=[
            "mse",
            "mae",
            "huber",
            "logcosh"
        ],
        help="Training loss."
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="adam",
        choices=[
            "sgd",
            "adam"
        ],
        help="Optimizer."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay."
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam."
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam."
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for Adam."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout."
    )

    parser.add_argument(
        "--ldist",
        type=float,
        default=0.33,
        help="Distance attention weight."
    )

    parser.add_argument(
        "--lattn",
        type=float,
        default=0.33,
        help="Learned attention weight."
    )

    parser.add_argument(
        "--Ndense",
        type=int,
        default=1,
        help="Number of dense layers."
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads."
    )

    parser.add_argument(
        "--dmodel",
        type=int,
        default=256,
        help="Hidden model dimension."
    )

    parser.add_argument(
        "--nstacklayers",
        type=int,
        default=4,
        help="Number of stacked encoder layers."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size."
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU training."
    )

    parser.add_argument(
        "--twod",
        action="store_true",
        help="Use 2D coordinates."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=420,
        help="Random seed."
    )

    parser.add_argument(
        "--wandb",
        default=None,
        help="Optional Weights & Biases project name."
    )

    return parser


def resolve_data_files(args):
    if args.trainfile:
        if not args.testfile:
            raise ValueError(
                "--testfile is required when --trainfile is used."
            )

        train_file = Path(
            args.trainfile
        ).expanduser().resolve()

        test_file = Path(
            args.testfile
        ).expanduser().resolve()

    elif args.prefix:
        if not args.fold:
            raise ValueError(
                "--fold is required when --prefix is used."
            )

        train_file = Path(
            f"{args.prefix}_train{args.fold}.csv"
        ).expanduser().resolve()

        test_file = Path(
            f"{args.prefix}_test{args.fold}.csv"
        ).expanduser().resolve()

    else:
        raise ValueError(
            "Use either --trainfile and --testfile, or --prefix and --fold."
        )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Validation file not found: {test_file}"
        )

    return train_file, test_file


def build_output_prefix(args):
    prefix = (
        "uvvis_transformer"
        f"_cpu{int(args.cpu)}"
        f"_2d{int(args.twod)}"
        f"_drop{args.dropout}"
        f"_ldist{args.ldist}"
        f"_lattn{args.lattn}"
        f"_Ndense{args.Ndense}"
        f"_heads{args.heads}"
        f"_dmodel{args.dmodel}"
        f"_nsl{args.nstacklayers}"
        f"_epochs{args.epochs}"
        f"_dyn{args.dynamic}"
        f"_seed{args.seed}"
    )

    return prefix


def build_loss(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss(
            reduction="mean"
        )

    if loss_name == "mae":
        return torch.nn.L1Loss(
            reduction="mean"
        )

    if loss_name == "huber":
        return torch.nn.SmoothL1Loss(
            reduction="mean"
        )

    if loss_name == "logcosh":
        def log_cosh_loss(
                prediction,
                target
        ):
            difference = (
                prediction
                - target
            )

            return torch.mean(
                difference
                + torch.nn.functional.softplus(
                    -2.0
                    * difference
                )
                - np.log(
                    2.0
                )
            )

        return log_cosh_loss

    raise ValueError(
        f"Unsupported loss: {loss_name}"
    )


def build_optimizer(
        args,
        model
):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    return torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(
            args.beta1,
            args.beta2
        ),
        eps=args.epsilon,
        weight_decay=args.weight_decay
    )


def move_batch_to_device(
        batch,
        device
):
    adjacency_matrix, node_features, distance_matrix, targets = batch

    return (
        adjacency_matrix.to(
            device
        ),
        node_features.to(
            device
        ),
        distance_matrix.to(
            device
        ),
        targets.to(
            device
        )
    )


def validate_shapes(
        predictions,
        targets
):
    if predictions.ndim != 2:
        raise ValueError(
            "Transformer predictions must be two-dimensional. "
            f"Found shape {tuple(predictions.shape)}."
        )

    if targets.ndim != 2:
        raise ValueError(
            "Transformer targets must be two-dimensional. "
            f"Found shape {tuple(targets.shape)}."
        )

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes do not match. "
            f"Prediction shape: {tuple(predictions.shape)}. "
            f"Target shape: {tuple(targets.shape)}."
        )

    if predictions.shape[1] != 181:
        raise ValueError(
            "Transformer output must contain 181 wavelength values. "
            f"Found {predictions.shape[1]}."
        )


def calculate_metrics(
        predictions,
        targets
):
    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    difference = (
        predictions
        - targets
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                difference
            )
        )
    )

    prediction_flat = predictions.reshape(
        -1
    )

    target_flat = targets.reshape(
        -1
    )

    if (
        np.std(
            prediction_flat
        ) == 0
        or np.std(
            target_flat
        ) == 0
    ):
        r2 = 0.0
    else:
        correlation = np.corrcoef(
            prediction_flat,
            target_flat
        )[
            0,
            1
        ]

        r2 = float(
            correlation ** 2
        )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def evaluate_model(
        model,
        loader,
        device,
        criterion
):
    model.eval()

    predictions = []
    targets = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            (
                adjacency_matrix,
                node_features,
                distance_matrix,
                target
            ) = move_batch_to_device(
                batch=batch,
                device=device
            )

            batch_mask = (
                torch.sum(
                    torch.abs(
                        node_features
                    ),
                    dim=-1
                )
                != 0
            )

            prediction = model(
                node_features,
                batch_mask,
                adjacency_matrix,
                distance_matrix,
                None
            )

            validate_shapes(
                predictions=prediction,
                targets=target
            )

            loss = criterion(
                prediction,
                target
            )

            losses.append(
                float(
                    loss.item()
                )
            )

            predictions.append(
                prediction.detach().cpu().numpy()
            )

            targets.append(
                target.detach().cpu().numpy()
            )

    if not predictions:
        raise ValueError(
            "Evaluation loader produced no batches."
        )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    targets = np.concatenate(
        targets,
        axis=0
    )

    metrics = calculate_metrics(
        predictions=predictions,
        targets=targets
    )

    metrics[
        "loss"
    ] = float(
        np.mean(
            losses
        )
    )

    return predictions, targets, metrics


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.dynamic < 0:
        raise ValueError(
            "--dynamic cannot be negative."
        )

    if (
        args.dynamic > 0
        and args.dynamic >= args.epochs
    ):
        raise ValueError(
            "--dynamic must be smaller than --epochs."
        )

    if args.dmodel % args.heads != 0:
        raise ValueError(
            "--heads must evenly divide --dmodel."
        )

    train_file, validation_file = resolve_data_files(
        args
    )

    output_folder = Path(
        args.datadir
    ).expanduser()

    if not output_folder.is_absolute():
        output_folder = (
            project_folder
            / output_folder
        )

    output_folder = output_folder.resolve()

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_prefix = build_output_prefix(
        args
    )

    if args.cpu or not torch.cuda.is_available():
        device = torch.device(
            "cpu"
        )
    else:
        device = torch.device(
            "cuda"
        )

    if device.type == "cpu":
        from cpu_data_utils import (
            construct_loader,
            load_data_from_df
        )
    else:
        from data_utils import (
            construct_loader,
            load_data_from_df
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    wandb_run = None

    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb,
            name=output_prefix
        )

    print()
    print(
        "UV-Vis Transformer Training"
    )
    print(
        "---------------------------"
    )
    print(
        "Device:",
        device
    )
    print(
        "Training file:",
        train_file
    )
    print(
        "Validation file:",
        validation_file
    )
    print(
        "Output folder:",
        output_folder
    )
    print(
        "Output prefix:",
        output_prefix
    )
    print()

    train_x, train_y = load_data_from_df(
        str(
            train_file
        ),
        one_hot_formal_charge=True,
        two_d_only=args.twod
    )

    validation_x, validation_y = load_data_from_df(
        str(
            validation_file
        ),
        one_hot_formal_charge=True,
        two_d_only=args.twod
    )

    if not train_x:
        raise ValueError(
            "The training dataset contains no valid molecules."
        )

    if not validation_x:
        raise ValueError(
            "The validation dataset contains no valid molecules."
        )

    print(
        "Training molecules:",
        len(
            train_x
        )
    )

    print(
        "Validation molecules:",
        len(
            validation_x
        )
    )

    print(
        "Training target shape:",
        np.asarray(
            train_y
        ).shape
    )

    print(
        "Validation target shape:",
        np.asarray(
            validation_y
        ).shape
    )

    train_loader = construct_loader(
        train_x,
        train_y,
        args.batch_size,
        shuffle=True
    )

    validation_loader = construct_loader(
        validation_x,
        validation_y,
        args.batch_size,
        shuffle=False
    )

    d_atom = train_x[
        0
    ][
        0
    ].shape[
        1
    ]

    model_parameters = {
        "d_atom": d_atom,
        "d_model": args.dmodel,
        "N": args.nstacklayers,
        "h": args.heads,
        "N_dense": args.Ndense,
        "lambda_attention": args.lattn,
        "lambda_distance": args.ldist,
        "leaky_relu_slope": 0.1,
        "dense_output_nonlinearity": "relu",
        "distance_matrix_kernel": "exp",
        "dropout": args.dropout,
        "aggregation_type": "mean",
        "n_output": 181
    }

    model = make_model(
        **model_parameters
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        parameter_count
    )

    criterion = build_loss(
        args.loss
    )

    optimizer = build_optimizer(
        args=args,
        model=model
    )

    if wandb_run is not None:
        wandb.watch(
            model,
            "all"
        )

        wandb.log(
            {
                "Parameters": parameter_count
            },
            step=0
        )

    training_history = []
    best_validation_rmse = float(
        "inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    best_model_file = (
        output_folder
        / f"{output_prefix}_best.model"
    )

    final_model_file = (
        output_folder
        / f"{output_prefix}_trained.model"
    )

    history_file = (
        output_folder
        / f"{output_prefix}_history.json"
    )

    test_results_file = (
        output_folder
        / f"{output_prefix}_validation_results.pkl"
    )

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1
    ):
        model.train()

        epoch_losses = []
        epoch_predictions = []
        epoch_targets = []

        for batch in train_loader:
            (
                adjacency_matrix,
                node_features,
                distance_matrix,
                targets
            ) = move_batch_to_device(
                batch=batch,
                device=device
            )

            optimizer.zero_grad()

            batch_mask = (
                torch.sum(
                    torch.abs(
                        node_features
                    ),
                    dim=-1
                )
                != 0
            )

            predictions = model(
                node_features,
                batch_mask,
                adjacency_matrix,
                distance_matrix,
                None
            )

            validate_shapes(
                predictions=predictions,
                targets=targets
            )

            loss = criterion(
                predictions,
                targets
            )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}: {loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

            epoch_predictions.append(
                predictions.detach().cpu().numpy()
            )

            epoch_targets.append(
                targets.detach().cpu().numpy()
            )

        epoch_predictions = np.concatenate(
            epoch_predictions,
            axis=0
        )

        epoch_targets = np.concatenate(
            epoch_targets,
            axis=0
        )

        training_metrics = calculate_metrics(
            predictions=epoch_predictions,
            targets=epoch_targets
        )

        training_metrics[
            "loss"
        ] = float(
            np.mean(
                epoch_losses
            )
        )

        (
            validation_predictions,
            validation_targets,
            validation_metrics
        ) = evaluate_model(
            model=model,
            loader=validation_loader,
            device=device,
            criterion=criterion
        )

        epoch_record = {
            "epoch": epoch,
            "training_loss": training_metrics[
                "loss"
            ],
            "training_rmse": training_metrics[
                "rmse"
            ],
            "training_mae": training_metrics[
                "mae"
            ],
            "training_r2": training_metrics[
                "r2"
            ],
            "validation_loss": validation_metrics[
                "loss"
            ],
            "validation_rmse": validation_metrics[
                "rmse"
            ],
            "validation_mae": validation_metrics[
                "mae"
            ],
            "validation_r2": validation_metrics[
                "r2"
            ]
        }

        training_history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train loss {training_metrics['loss']:.6f} | "
            f"train RMSE {training_metrics['rmse']:.6f} | "
            f"val loss {validation_metrics['loss']:.6f} | "
            f"val RMSE {validation_metrics['rmse']:.6f}"
        )

        if wandb_run is not None:
            wandb.log(
                epoch_record,
                step=epoch
            )

        if validation_metrics[
            "rmse"
        ] < best_validation_rmse:
            best_validation_rmse = validation_metrics[
                "rmse"
            ]

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_parameters": model_parameters,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "wavelength_min": 220,
                    "wavelength_max": 400,
                    "wavelength_count": 181
                },
                best_model_file
            )

        else:
            epochs_without_improvement += 1

        with open(
            history_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                training_history,
                file,
                indent=4
            )

        if (
            args.dynamic > 0
            and epochs_without_improvement >= args.dynamic
        ):
            print(
                "Early stopping at epoch:",
                epoch
            )
            break

    if args.savemodel:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_parameters": model_parameters,
                "epoch": training_history[
                    -1
                ][
                    "epoch"
                ],
                "validation_metrics": training_history[
                    -1
                ],
                "wavelength_min": 220,
                "wavelength_max": 400,
                "wavelength_count": 181
            },
            final_model_file
        )

    if best_model_file.exists():
        checkpoint = torch.load(
            best_model_file,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    (
        final_validation_predictions,
        final_validation_targets,
        final_validation_metrics
    ) = evaluate_model(
        model=model,
        loader=validation_loader,
        device=device,
        criterion=criterion
    )

    with open(
        test_results_file,
        "wb"
    ) as file:
        pickle.dump(
            {
                "predicted": final_validation_predictions,
                "target": final_validation_targets,
                "metrics": final_validation_metrics,
                "best_epoch": best_epoch,
                "best_model_file": str(
                    best_model_file
                )
            },
            file
        )

    total_minutes = (
        time.time()
        - start_time
    ) / 60.0

    print()
    print(
        "Training complete"
    )
    print(
        "Best epoch:",
        best_epoch
    )
    print(
        "Best validation RMSE:",
        best_validation_rmse
    )
    print(
        "Final validation MSE:",
        final_validation_metrics[
            "mse"
        ]
    )
    print(
        "Final validation RMSE:",
        final_validation_metrics[
            "rmse"
        ]
    )
    print(
        "Final validation MAE:",
        final_validation_metrics[
            "mae"
        ]
    )
    print(
        "Final validation R2:",
        final_validation_metrics[
            "r2"
        ]
    )
    print(
        "Total training minutes:",
        round(
            total_minutes,
            2
        )
    )
    print(
        "Best model:",
        best_model_file
    )

    if args.savemodel:
        print(
            "Final model:",
            final_model_file
        )

    print(
        "Training history:",
        history_file
    )
    print(
        "Validation results:",
        test_results_file
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
