from pathlib import Path
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm


training_folder = Path(
    __file__
).resolve().parent

input_file = (
    training_folder.parent
    / "generated"
    / "base_predictions_mpnn.csv"
)

models_folder = (
    training_folder
    / "models"
)

models_folder.mkdir(
    parents=True,
    exist_ok=True
)

best_model_file = (
    models_folder
    / "counter_prop_best.pth"
)

final_model_file = (
    models_folder
    / "counter_prop.pth"
)

input_scaler_file = (
    models_folder
    / "counter_prop_input_scaler.json"
)

target_scaler_file = (
    models_folder
    / "counter_prop_target_scaler.json"
)

training_history_file = (
    training_folder
    / "counter_prop_training_history.csv"
)

training_plot_file = (
    training_folder
    / "counter_prop_training.png"
)

mse_plot_file = (
    training_folder
    / "counter_prop_training_mse.png"
)

combined_metrics_plot_file = (
    training_folder
    / "counter_prop_training_metrics.png"
)

predicted_spectrum_plot_file = (
    training_folder
    / "counter_prop_predicted_spectrum.png"
)


wavelength_min = 220
wavelength_max = 400

wavelengths = list(
    range(
        wavelength_min,
        wavelength_max + 1
    )
)

input_columns = [
    f"MPNN_{wavelength}"
    for wavelength in wavelengths
]

target_columns = [
    f"experimental_{wavelength}"
    for wavelength in wavelengths
]


random_seed = 42

validation_size = 0.10
test_size = 0.10

batch_size = 8
number_epochs = 300
learning_rate = 0.0005
weight_decay = 0.00001
early_stopping_patience = 30

hidden_size_1 = 256
hidden_size_2 = 128
hidden_size_3 = 128

dropout_probability = 0.0


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def set_random_seed(
        seed
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


class CounterPropagationNetwork(
    nn.Module
):
    def __init__(
        self,
        input_size,
        output_size
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size_1
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_size_1
            ),
            nn.Dropout(
                dropout_probability
            ),

            nn.Linear(
                hidden_size_1,
                hidden_size_2
            ),
            nn.ReLU(),
            nn.LayerNorm(
                hidden_size_2
            ),
            nn.Dropout(
                dropout_probability
            ),

            nn.Linear(
                hidden_size_2,
                hidden_size_3
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_size_3,
                output_size
            )
        )

    def forward(
        self,
        inputs
    ):
        return self.network(
            inputs
        )


def validate_columns(
        dataframe
):
    missing_input_columns = [
        column
        for column in input_columns
        if column not in dataframe.columns
    ]

    missing_target_columns = [
        column
        for column in target_columns
        if column not in dataframe.columns
    ]

    if missing_input_columns:
        raise ValueError(
            "Missing MPNN prediction columns: "
            f"{missing_input_columns[:10]}"
        )

    if missing_target_columns:
        raise ValueError(
            "Missing experimental spectrum columns: "
            f"{missing_target_columns[:10]}"
        )


def load_training_data():
    if not input_file.exists():
        raise FileNotFoundError(
            f"Training file was not found: {input_file}"
        )

    dataframe = pd.read_csv(
        input_file
    )

    validate_columns(
        dataframe
    )

    if "MPNN_status" in dataframe.columns:
        dataframe = dataframe[
            dataframe[
                "MPNN_status"
            ].astype(
                str
            ).str.lower().eq(
                "success"
            )
        ].copy()

    dataframe[
        input_columns
    ] = dataframe[
        input_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    dataframe[
        target_columns
    ] = dataframe[
        target_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    dataframe.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )

    dataframe.dropna(
        subset=(
            input_columns
            + target_columns
        ),
        inplace=True
    )

    dataframe.reset_index(
        drop=True,
        inplace=True
    )

    if len(
        dataframe
    ) < 20:
        raise ValueError(
            "Not enough valid rows are available to train the model."
        )

    input_values = dataframe[
        input_columns
    ].to_numpy(
        dtype=np.float32
    )

    experimental_values = dataframe[
        target_columns
    ].to_numpy(
        dtype=np.float32
    )

    # Residual learning:
    # The CPNN learns the correction required to transform the
    # MPNN spectrum into the experimental spectrum.
    target_values = (
        experimental_values
        - input_values
    ).astype(
        np.float32
    )

    return (
        dataframe,
        input_values,
        target_values
    )


def split_training_data(
        input_values,
        target_values
):
    combined_test_size = (
        validation_size
        + test_size
    )

    (
        input_train,
        input_remaining,
        target_train,
        target_remaining
    ) = train_test_split(
        input_values,
        target_values,
        test_size=combined_test_size,
        random_state=random_seed
    )

    relative_test_size = (
        test_size
        / combined_test_size
    )

    (
        input_validation,
        input_test,
        target_validation,
        target_test
    ) = train_test_split(
        input_remaining,
        target_remaining,
        test_size=relative_test_size,
        random_state=random_seed
    )

    return (
        input_train,
        input_validation,
        input_test,
        target_train,
        target_validation,
        target_test
    )


def fit_scalers(
        input_train,
        target_train
):
    input_scaler = StandardScaler()

    target_scaler = StandardScaler()

    input_scaler.fit(
        input_train
    )

    target_scaler.fit(
        target_train
    )

    return (
        input_scaler,
        target_scaler
    )


def save_scaler(
        scaler,
        output_file
):
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
        "number_features": int(
            scaler.n_features_in_
        )
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            scaler_data,
            file,
            indent=4
        )


def create_data_loader(
        input_values,
        target_values,
        shuffle
):
    input_tensor = torch.tensor(
        input_values,
        dtype=torch.float32
    )

    target_tensor = torch.tensor(
        target_values,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        input_tensor,
        target_tensor
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )


def calculate_metrics(
        predictions,
        targets
):
    mse = float(
        np.mean(
            (
                predictions
                - targets
            ) ** 2
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
                predictions
                - targets
            )
        )
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }


def evaluate_model(
        model,
        data_loader,
        loss_function
):
    model.eval()

    total_loss = 0.0
    number_samples = 0

    prediction_batches = []
    target_batches = []

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(
                device
            )

            target_batch = target_batch.to(
                device
            )

            predictions = model(
                input_batch
            )

            loss = loss_function(
                predictions,
                target_batch
            )

            current_batch_size = input_batch.shape[
                0
            ]

            total_loss += (
                float(
                    loss.item()
                )
                * current_batch_size
            )

            number_samples += current_batch_size

            prediction_batches.append(
                predictions.cpu().numpy()
            )

            target_batches.append(
                target_batch.cpu().numpy()
            )

    average_loss = (
        total_loss
        / max(
            number_samples,
            1
        )
    )

    predictions = np.concatenate(
        prediction_batches,
        axis=0
    )

    targets = np.concatenate(
        target_batches,
        axis=0
    )

    return (
        average_loss,
        predictions,
        targets
    )


def save_model(
        model,
        optimizer,
        epoch,
        validation_loss,
        output_file
):
    torch.save(
        {
            "epoch": int(
                epoch
            ),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": float(
                validation_loss
            ),
            "input_size": len(
                input_columns
            ),
            "output_size": len(
                target_columns
            ),
            "wavelengths": wavelengths,
            "input_columns": input_columns,
            "target_columns": target_columns,
            "residual_learning": True,
            "normalization": "LayerNorm",
            "loss_function": "SmoothL1Loss",
            "smooth_l1_beta": 0.1,
            "hidden_sizes": [
                hidden_size_1,
                hidden_size_2,
                hidden_size_3
            ],
            "dropout_probability": dropout_probability
        },
        output_file
    )


def plot_training_history(
        history_dataframe
):
    figure = plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "training_loss"
        ],
        label="Training Loss"
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "validation_loss"
        ],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Counter Propagation Residual Training"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        training_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def plot_mse_history(
        history_dataframe
):
    figure = plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "training_mse"
        ],
        label="Training MSE"
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "validation_mse"
        ],
        label="Validation MSE"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean Squared Error"
    )

    plt.title(
        "CPNN Final-Spectrum Training and Validation MSE"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        mse_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def plot_combined_metrics(
        history_dataframe
):
    figure = plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "training_loss"
        ],
        label="Training Loss"
    )

    plt.plot(
        history_dataframe[
            "epoch"
        ],
        history_dataframe[
            "validation_loss"
        ],
        label="Validation Loss"
    )

    final_training_mse = float(
        history_dataframe[
            "training_mse"
        ].iloc[
            -1
        ]
    )

    final_validation_mse = float(
        history_dataframe[
            "validation_mse"
        ].iloc[
            -1
        ]
    )

    metrics_text = (
        f"Final training MSE: {final_training_mse:.6f}\n"
        f"Final validation MSE: {final_validation_mse:.6f}"
    )

    plt.text(
        0.98,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.2
        }
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Scaled Smooth L1 Loss"
    )

    plt.title(
        "CPNN Residual Training vs Validation Loss"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        combined_metrics_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def plot_prediction_example(
        predicted_spectrum,
        experimental_spectrum
):
    figure = plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        wavelengths,
        experimental_spectrum,
        label="Experimental"
    )

    plt.plot(
        wavelengths,
        predicted_spectrum,
        label="Residual-Corrected CPNN Prediction"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Absorbance"
    )

    plt.title(
        "Residual-Corrected vs Experimental UV-Vis Spectrum"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        predicted_spectrum_plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def main():
    set_random_seed(
        random_seed
    )

    (
        dataframe,
        input_values,
        target_values
    ) = load_training_data()

    (
        input_train,
        input_validation,
        input_test,
        target_train,
        target_validation,
        target_test
    ) = split_training_data(
        input_values=input_values,
        target_values=target_values
    )

    (
        input_scaler,
        target_scaler
    ) = fit_scalers(
        input_train=input_train,
        target_train=target_train
    )

    save_scaler(
        scaler=input_scaler,
        output_file=input_scaler_file
    )

    save_scaler(
        scaler=target_scaler,
        output_file=target_scaler_file
    )

    input_train_scaled = input_scaler.transform(
        input_train
    ).astype(
        np.float32
    )

    input_validation_scaled = input_scaler.transform(
        input_validation
    ).astype(
        np.float32
    )

    input_test_scaled = input_scaler.transform(
        input_test
    ).astype(
        np.float32
    )

    target_train_scaled = target_scaler.transform(
        target_train
    ).astype(
        np.float32
    )

    target_validation_scaled = target_scaler.transform(
        target_validation
    ).astype(
        np.float32
    )

    target_test_scaled = target_scaler.transform(
        target_test
    ).astype(
        np.float32
    )

    training_loader = create_data_loader(
        input_values=input_train_scaled,
        target_values=target_train_scaled,
        shuffle=True
    )

    validation_loader = create_data_loader(
        input_values=input_validation_scaled,
        target_values=target_validation_scaled,
        shuffle=False
    )

    test_loader = create_data_loader(
        input_values=input_test_scaled,
        target_values=target_test_scaled,
        shuffle=False
    )

    model = CounterPropagationNetwork(
        input_size=len(
            input_columns
        ),
        output_size=len(
            target_columns
        )
    ).to(
        device
    )

    loss_function = nn.SmoothL1Loss(
        beta=0.1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6
    )

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    training_start_time = time.time()

    print()
    print(
        "Counter Propagation Residual Training"
    )
    print(
        "----------------------------"
    )
    print(
        "Device:",
        device
    )
    print(
        "Rows:",
        len(
            dataframe
        )
    )
    print(
        "Training rows:",
        len(
            input_train
        )
    )
    print(
        "Validation rows:",
        len(
            input_validation
        )
    )
    print(
        "Test rows:",
        len(
            input_test
        )
    )
    print(
        "Input features:",
        len(
            input_columns
        )
    )
    print(
        "Output values:",
        len(
            target_columns
        )
    )
    print(
        "Target:",
        "experimental spectrum - MPNN spectrum"
    )
    print(
        "Normalization:",
        "LayerNorm"
    )
    print(
        "Loss:",
        "SmoothL1Loss(beta=0.1)"
    )
    print()

    epoch_progress = tqdm(
        range(
            1,
            number_epochs + 1
        ),
        desc="Training",
        unit="epoch"
    )

    for epoch in epoch_progress:
        epoch_start_time = time.time()

        model.train()

        total_training_loss = 0.0
        number_training_samples = 0
        training_prediction_batches = []
        training_target_batches = []

        batch_progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch}/{number_epochs}",
            leave=False,
            unit="batch"
        )

        for input_batch, target_batch in batch_progress:
            input_batch = input_batch.to(
                device
            )

            target_batch = target_batch.to(
                device
            )

            optimizer.zero_grad()

            predictions = model(
                input_batch
            )

            loss = loss_function(
                predictions,
                target_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            current_batch_size = input_batch.shape[
                0
            ]

            total_training_loss += (
                float(
                    loss.item()
                )
                * current_batch_size
            )

            number_training_samples += current_batch_size

            training_prediction_batches.append(
                predictions.detach().cpu().numpy()
            )

            training_target_batches.append(
                target_batch.detach().cpu().numpy()
            )

            batch_mse = float(
                torch.mean(
                    (
                        predictions
                        - target_batch
                    ) ** 2
                ).item()
            )

            batch_progress.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "mse": f"{batch_mse:.6f}"
                }
            )

        training_loss = (
            total_training_loss
            / max(
                number_training_samples,
                1
            )
        )

        training_predictions_scaled = np.concatenate(
            training_prediction_batches,
            axis=0
        )

        training_targets_scaled = np.concatenate(
            training_target_batches,
            axis=0
        )

        training_predictions = target_scaler.inverse_transform(
            training_predictions_scaled
        )

        training_targets = target_scaler.inverse_transform(
            training_targets_scaled
        )

        training_metrics = calculate_metrics(
            predictions=training_predictions,
            targets=training_targets
        )

        (
            validation_loss,
            validation_predictions_scaled,
            validation_targets_scaled
        ) = evaluate_model(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function
        )

        scheduler.step(
            validation_loss
        )

        validation_predictions = target_scaler.inverse_transform(
            validation_predictions_scaled
        )

        validation_targets = target_scaler.inverse_transform(
            validation_targets_scaled
        )

        validation_metrics = calculate_metrics(
            predictions=validation_predictions,
            targets=validation_targets
        )

        epoch_seconds = (
            time.time()
            - epoch_start_time
        )

        current_learning_rate = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_mse": training_metrics[
                    "mse"
                ],
                "training_rmse": training_metrics[
                    "rmse"
                ],
                "training_mae": training_metrics[
                    "mae"
                ],
                "validation_mse": validation_metrics[
                    "mse"
                ],
                "validation_rmse": validation_metrics[
                    "rmse"
                ],
                "validation_mae": validation_metrics[
                    "mae"
                ],
                "learning_rate": current_learning_rate,
                "epoch_seconds": epoch_seconds
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            epochs_without_improvement = 0

            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                output_file=best_model_file
            )

        else:
            epochs_without_improvement += 1

        elapsed_training_seconds = (
            time.time()
            - training_start_time
        )

        average_epoch_seconds = (
            elapsed_training_seconds
            / epoch
        )

        remaining_seconds = (
            number_epochs
            - epoch
        ) * average_epoch_seconds

        epoch_progress.set_postfix(
            {
                "train_loss": f"{training_loss:.6f}",
                "val_loss": f"{validation_loss:.6f}",
                "train_mse": f"{training_metrics['mse']:.6f}",
                "val_mse": f"{validation_metrics['mse']:.6f}",
                "best": f"{best_validation_loss:.6f}",
                "time": f"{epoch_seconds:.1f}s",
                "eta": f"{remaining_seconds / 60:.1f}m"
            }
        )

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            print()
            print(
                "Early stopping at epoch:",
                epoch
            )
            break

    epoch_progress.close()

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        training_history_file,
        index=False
    )

    checkpoint = torch.load(
        best_model_file,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        test_loss,
        test_predictions_scaled,
        test_targets_scaled
    ) = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function
    )

    test_predicted_corrections = target_scaler.inverse_transform(
        test_predictions_scaled
    )

    test_target_corrections = target_scaler.inverse_transform(
        test_targets_scaled
    )

    # Reconstruct the complete spectra.
    test_predictions = (
        input_test
        + test_predicted_corrections
    )

    test_targets = (
        input_test
        + test_target_corrections
    )

    test_metrics = calculate_metrics(
        predictions=test_predictions,
        targets=test_targets
    )

    save_model(
        model=model,
        optimizer=optimizer,
        epoch=checkpoint[
            "epoch"
        ],
        validation_loss=checkpoint[
            "validation_loss"
        ],
        output_file=final_model_file
    )

    plot_training_history(
        history_dataframe
    )

    plot_mse_history(
        history_dataframe
    )

    plot_combined_metrics(
        history_dataframe
    )

    plot_prediction_example(
        predicted_spectrum=test_predictions[
            0
        ],
        experimental_spectrum=test_targets[
            0
        ]
    )

    predicted_lambda_max = float(
        wavelengths[
            int(
                np.argmax(
                    test_predictions[
                        0
                    ]
                )
            )
        ]
    )

    experimental_lambda_max = float(
        wavelengths[
            int(
                np.argmax(
                    test_targets[
                        0
                    ]
                )
            )
        ]
    )

    total_training_seconds = (
        time.time()
        - training_start_time
    )

    print()
    print(
        "Training finished"
    )
    print(
        "Best epoch:",
        checkpoint[
            "epoch"
        ]
    )
    print(
        "Best validation loss:",
        checkpoint[
            "validation_loss"
        ]
    )
    print(
        "Test scaled loss:",
        test_loss
    )
    print(
        "Test MSE:",
        test_metrics[
            "mse"
        ]
    )
    print(
        "Test RMSE:",
        test_metrics[
            "rmse"
        ]
    )
    print(
        "Test MAE:",
        test_metrics[
            "mae"
        ]
    )
    print(
        "Example predicted lambda max:",
        predicted_lambda_max
    )
    print(
        "Example experimental lambda max:",
        experimental_lambda_max
    )
    print(
        "Total training minutes:",
        round(
            total_training_seconds
            / 60,
            2
        )
    )
    print()
    print(
        "Best model:",
        best_model_file
    )
    print(
        "Final model:",
        final_model_file
    )
    print(
        "Training history:",
        training_history_file
    )
    print(
        "Training plot:",
        training_plot_file
    )
    print(
        "MSE plot:",
        mse_plot_file
    )
    print(
        "Combined metrics plot:",
        combined_metrics_plot_file
    )
    print(
        "Spectrum plot:",
        predicted_spectrum_plot_file
    )
    print(
        "Residual learning:",
        True
    )


if __name__ == "__main__":
    main()