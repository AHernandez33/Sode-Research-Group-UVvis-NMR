from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn

from knowledge_base import build_knowledge_base
from llm_qwen import ask_llm
from model_features import build_model_features
from model_features import save_model_features
from router import MODELS
from router import route_models
from predict import run_prediction


PROJECT_FOLDER = Path(__file__).resolve().parent
UVVIS_FOLDER = PROJECT_FOLDER.parent
COUNTER_PROP_FOLDER = UVVIS_FOLDER / "Counter-Prop"

AGENT_OUTPUT_FILE = PROJECT_FOLDER / "latest_agent_output.json"
MODEL_FEATURES_OUTPUT_FILE = PROJECT_FOLDER / "latest_model_features.json"
ROUTING_OUTPUT_FILE = PROJECT_FOLDER / "latest_routing_output.json"
PREDICTION_OUTPUT_FILE = PROJECT_FOLDER / "latest_prediction.json"

CPNN_MODEL_FILE = COUNTER_PROP_FOLDER / "models" / "counter_prop_best.pth"
CPNN_INPUT_SCALER_FILE = (
    COUNTER_PROP_FOLDER
    / "models"
    / "counter_prop_input_scaler.json"
)
CPNN_TARGET_SCALER_FILE = (
    COUNTER_PROP_FOLDER
    / "models"
    / "counter_prop_target_scaler.json"
)

WAVELENGTHS = np.arange(
    220,
    401,
    dtype=np.float32
)

HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128
HIDDEN_SIZE_3 = 128
DROPOUT_PROBABILITY = 0.0


def save_json(
    data: Dict[str, Any],
    output_file: Path
) -> None:
    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )



def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the UV-Vis agent and optionally evaluate the prediction "
            "against an experimental spectrum."
        )
    )

    parser.add_argument(
        "--smiles",
        type=str,
        default=None
    )

    parser.add_argument(
        "--experimental-csv",
        type=str,
        default=None
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    parser.add_argument(
        "--max-models",
        type=int,
        default=1
    )

    parser.add_argument(
        "--no-cpnn",
        action="store_true"
    )

    return parser


def normalize_spectrum(
    spectrum
):
    if spectrum is None:
        return None

    values = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    if values.shape[
        0
    ] != 181:
        return None

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimum = float(
        values.min()
    )

    if minimum < 0:
        values = (
            values
            - minimum
        )

    maximum = float(
        values.max()
    )

    if maximum > 0:
        values = (
            values
            / maximum
        )

    return values.astype(
        np.float32
    )


def calculate_evaluation_metrics(
    predicted_spectrum,
    experimental_spectrum
):
    difference = (
        predicted_spectrum
        - experimental_spectrum
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

    predicted_lambda_max = float(
        WAVELENGTHS[
            int(
                np.argmax(
                    predicted_spectrum
                )
            )
        ]
    )

    experimental_lambda_max = float(
        WAVELENGTHS[
            int(
                np.argmax(
                    experimental_spectrum
                )
            )
        ]
    )

    return {
        "spectrum_mse": mse,
        "spectrum_rmse": rmse,
        "spectrum_mae": mae,
        "predicted_lambda_max": predicted_lambda_max,
        "experimental_lambda_max": experimental_lambda_max,
        "lambda_max_absolute_error": float(
            abs(
                predicted_lambda_max
                - experimental_lambda_max
            )
        )
    }


def find_smiles_column(
    dataframe
):
    possible_columns = [
        "smiles",
        "SMILES",
        "canonical_smiles",
        "molecule_smiles"
    ]

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    for column in dataframe.columns:
        if "smiles" in str(
            column
        ).lower():
            return column

    raise ValueError(
        "No SMILES column was found in the experimental CSV."
    )


def find_experimental_columns(
    dataframe
):
    integer_wavelengths = WAVELENGTHS.astype(
        int
    )

    possible_column_sets = [
        [
            str(
                wavelength
            )
            for wavelength in integer_wavelengths
        ],
        [
            f"experimental_{wavelength}"
            for wavelength in integer_wavelengths
        ],
        [
            f"absorbance_{wavelength}"
            for wavelength in integer_wavelengths
        ]
    ]

    for columns in possible_column_sets:
        if all(
            column in dataframe.columns
            for column in columns
        ):
            return columns

    raise ValueError(
        "Could not find 181 experimental columns from 220 to 400 nm."
    )


def canonicalize_smiles(
    smiles
):
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(
            str(
                smiles
            )
        )

        if molecule is None:
            return str(
                smiles
            ).strip()

        return Chem.MolToSmiles(
            molecule,
            canonical=True
        )

    except Exception:
        return str(
            smiles
        ).strip()


def load_experimental_spectrum(
    csv_file,
    canonical_smiles
):
    csv_file = Path(
        csv_file
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Experimental CSV not found: {csv_file}"
        )

    dataframe = pd.read_csv(
        csv_file
    )

    smiles_column = find_smiles_column(
        dataframe
    )

    experimental_columns = find_experimental_columns(
        dataframe
    )

    dataframe[
        "_canonical_smiles"
    ] = dataframe[
        smiles_column
    ].astype(
        str
    ).map(
        canonicalize_smiles
    )

    matching_rows = dataframe[
        dataframe[
            "_canonical_smiles"
        ].eq(
            canonical_smiles
        )
    ]

    if matching_rows.empty:
        raise ValueError(
            "The input molecule was not found in the experimental CSV."
        )

    selected_row = matching_rows.iloc[
        0
    ]

    spectrum = selected_row[
        experimental_columns
    ].to_numpy(
        dtype=np.float32
    )

    return spectrum



def build_default_llm_analysis(
    canonical_smiles: str,
    error: Optional[Exception] = None
) -> Dict[str, Any]:
    warnings = []

    if error is not None:
        warnings.append(
            f"LLM analysis failed: {error}"
        )

    return {
        "query_smiles": canonical_smiles,
        "retrieval_confidence": "unknown",
        "reasoning": "",
        "functional_groups": [],
        "chromophores": [],
        "auxochromes": [],
        "aromatic": False,
        "conjugated": False,
        "retrieval_analysis": "",
        "routing_notes": "",
        "warnings": warnings,
        "recommended_models": [],
        "descriptor_importance": [],
        "uncertainty_notes": []
    }


class SimpleCounterPropagationNetwork(
    nn.Module
):
    def __init__(
        self,
        input_size: int = 181,
        output_size: int = 181
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                HIDDEN_SIZE_1
            ),
            nn.ReLU(),
            nn.LayerNorm(
                HIDDEN_SIZE_1
            ),
            nn.Dropout(
                DROPOUT_PROBABILITY
            ),
            nn.Linear(
                HIDDEN_SIZE_1,
                HIDDEN_SIZE_2
            ),
            nn.ReLU(),
            nn.LayerNorm(
                HIDDEN_SIZE_2
            ),
            nn.Dropout(
                DROPOUT_PROBABILITY
            ),
            nn.Linear(
                HIDDEN_SIZE_2,
                HIDDEN_SIZE_3
            ),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_SIZE_3,
                output_size
            )
        )

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.network(
            inputs
        )


def load_scaler(
    scaler_file: Path
) -> Dict[str, np.ndarray]:
    if not scaler_file.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_file}"
        )

    with open(
        scaler_file,
        "r",
        encoding="utf-8"
    ) as file:
        state = json.load(
            file
        )

    mean = np.asarray(
        state[
            "mean"
        ],
        dtype=np.float32
    )

    scale = np.asarray(
        state[
            "scale"
        ],
        dtype=np.float32
    )

    scale = np.where(
        scale == 0,
        1.0,
        scale
    ).astype(
        np.float32
    )

    return {
        "mean": mean,
        "scale": scale
    }


class SimpleCPNNPredictor:
    def __init__(
        self,
        model_file: Path = CPNN_MODEL_FILE,
        input_scaler_file: Path = CPNN_INPUT_SCALER_FILE,
        target_scaler_file: Path = CPNN_TARGET_SCALER_FILE,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        if not model_file.exists():
            raise FileNotFoundError(
                f"CPNN model file not found: {model_file}"
            )

        checkpoint = torch.load(
            model_file,
            map_location=self.device,
            weights_only=False
        )

        input_size = int(
            checkpoint.get(
                "input_size",
                181
            )
        )

        output_size = int(
            checkpoint.get(
                "output_size",
                181
            )
        )

        self.model = SimpleCounterPropagationNetwork(
            input_size=input_size,
            output_size=output_size
        ).to(
            self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.input_scaler = load_scaler(
            input_scaler_file
        )

        self.target_scaler = load_scaler(
            target_scaler_file
        )

        self.checkpoint_epoch = checkpoint.get(
            "epoch"
        )

        self.validation_loss = checkpoint.get(
            "validation_loss"
        )

        self.residual_learning = bool(
            checkpoint.get(
                "residual_learning",
                False
            )
        )

    def predict(
        self,
        model_features: Optional[Dict[str, Any]],
        predictions: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        mpnn_prediction = None

        for prediction in predictions:
            if (
                prediction.get(
                    "model_name"
                ) == "MPNN"
                and prediction.get(
                    "status"
                ) == "success"
            ):
                mpnn_prediction = prediction
                break

        if mpnn_prediction is None:
            return {
                "model_name": "CounterPropagation",
                "status": "skipped",
                "error": (
                    "No successful MPNN prediction was available. "
                    "The current CPNN was trained only on MPNN spectra."
                )
            }

        spectrum = np.asarray(
            mpnn_prediction.get(
                "spectrum"
            ),
            dtype=np.float32
        ).reshape(
            -1
        )

        if spectrum.shape[
            0
        ] != 181:
            return {
                "model_name": "CounterPropagation",
                "status": "failed",
                "error": (
                    "The MPNN spectrum must contain 181 values. "
                    f"Found {spectrum.shape[0]}."
                )
            }

        scaled_input = (
            spectrum
            - self.input_scaler[
                "mean"
            ]
        ) / self.input_scaler[
            "scale"
        ]

        input_tensor = torch.tensor(
            scaled_input.reshape(
                1,
                -1
            ),
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            scaled_output = self.model(
                input_tensor
            ).cpu().numpy()[
                0
            ]

        predicted_output = (
            scaled_output
            * self.target_scaler[
                "scale"
            ]
            + self.target_scaler[
                "mean"
            ]
        ).astype(
            np.float32
        )

        if self.residual_learning:
            corrected_spectrum = (
                spectrum
                + predicted_output
            )
        else:
            corrected_spectrum = predicted_output

        corrected_spectrum = np.nan_to_num(
            corrected_spectrum,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        minimum = float(
            corrected_spectrum.min()
        )

        if minimum < 0:
            corrected_spectrum = (
                corrected_spectrum
                - minimum
            )

        maximum = float(
            corrected_spectrum.max()
        )

        if maximum > 0:
            corrected_spectrum = (
                corrected_spectrum
                / maximum
            )

        lambda_max = float(
            WAVELENGTHS[
                int(
                    np.argmax(
                        corrected_spectrum
                    )
                )
            ]
        )

        try:
            uncertainty = float(
                self.validation_loss
            )
            confidence = float(
                np.clip(
                    1.0
                    / (
                        1.0
                        + max(
                            uncertainty,
                            0.0
                        )
                    ),
                    0.0,
                    1.0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            uncertainty = None
            confidence = None

        return {
            "model_name": "CounterPropagation",
            "source_models": [
                "MPNN"
            ],
            "wavelengths": WAVELENGTHS.tolist(),
            "spectrum": corrected_spectrum.tolist(),
            "lambda_max": lambda_max,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "status": "success",
            "error": None,
            "checkpoint": str(
                CPNN_MODEL_FILE
            ),
            "checkpoint_epoch": self.checkpoint_epoch,
            "residual_learning": self.residual_learning
        }


def build_counter_prop_predictor() -> Optional[SimpleCPNNPredictor]:
    required_files = [
        CPNN_MODEL_FILE,
        CPNN_INPUT_SCALER_FILE,
        CPNN_TARGET_SCALER_FILE
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print(
            "CPNN disabled because required files are missing:"
        )

        for path in missing_files:
            print(
                "-",
                path
            )

        return None

    try:
        return SimpleCPNNPredictor()

    except Exception as error:
        print(
            "CPNN loading warning:",
            error
        )

        return None


def create_final_summary(
    agent_output: Dict[str, Any]
) -> Dict[str, Any]:
    prediction_output = agent_output.get(
        "prediction_output",
        {}
    )

    final_prediction = prediction_output.get(
        "final_prediction",
        {}
    )

    llm_analysis = agent_output.get(
        "llm_analysis",
        {}
    )

    routing_output = agent_output.get(
        "routing_output",
        {}
    )

    successful_predictions = prediction_output.get(
        "successful_predictions",
        []
    )

    failed_predictions = prediction_output.get(
        "failed_predictions",
        []
    )

    return {
        "input_smiles": agent_output.get(
            "input_smiles"
        ),
        "canonical_smiles": agent_output.get(
            "canonical_smiles"
        ),
        "runtime_seconds": agent_output.get(
            "runtime_seconds"
        ),
        "primary_model": routing_output.get(
            "primary_model"
        ),
        "selected_models": routing_output.get(
            "selected_models",
            []
        ),
        "routing_confidence": routing_output.get(
            "routing_confidence"
        ),
        "successful_models": [
            prediction.get(
                "model_name"
            )
            for prediction in successful_predictions
        ],
        "failed_models": [
            prediction.get(
                "model_name"
            )
            for prediction in failed_predictions
        ],
        "final_model": final_prediction.get(
            "model_name"
        ),
        "prediction_status": final_prediction.get(
            "status"
        ),
        "lambda_max": final_prediction.get(
            "lambda_max"
        ),
        "confidence": final_prediction.get(
            "confidence"
        ),
        "uncertainty": final_prediction.get(
            "uncertainty"
        ),
        "checkpoint_epoch": final_prediction.get(
            "checkpoint_epoch"
        ),
        "retrieval_confidence": llm_analysis.get(
            "retrieval_confidence",
            "unknown"
        ),
        "evaluation_metrics": agent_output.get(
            "evaluation_metrics"
        ),
        "warnings": llm_analysis.get(
            "warnings",
            []
        ),
        "error": final_prediction.get(
            "error"
        )
    }


def run_agent(
    smiles: str,
    top_k: int = 10,
    max_models: int = 1,
    target_wavelengths: Optional[np.ndarray] = None,
    use_llm: bool = True,
    use_counter_prop: bool = True,
    counter_prop_predictor: Optional[Any] = None,
    available_models: Optional[list[str]] = None,
    experimental_csv: Optional[Path] = None
) -> Dict[str, Any]:
    total_start_time = time.time()

    smiles = str(
        smiles
    ).strip()

    if not smiles:
        raise ValueError(
            "SMILES cannot be empty."
        )

    if target_wavelengths is None:
        target_wavelengths = WAVELENGTHS

    if available_models is None:
        available_models = list(
            MODELS
        )

    print()
    print(
        "1. Building knowledge base..."
    )

    knowledge_base = build_knowledge_base(
        smiles=smiles,
        top_k=top_k,
        save=True
    )

    query_data = knowledge_base.get(
        "query",
        {}
    )

    canonical_smiles = query_data.get(
        "canonical_smiles",
        smiles
    )

    print(
        "Canonical SMILES:",
        canonical_smiles
    )

    print()
    print(
        "2. Running LLM analysis..."
    )

    llm_error = None

    if use_llm:
        try:
            llm_analysis = ask_llm(
                knowledge_base
            )

            if not isinstance(
                llm_analysis,
                dict
            ):
                raise TypeError(
                    "ask_llm() must return a dictionary."
                )

        except Exception as error:
            llm_error = str(
                error
            )

            llm_analysis = build_default_llm_analysis(
                canonical_smiles=canonical_smiles,
                error=error
            )

            print(
                "LLM warning:",
                error
            )

    else:
        llm_analysis = build_default_llm_analysis(
            canonical_smiles=canonical_smiles
        )

    print()
    print(
        "3. Building model features..."
    )

    model_features = build_model_features(
        smiles=canonical_smiles,
        knowledge_base=knowledge_base,
        llm_output=llm_analysis
    )

    save_model_features(
        model_features=model_features,
        file=MODEL_FEATURES_OUTPUT_FILE
    )

    print()
    print(
        "4. Routing prediction..."
    )

    routing_output = route_models(
        model_features=model_features,
        max_models=max_models,
        available_models=available_models
    )

    save_json(
        data=routing_output,
        output_file=ROUTING_OUTPUT_FILE
    )

    print(
        "Selected models:",
        routing_output.get(
            "selected_models",
            []
        )
    )

    if (
        use_counter_prop
        and counter_prop_predictor is None
        and routing_output.get(
            "primary_model"
        ) == "MPNN"
    ):
        counter_prop_predictor = build_counter_prop_predictor()

    print()
    print(
        "5. Running model prediction..."
    )

    prediction_output = run_prediction(
        smiles=canonical_smiles,
        routing_output=routing_output,
        model_features=model_features,
        wrappers=None,
        target_wavelengths=target_wavelengths,
        use_counter_prop=(
            use_counter_prop
            and counter_prop_predictor is not None
        ),
        counter_prop_predictor=counter_prop_predictor
    )

    save_json(
        data=prediction_output,
        output_file=PREDICTION_OUTPUT_FILE
    )

    evaluation_metrics = None
    evaluation_error = None

    if experimental_csv is not None:
        try:
            experimental_spectrum = load_experimental_spectrum(
                csv_file=experimental_csv,
                canonical_smiles=canonical_smiles
            )

            final_prediction = prediction_output.get(
                "final_prediction",
                {}
            )

            predicted_spectrum = normalize_spectrum(
                final_prediction.get(
                    "spectrum"
                )
            )

            experimental_spectrum = normalize_spectrum(
                experimental_spectrum
            )

            if predicted_spectrum is None:
                raise ValueError(
                    "The final prediction does not contain a valid 181-point spectrum."
                )

            if experimental_spectrum is None:
                raise ValueError(
                    "The experimental spectrum does not contain 181 valid values."
                )

            evaluation_metrics = calculate_evaluation_metrics(
                predicted_spectrum=predicted_spectrum,
                experimental_spectrum=experimental_spectrum
            )

        except Exception as error:
            evaluation_error = str(
                error
            )

            print(
                "Evaluation warning:",
                error
            )

    runtime_seconds = float(
        time.time()
        - total_start_time
    )

    agent_output = {
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "runtime_seconds": runtime_seconds,
        "knowledge_base": knowledge_base,
        "llm_analysis": llm_analysis,
        "llm_error": llm_error,
        "model_features": model_features,
        "routing_output": routing_output,
        "prediction_output": prediction_output,
        "experimental_csv": (
            None
            if experimental_csv is None
            else str(
                experimental_csv
            )
        ),
        "evaluation_metrics": evaluation_metrics,
        "evaluation_error": evaluation_error
    }

    agent_output[
        "summary"
    ] = create_final_summary(
        agent_output
    )

    return agent_output


def save_agent_output(
    agent_output: Dict[str, Any],
    output_file: Path = AGENT_OUTPUT_FILE
) -> None:
    save_json(
        data=agent_output,
        output_file=output_file
    )

    print()
    print(
        "Agent output saved:",
        output_file
    )


def print_agent_summary(
    agent_output: Dict[str, Any]
) -> None:
    summary = agent_output.get(
        "summary",
        {}
    )

    print()
    print(
        "UV-Vis Agent Result"
    )
    print(
        "-------------------"
    )
    print(
        "Input SMILES:",
        summary.get(
            "input_smiles"
        )
    )
    print(
        "Canonical SMILES:",
        summary.get(
            "canonical_smiles"
        )
    )
    print(
        "Primary model:",
        summary.get(
            "primary_model"
        )
    )
    print(
        "Selected models:",
        summary.get(
            "selected_models"
        )
    )
    print(
        "Successful models:",
        summary.get(
            "successful_models"
        )
    )
    print(
        "Failed models:",
        summary.get(
            "failed_models"
        )
    )
    print(
        "Final model:",
        summary.get(
            "final_model"
        )
    )
    print(
        "Prediction status:",
        summary.get(
            "prediction_status"
        )
    )
    print(
        "Lambda max:",
        summary.get(
            "lambda_max"
        )
    )
    print(
        "Confidence:",
        summary.get(
            "confidence"
        )
    )
    print(
        "Uncertainty:",
        summary.get(
            "uncertainty"
        )
    )
    print(
        "Checkpoint epoch:",
        summary.get(
            "checkpoint_epoch"
        )
    )
    print(
        "Routing confidence:",
        summary.get(
            "routing_confidence"
        )
    )
    print(
        "Retrieval confidence:",
        summary.get(
            "retrieval_confidence"
        )
    )
    print(
        "Runtime seconds:",
        summary.get(
            "runtime_seconds"
        )
    )

    evaluation_metrics = summary.get(
        "evaluation_metrics"
    )

    if evaluation_metrics is not None:
        print()
        print(
            "Experimental Evaluation"
        )
        print(
            "-----------------------"
        )
        print(
            "Spectrum MSE:",
            evaluation_metrics.get(
                "spectrum_mse"
            )
        )
        print(
            "Spectrum RMSE:",
            evaluation_metrics.get(
                "spectrum_rmse"
            )
        )
        print(
            "Spectrum MAE:",
            evaluation_metrics.get(
                "spectrum_mae"
            )
        )
        print(
            "Predicted lambda max:",
            evaluation_metrics.get(
                "predicted_lambda_max"
            )
        )
        print(
            "Experimental lambda max:",
            evaluation_metrics.get(
                "experimental_lambda_max"
            )
        )
        print(
            "Lambda-max absolute error:",
            evaluation_metrics.get(
                "lambda_max_absolute_error"
            )
        )

    error = summary.get(
        "error"
    )

    if error:
        print(
            "Prediction error:",
            error
        )

    warnings = summary.get(
        "warnings",
        []
    )

    if warnings:
        print(
            "Warnings:"
        )

        for warning in warnings:
            print(
                "-",
                warning
            )


def main() -> None:
    args = build_parser().parse_args()

    smiles = (
        args.smiles
        if args.smiles is not None
        else input(
            "Enter a SMILES string: "
        ).strip()
    )

    experimental_csv = (
        None
        if args.experimental_csv is None
        else Path(
            args.experimental_csv
        ).expanduser().resolve()
    )

    try:
        agent_output = run_agent(
            smiles=smiles,
            top_k=args.top_k,
            max_models=args.max_models,
            use_llm=True,
            use_counter_prop=not args.no_cpnn,
            available_models=list(
                MODELS
            ),
            experimental_csv=experimental_csv
        )

        save_agent_output(
            agent_output
        )

        print_agent_summary(
            agent_output
        )

    except Exception as error:
        print()
        print(
            "Agent error:",
            error
        )
        print(
            traceback.format_exc()
        )


if __name__ == "__main__":
    main()