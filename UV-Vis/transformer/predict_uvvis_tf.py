#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile
import time

import numpy as np
import torch


PROJECT_FOLDER = Path(__file__).resolve().parent
SRC_FOLDER = PROJECT_FOLDER / "src"
MODELS_FOLDER = PROJECT_FOLDER / "models"
DEFAULT_OUTPUT = PROJECT_FOLDER / "latest_transformer_prediction.json"

if str(SRC_FOLDER) not in sys.path:
    sys.path.insert(0, str(SRC_FOLDER))

from transformer import make_model
from cpu_data_utils import construct_loader, load_data_from_df


WAVELENGTHS = np.arange(220, 401, dtype=np.float32)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict a 181-point UV-Vis spectrum with the Transformer."
    )
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--twod", action="store_true", default=True)
    return parser


def resolve_checkpoint(model_path):
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_FOLDER / path
        path = path.resolve()
    else:
        candidates = sorted(
            MODELS_FOLDER.glob("*_best.model"),
            key=lambda item: item.stat().st_mtime,
            reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"No *_best.model checkpoint was found in {MODELS_FOLDER}"
            )
        path = candidates[0]

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def create_inference_csv(smiles, folder):
    path = Path(folder) / "transformer_inference.csv"
    fieldnames = ["smiles"] + [str(int(w)) for w in WAVELENGTHS]
    row = {"smiles": smiles}
    row.update({str(int(w)): 0.0 for w in WAVELENGTHS})

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return path


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict.")

    model_parameters = dict(
        checkpoint.get("model_parameters", {})
    )

    if not model_parameters:
        raise KeyError("Checkpoint is missing model_parameters.")

    model_parameters["n_output"] = 181

    model = make_model(**model_parameters)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, model_parameters


def normalize_spectrum(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if values.shape[0] != 181:
        raise ValueError(
            f"Expected 181 predicted values, found {values.shape[0]}."
        )

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    minimum = float(values.min())
    if minimum < 0:
        values = values - minimum

    maximum = float(values.max())
    if maximum > 0:
        values = values / maximum

    return values.astype(np.float32)


def predict_transformer(smiles, checkpoint_path, device, two_d_only=True):
    start = time.time()

    with tempfile.TemporaryDirectory(
        prefix="uvvis_transformer_"
    ) as temporary_folder:
        input_file = create_inference_csv(
            smiles,
            temporary_folder
        )

        features, targets = load_data_from_df(
            str(input_file),
            one_hot_formal_charge=True,
            use_data_saving=False,
            two_d_only=two_d_only
        )

        if len(features) != 1:
            raise ValueError(
                f"Expected one valid molecule, loaded {len(features)}."
            )

        loader = construct_loader(
            features,
            targets,
            batch_size=1,
            shuffle=False
        )

        model, checkpoint, model_parameters = load_model(
            checkpoint_path,
            device
        )

        batch = next(iter(loader))
        adjacency_matrix, node_features, distance_matrix, _ = batch

        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)

        batch_mask = (
            torch.sum(torch.abs(node_features), dim=-1) != 0
        )

        with torch.no_grad():
            prediction = model(
                node_features,
                batch_mask,
                adjacency_matrix,
                distance_matrix,
                None
            )

        prediction = prediction.detach().cpu().numpy()

        if prediction.shape != (1, 181):
            raise ValueError(
                f"Expected output shape (1, 181), found {prediction.shape}."
            )

        raw_spectrum = prediction[0].astype(np.float32)
        spectrum = normalize_spectrum(raw_spectrum)

        lambda_max_index = int(np.argmax(spectrum))
        lambda_max = float(WAVELENGTHS[lambda_max_index])

        validation_metrics = checkpoint.get(
            "validation_metrics",
            {}
        )
        validation_rmse = validation_metrics.get("rmse")

        try:
            uncertainty = float(validation_rmse)
            confidence = float(
                np.clip(
                    1.0 / (1.0 + max(uncertainty, 0.0)),
                    0.0,
                    1.0
                )
            )
        except (TypeError, ValueError):
            uncertainty = None
            confidence = None

        return {
            "model_name": "Transformer",
            "smiles": smiles,
            "wavelengths": WAVELENGTHS.tolist(),
            "spectrum": spectrum.tolist(),
            "raw_spectrum": raw_spectrum.tolist(),
            "lambda_max": lambda_max,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "status": "success",
            "error": None,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "validation_metrics": validation_metrics,
            "model_parameters": model_parameters,
            "device": str(device),
            "runtime_seconds": float(time.time() - start)
        }


def save_json(result, output_path):
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_FOLDER / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    return path


def save_csv(result, output_path):
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_FOLDER / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["smiles"]
        + [str(int(w)) for w in WAVELENGTHS]
        + [
            "lambda_max",
            "confidence",
            "uncertainty",
            "status",
            "error",
            "checkpoint_epoch"
        ]
    )

    row = {"smiles": result["smiles"]}

    for wavelength, absorbance in zip(
        WAVELENGTHS,
        result["spectrum"]
    ):
        row[str(int(wavelength))] = absorbance

    row.update({
        "lambda_max": result.get("lambda_max"),
        "confidence": result.get("confidence"),
        "uncertainty": result.get("uncertainty"),
        "status": result.get("status"),
        "error": result.get("error"),
        "checkpoint_epoch": result.get("checkpoint_epoch")
    })

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return path


def main():
    args = build_parser().parse_args()

    smiles = args.smiles
    if smiles is None:
        smiles = input("Enter a SMILES string: ").strip()
    else:
        smiles = str(smiles).strip()

    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    checkpoint_path = None

    try:
        checkpoint_path = resolve_checkpoint(args.model)

        result = predict_transformer(
            smiles=smiles,
            checkpoint_path=checkpoint_path,
            device=device,
            two_d_only=args.twod
        )

    except Exception as error:
        result = {
            "model_name": "Transformer",
            "smiles": smiles,
            "wavelengths": None,
            "spectrum": None,
            "raw_spectrum": None,
            "lambda_max": None,
            "confidence": None,
            "uncertainty": None,
            "status": "failed",
            "error": str(error),
            "checkpoint": (
                None
                if checkpoint_path is None
                else str(checkpoint_path)
            ),
            "checkpoint_epoch": None,
            "validation_metrics": None,
            "model_parameters": None,
            "device": str(device),
            "runtime_seconds": None
        }

    output_path = save_json(result, args.output)

    print(json.dumps(result, indent=4, ensure_ascii=False))
    print()
    print("Prediction saved:", output_path)

    if args.csv_output and result["status"] == "success":
        csv_path = save_csv(result, args.csv_output)
        print("Prediction CSV saved:", csv_path)


if __name__ == "__main__":
    main()