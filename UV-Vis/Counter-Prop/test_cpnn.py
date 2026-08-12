import json
import os
from pathlib import Path

import numpy as np

from model_wrappers import build_model_wrappers
from model_wrappers import run_selected_models
from model_wrappers import get_successful_predictions
from model_wrappers import get_failed_predictions

project_folder = Path(__file__).resolve().parent

model_paths = {
    "DTNN": project_folder / "models" / "dtnn_model.pth",
    "MPNN": project_folder / "models" / "mpnn_model.pth",
    "SchNet": project_folder / "models" / "schnet_model.pth",
    "Transformer": project_folder / "models" / "transformer_model.pth"
}


def normalize_prediction_spectrum(prediction, target_wavelengths=None):
    spectrum = prediction.get("spectrum")
    wavelengths = prediction.get("wavelengths")
    if spectrum is None:
        return None, None

    spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)

    if target_wavelengths is not None:
        target_wavelengths = np.asarray(target_wavelengths, dtype=np.float32).reshape(-1)
        if wavelengths is None:
            if len(spectrum) != len(target_wavelengths):
                raise ValueError("The spectrum length does not match the target wavelength grid.")
            wavelengths = target_wavelengths
        elif len(wavelengths) != len(target_wavelengths) or not np.allclose(wavelengths, target_wavelengths):
            spectrum = np.interp(target_wavelengths, wavelengths, spectrum).astype(np.float32)
            wavelengths = target_wavelengths

    maximum = float(np.max(np.abs(spectrum)))
    if maximum > 0:
        spectrum = spectrum / maximum
    return wavelengths, spectrum


def average_predictions(predictions, target_wavelengths=None):
    successful_predictions = get_successful_predictions(predictions)
    if not successful_predictions:
        return {
            "model_name": "Ensemble",
            "wavelengths": None,
            "spectrum": None,
            "lambda_max": None,
            "confidence": None,
            "uncertainty": None,
            "status": "failed",
            "error": "No successful model predictions were available."
        }

    aligned_spectra = []
    prediction_weights = []
    model_names = []
    wavelengths = None

    for prediction in successful_predictions:
        aligned_wavelengths, aligned_spectrum = normalize_prediction_spectrum(
            prediction=prediction,
            target_wavelengths=target_wavelengths
        )
        if aligned_spectrum is None:
            continue
        if wavelengths is None:
            wavelengths = aligned_wavelengths
        confidence = prediction.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else 1.0
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(confidence, 0.01)
        aligned_spectra.append(aligned_spectrum)
        prediction_weights.append(confidence)
        model_names.append(prediction.get("model_name", "Unknown"))

    if not aligned_spectra:
        return {
            "model_name": "Ensemble",
            "wavelengths": None,
            "spectrum": None,
            "lambda_max": None,
            "confidence": None,
            "uncertainty": None,
            "status": "failed",
            "error": "No valid spectra were available for the ensemble."
        }

    aligned_spectra = np.asarray(aligned_spectra, dtype=np.float32)
    prediction_weights = np.asarray(prediction_weights, dtype=np.float32)
    prediction_weights = prediction_weights / np.sum(prediction_weights)
    ensemble_spectrum = np.average(aligned_spectra, axis=0, weights=prediction_weights)
    ensemble_uncertainty = float(np.mean(np.std(aligned_spectra, axis=0)))

    lambda_max = None
    if wavelengths is not None and len(wavelengths) == len(ensemble_spectrum):
        lambda_max = float(wavelengths[int(np.argmax(ensemble_spectrum))])

    ensemble_confidence = float(np.clip(
        1.0 / (1.0 + ensemble_uncertainty),
        0.0,
        1.0
    ))

    return {
        "model_name": "Ensemble",
        "source_models": model_names,
        "wavelengths": None if wavelengths is None else wavelengths.tolist(),
        "spectrum": ensemble_spectrum.tolist(),
        "lambda_max": lambda_max,
        "confidence": ensemble_confidence,
        "uncertainty": ensemble_uncertainty,
        "status": "success",
        "error": None
    }


def run_prediction(
        smiles,
        routing_output,
        model_features=None,
        wrappers=None,
        preprocessors=None,
        model_builders=None,
        postprocessors=None,
        device=None,
        target_wavelengths=None,
        use_counter_prop=False,
        counter_prop_predictor=None
):
    if wrappers is None:
        wrappers = build_model_wrappers(
            model_paths={model_name: str(path) for model_name, path in model_paths.items()},
            model_builders=model_builders,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            device=device
        )

    selected_models = routing_output.get("selected_models", [])
    predictions = run_selected_models(
        smiles=smiles,
        selected_models=selected_models,
        wrappers=wrappers,
        model_features=model_features
    )
    successful_predictions = get_successful_predictions(predictions)
    failed_predictions = get_failed_predictions(predictions)
    ensemble_prediction = average_predictions(
        predictions=predictions,
        target_wavelengths=target_wavelengths
    )

    counter_prop_prediction = None
    if use_counter_prop:
        if counter_prop_predictor is None:
            counter_prop_prediction = {
                "model_name": "CounterPropagation",
                "status": "failed",
                "error": "No counter-propagation predictor was provided."
            }
        else:
            counter_prop_prediction = counter_prop_predictor.predict(
                model_features=model_features,
                predictions=predictions
            )

    final_prediction = ensemble_prediction
    if counter_prop_prediction is not None and counter_prop_prediction.get("status") == "success":
        final_prediction = counter_prop_prediction

    return {
        "smiles": smiles,
        "routing_output": routing_output,
        "model_predictions": predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "ensemble_prediction": ensemble_prediction,
        "counter_prop_prediction": counter_prop_prediction,
        "final_prediction": final_prediction
    }


def save_prediction(prediction_output, output_file="latest_prediction.json"):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(prediction_output, file, indent=4, ensure_ascii=False)
    print("Prediction saved:", output_file)


def main():
    smiles = input("Enter a SMILES string: ").strip()
    model_features_file = "latest_model_features.json"
    routing_file = "latest_routing_output.json"

    if not os.path.exists(model_features_file):
        raise FileNotFoundError(f"File not found: {model_features_file}")
    if not os.path.exists(routing_file):
        raise FileNotFoundError(f"File not found: {routing_file}")

    with open(model_features_file, "r", encoding="utf-8") as file:
        model_features = json.load(file)
    with open(routing_file, "r", encoding="utf-8") as file:
        routing_output = json.load(file)

    prediction_output = run_prediction(
        smiles=smiles,
        routing_output=routing_output,
        model_features=model_features,
        target_wavelengths=np.arange(220, 401, dtype=np.float32)
    )
    save_prediction(prediction_output)
    print(json.dumps(prediction_output, indent=4))


if __name__ == "__main__":
    main()