from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from model_wrappers import (
    build_model_wrappers,
    get_failed_predictions,
    get_successful_predictions,
    run_selected_models,
)


PROJECT_FOLDER = Path(__file__).resolve().parent
TARGET_WAVELENGTHS = np.arange(220, 401, dtype=np.float32)


def normalize_prediction_spectrum(
    prediction: Dict[str, Any],
    target_wavelengths: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    spectrum = prediction.get("spectrum")
    wavelengths = prediction.get("wavelengths")

    if spectrum is None:
        return None, None

    spectrum_array = np.asarray(spectrum, dtype=np.float32).reshape(-1)

    if spectrum_array.size == 0:
        return None, None

    wavelength_array = (
        None
        if wavelengths is None
        else np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    )

    if target_wavelengths is not None:
        target = np.asarray(target_wavelengths, dtype=np.float32).reshape(-1)

        if wavelength_array is None:
            if len(spectrum_array) != len(target):
                raise ValueError(
                    "Spectrum length does not match the target wavelength grid."
                )
            wavelength_array = target

        elif (
            len(wavelength_array) != len(target)
            or not np.allclose(wavelength_array, target)
        ):
            if len(wavelength_array) != len(spectrum_array):
                raise ValueError(
                    "Wavelength and spectrum lengths do not match."
                )

            order = np.argsort(wavelength_array)
            spectrum_array = np.interp(
                target,
                wavelength_array[order],
                spectrum_array[order],
            ).astype(np.float32)
            wavelength_array = target

    spectrum_array = np.nan_to_num(
        spectrum_array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(np.min(spectrum_array))
    if minimum < 0:
        spectrum_array = spectrum_array - minimum

    maximum = float(np.max(np.abs(spectrum_array)))
    if maximum > 0:
        spectrum_array = spectrum_array / maximum

    return wavelength_array, spectrum_array.astype(np.float32)


def average_predictions(
    predictions: Iterable[Dict[str, Any]],
    target_wavelengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    successful = get_successful_predictions(predictions)

    if not successful:
        return {
            "model_name": "Ensemble",
            "source_models": [],
            "wavelengths": None,
            "spectrum": None,
            "lambda_max": None,
            "confidence": None,
            "uncertainty": None,
            "status": "failed",
            "error": "No successful model predictions were available.",
        }

    aligned_spectra: List[np.ndarray] = []
    weights: List[float] = []
    source_models: List[str] = []
    wavelengths: Optional[np.ndarray] = None

    for prediction in successful:
        aligned_wavelengths, aligned_spectrum = normalize_prediction_spectrum(
            prediction=prediction,
            target_wavelengths=target_wavelengths,
        )

        if aligned_spectrum is None:
            continue

        if wavelengths is None:
            wavelengths = aligned_wavelengths

        confidence = prediction.get("confidence")

        try:
            confidence_value = (
                float(confidence)
                if confidence is not None
                else 1.0
            )
        except (TypeError, ValueError):
            confidence_value = 1.0

        aligned_spectra.append(aligned_spectrum)
        weights.append(max(confidence_value, 0.01))
        source_models.append(prediction.get("model_name", "Unknown"))

    if not aligned_spectra:
        return {
            "model_name": "Ensemble",
            "source_models": [],
            "wavelengths": None,
            "spectrum": None,
            "lambda_max": None,
            "confidence": None,
            "uncertainty": None,
            "status": "failed",
            "error": "No valid spectra were available for the ensemble.",
        }

    stacked = np.asarray(aligned_spectra, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array = weight_array / np.sum(weight_array)

    ensemble_spectrum = np.average(
        stacked,
        axis=0,
        weights=weight_array,
    ).astype(np.float32)

    ensemble_uncertainty = (
        float(np.mean(np.std(stacked, axis=0)))
        if len(stacked) > 1
        else float(
            successful[0].get("uncertainty")
            if successful[0].get("uncertainty") is not None
            else 0.0
        )
    )

    lambda_max = None
    if wavelengths is not None:
        lambda_max = float(
            wavelengths[int(np.argmax(ensemble_spectrum))]
        )

    ensemble_confidence = float(
        np.clip(
            1.0 / (1.0 + max(ensemble_uncertainty, 0.0)),
            0.0,
            1.0,
        )
    )

    return {
        "model_name": (
            source_models[0]
            if len(source_models) == 1
            else "Ensemble"
        ),
        "source_models": source_models,
        "wavelengths": (
            None
            if wavelengths is None
            else wavelengths.tolist()
        ),
        "spectrum": ensemble_spectrum.tolist(),
        "lambda_max": lambda_max,
        "confidence": ensemble_confidence,
        "uncertainty": ensemble_uncertainty,
        "status": "success",
        "error": None,
    }


def run_prediction(
    smiles: str,
    routing_output: Dict[str, Any],
    model_features: Optional[Dict[str, Any]] = None,
    wrappers: Optional[Dict[str, Any]] = None,
    preprocessors: Optional[Dict[str, Any]] = None,
    model_builders: Optional[Dict[str, Any]] = None,
    postprocessors: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    target_wavelengths: Optional[np.ndarray] = None,
    use_counter_prop: bool = False,
    counter_prop_predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    smiles = str(smiles).strip()

    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    if target_wavelengths is None:
        target_wavelengths = TARGET_WAVELENGTHS

    if wrappers is None:
        wrappers = build_model_wrappers(
            model_builders=model_builders,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            device=device,
        )

    selected_models = routing_output.get("selected_models", [])

    if not selected_models:
        primary_model = routing_output.get("primary_model")
        selected_models = [primary_model] if primary_model else ["MPNN"]

    predictions = run_selected_models(
        smiles=smiles,
        selected_models=selected_models,
        wrappers=wrappers,
        model_features=model_features,
    )

    successful_predictions = get_successful_predictions(predictions)
    failed_predictions = get_failed_predictions(predictions)

    ensemble_prediction = average_predictions(
        predictions=predictions,
        target_wavelengths=target_wavelengths,
    )

    counter_prop_prediction = None

    if use_counter_prop:
        if counter_prop_predictor is None:
            counter_prop_prediction = {
                "model_name": "CounterPropagation",
                "status": "skipped",
                "error": (
                    "Counter-propagation was requested, but no predictor "
                    "was supplied. The selected model prediction was retained."
                ),
            }
        else:
            primary_model = routing_output.get("primary_model")
            if primary_model != "MPNN":
                counter_prop_prediction = {
                    "model_name": "CounterPropagation",
                    "status": "skipped",
                    "error": (
                        "The current counter-propagation model was trained "
                        "from MPNN predictions and is not applied to Transformer output."
                    ),
                }
            else:
                try:
                    counter_prop_prediction = counter_prop_predictor.predict(
                        model_features=model_features,
                        predictions=predictions,
                    )
                except Exception as error:
                    counter_prop_prediction = {
                        "model_name": "CounterPropagation",
                        "status": "failed",
                        "error": str(error),
                    }

    final_prediction = ensemble_prediction

    if (
        isinstance(counter_prop_prediction, dict)
        and counter_prop_prediction.get("status") == "success"
    ):
        final_prediction = counter_prop_prediction

    return {
        "smiles": smiles,
        "routing_output": routing_output,
        "model_predictions": predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "ensemble_prediction": ensemble_prediction,
        "counter_prop_prediction": counter_prop_prediction,
        "final_prediction": final_prediction,
    }


def save_prediction(
    prediction_output: Dict[str, Any],
    output_file: str | Path = "latest_prediction.json",
) -> Path:
    path = Path(output_file)

    if not path.is_absolute():
        path = PROJECT_FOLDER / path

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            prediction_output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("Prediction saved:", path)
    return path


def main() -> None:
    smiles = input("Enter a SMILES string: ").strip()

    model_features_file = PROJECT_FOLDER / "latest_model_features.json"
    routing_file = PROJECT_FOLDER / "latest_routing_output.json"

    if not model_features_file.exists():
        raise FileNotFoundError(
            "Run model_features.py first. "
            f"File not found: {model_features_file}"
        )

    if not routing_file.exists():
        raise FileNotFoundError(
            "Run router.py first. "
            f"File not found: {routing_file}"
        )

    with open(model_features_file, "r", encoding="utf-8") as file:
        model_features = json.load(file)

    feature_smiles = str(model_features.get("smiles", "")).strip()
    if feature_smiles and feature_smiles != smiles:
        print(
            "Warning: latest_model_features.json was created for "
            f"{feature_smiles}, but prediction input is {smiles}."
        )

    with open(routing_file, "r", encoding="utf-8") as file:
        routing_output = json.load(file)

    prediction_output = run_prediction(
        smiles=smiles,
        routing_output=routing_output,
        model_features=model_features,
        target_wavelengths=TARGET_WAVELENGTHS,
        use_counter_prop=False,
    )

    save_prediction(prediction_output)

    summary = prediction_output["final_prediction"]

    print()
    print("UV-Vis Prediction")
    print("-----------------")
    print("Selected models:", routing_output.get("selected_models"))
    print("Final model:", summary.get("model_name"))
    print("Status:", summary.get("status"))
    print("Lambda max:", summary.get("lambda_max"))
    print("Confidence:", summary.get("confidence"))
    print("Uncertainty:", summary.get("uncertainty"))
    print("Error:", summary.get("error"))


if __name__ == "__main__":
    main()


# from pathlib import Path
# import json
# import sys

# import numpy as np
# import torch
# from torch import nn

# project_folder = Path(
#     __file__
# ).resolve().parent

# uv_vis_folder = project_folder.parent

# integration_folder = (
#     uv_vis_folder
#     / "PNNL repo integration"
# )

# if str(integration_folder) not in sys.path:
#     sys.path.insert(
#         0,
#         str(integration_folder)
#     )

# from model_wrappers_pnnl import build_pnnl_wrappers



# model_features_file = (
#     project_folder
#     / "latest_model_features.json"
# )

# routing_file = (
#     project_folder
#     / "latest_routing_output.json"
# )

# prediction_output_file = (
#     project_folder
#     / "latest_prediction.json"
# )

# counter_prop_folder = (
#     uv_vis_folder
#     / "Counter-Prop"
# )

# models_folder = (
#     counter_prop_folder
#     / "models"
# )

# counter_prop_model_file = (
#     models_folder
#     / "counter_prop.pth"
# )

# input_scaler_file = (
#     models_folder
#     / "counter_prop_input_scaler.json"
# )

# target_scaler_file = (
#     models_folder
#     / "counter_prop_target_scaler.json"
# )


# wavelength_min = 220
# wavelength_max = 400

# target_wavelengths = np.arange(
#     wavelength_min,
#     wavelength_max + 1,
#     dtype=np.float32
# )


# hidden_size_1 = 512
# hidden_size_2 = 256
# hidden_size_3 = 181

# dropout_probability = 0.15


# device = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )


# class CounterPropagationNetwork(
#     nn.Module
# ):
#     def __init__(
#         self,
#         input_size,
#         output_size
#     ):
#         super().__init__()

#         self.network = nn.Sequential(
#             nn.Linear(
#                 input_size,
#                 hidden_size_1
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(
#                 hidden_size_1
#             ),
#             nn.Dropout(
#                 dropout_probability
#             ),

#             nn.Linear(
#                 hidden_size_1,
#                 hidden_size_2
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(
#                 hidden_size_2
#             ),
#             nn.Dropout(
#                 dropout_probability
#             ),

#             nn.Linear(
#                 hidden_size_2,
#                 hidden_size_3
#             ),
#             nn.ReLU(),

#             nn.Linear(
#                 hidden_size_3,
#                 output_size
#             )
#         )

#     def forward(
#         self,
#         inputs
#     ):
#         return self.network(
#             inputs
#         )


# def load_scaler(
#         scaler_file
# ):
#     if not scaler_file.exists():
#         raise FileNotFoundError(
#             f"Scaler file not found: {scaler_file}"
#         )

#     with open(
#         scaler_file,
#         "r",
#         encoding="utf-8"
#     ) as file:
#         scaler_data = json.load(
#             file
#         )

#     mean = np.asarray(
#         scaler_data[
#             "mean"
#         ],
#         dtype=np.float32
#     )

#     scale = np.asarray(
#         scaler_data[
#             "scale"
#         ],
#         dtype=np.float32
#     )

#     scale[
#         scale == 0
#     ] = 1.0

#     return {
#         "mean": mean,
#         "scale": scale
#     }


# def transform_values(
#         values,
#         scaler
# ):
#     values = np.asarray(
#         values,
#         dtype=np.float32
#     )

#     return (
#         values
#         - scaler[
#             "mean"
#         ]
#     ) / scaler[
#         "scale"
#     ]


# def inverse_transform_values(
#         values,
#         scaler
# ):
#     values = np.asarray(
#         values,
#         dtype=np.float32
#     )

#     return (
#         values
#         * scaler[
#             "scale"
#         ]
#         + scaler[
#             "mean"
#         ]
#     )


# class CounterPropPredictor:
#     def __init__(
#         self
#     ):
#         if not counter_prop_model_file.exists():
#             raise FileNotFoundError(
#                 "Counter-propagation model not found: "
#                 f"{counter_prop_model_file}"
#             )

#         self.input_scaler = load_scaler(
#             input_scaler_file
#         )

#         self.target_scaler = load_scaler(
#             target_scaler_file
#         )

#         self.checkpoint = torch.load(
#             counter_prop_model_file,
#             map_location=device,
#             weights_only=False
#         )

#         input_size = int(
#             self.checkpoint.get(
#                 "input_size",
#                 len(
#                     target_wavelengths
#                 )
#             )
#         )

#         output_size = int(
#             self.checkpoint.get(
#                 "output_size",
#                 len(
#                     target_wavelengths
#                 )
#             )
#         )

#         self.model = CounterPropagationNetwork(
#             input_size=input_size,
#             output_size=output_size
#         ).to(
#             device
#         )

#         self.model.load_state_dict(
#             self.checkpoint[
#                 "model_state_dict"
#             ]
#         )

#         self.model.eval()

#     def find_mpnn_prediction(
#         self,
#         predictions
#     ):
#         for prediction in predictions:
#             model_name = str(
#                 prediction.get(
#                     "model_name",
#                     ""
#                 )
#             ).strip().lower()

#             if (
#                 model_name == "mpnn"
#                 and prediction.get(
#                     "status"
#                 ) == "success"
#             ):
#                 return prediction

#         return None

#     def predict(
#         self,
#         model_features=None,
#         predictions=None
#     ):
#         try:
#             if predictions is None:
#                 predictions = []

#             mpnn_prediction = self.find_mpnn_prediction(
#                 predictions
#             )

#             if mpnn_prediction is None:
#                 raise ValueError(
#                     "A successful MPNN prediction is required "
#                     "for the current counter-propagation model."
#                 )

#             spectrum = np.asarray(
#                 mpnn_prediction.get(
#                     "spectrum"
#                 ),
#                 dtype=np.float32
#             ).reshape(
#                 -1
#             )

#             wavelengths = mpnn_prediction.get(
#                 "wavelengths"
#             )

#             if wavelengths is not None:
#                 wavelengths = np.asarray(
#                     wavelengths,
#                     dtype=np.float32
#                 ).reshape(
#                     -1
#                 )

#             if (
#                 wavelengths is not None
#                 and (
#                     len(
#                         wavelengths
#                     ) != len(
#                         target_wavelengths
#                     )
#                     or not np.allclose(
#                         wavelengths,
#                         target_wavelengths
#                     )
#                 )
#             ):
#                 spectrum = np.interp(
#                     target_wavelengths,
#                     wavelengths,
#                     spectrum
#                 ).astype(
#                     np.float32
#                 )

#             if len(
#                 spectrum
#             ) != len(
#                 target_wavelengths
#             ):
#                 raise ValueError(
#                     "The MPNN spectrum must contain exactly "
#                     f"{len(target_wavelengths)} values."
#                 )

#             scaled_input = transform_values(
#                 spectrum,
#                 self.input_scaler
#             ).reshape(
#                 1,
#                 -1
#             )

#             input_tensor = torch.tensor(
#                 scaled_input,
#                 dtype=torch.float32,
#                 device=device
#             )

#             with torch.no_grad():
#                 scaled_prediction = self.model(
#                     input_tensor
#                 ).cpu().numpy()[
#                     0
#                 ]

#             predicted_spectrum = inverse_transform_values(
#                 scaled_prediction,
#                 self.target_scaler
#             )

#             predicted_spectrum = np.clip(
#                 predicted_spectrum,
#                 0.0,
#                 None
#             )

#             maximum = float(
#                 np.max(
#                     predicted_spectrum
#                 )
#             )

#             if maximum > 0:
#                 predicted_spectrum = (
#                     predicted_spectrum
#                     / maximum
#                 )

#             lambda_max = float(
#                 target_wavelengths[
#                     int(
#                         np.argmax(
#                             predicted_spectrum
#                         )
#                     )
#                 ]
#             )

#             return {
#                 "model_name": "CounterPropagation",
#                 "source_models": [
#                     "MPNN"
#                 ],
#                 "wavelengths": target_wavelengths.tolist(),
#                 "spectrum": predicted_spectrum.tolist(),
#                 "lambda_max": lambda_max,
#                 "confidence": None,
#                 "uncertainty": None,
#                 "status": "success",
#                 "error": None,
#                 "checkpoint_epoch": self.checkpoint.get(
#                     "epoch"
#                 )
#             }

#         except Exception as error:
#             return {
#                 "model_name": "CounterPropagation",
#                 "source_models": [
#                     "MPNN"
#                 ],
#                 "wavelengths": None,
#                 "spectrum": None,
#                 "lambda_max": None,
#                 "confidence": None,
#                 "uncertainty": None,
#                 "status": "failed",
#                 "error": str(
#                     error
#                 )
#             }


# def normalize_prediction_spectrum(
#         prediction,
#         target_wavelengths=None
# ):
#     spectrum = prediction.get(
#         "spectrum"
#     )

#     wavelengths = prediction.get(
#         "wavelengths"
#     )

#     if spectrum is None:
#         return None, None

#     spectrum = np.asarray(
#         spectrum,
#         dtype=np.float32
#     ).reshape(
#         -1
#     )

#     if wavelengths is not None:
#         wavelengths = np.asarray(
#             wavelengths,
#             dtype=np.float32
#         ).reshape(
#             -1
#         )

#     if target_wavelengths is not None:
#         target_wavelengths = np.asarray(
#             target_wavelengths,
#             dtype=np.float32
#         ).reshape(
#             -1
#         )

#         if wavelengths is None:
#             if len(
#                 spectrum
#             ) != len(
#                 target_wavelengths
#             ):
#                 raise ValueError(
#                     "The spectrum length does not match "
#                     "the target wavelength grid."
#                 )

#             wavelengths = target_wavelengths

#         elif (
#             len(
#                 wavelengths
#             ) != len(
#                 target_wavelengths
#             )
#             or not np.allclose(
#                 wavelengths,
#                 target_wavelengths
#             )
#         ):
#             spectrum = np.interp(
#                 target_wavelengths,
#                 wavelengths,
#                 spectrum
#             ).astype(
#                 np.float32
#             )

#             wavelengths = target_wavelengths

#     maximum = float(
#         np.max(
#             np.abs(
#                 spectrum
#             )
#         )
#     )

#     if maximum > 0:
#         spectrum = (
#             spectrum
#             / maximum
#         )

#     return wavelengths, spectrum


# def get_successful_predictions(
#         predictions
# ):
#     return [
#         prediction
#         for prediction in predictions
#         if prediction.get(
#             "status"
#         ) == "success"
#     ]


# def get_failed_predictions(
#         predictions
# ):
#     return [
#         prediction
#         for prediction in predictions
#         if prediction.get(
#             "status"
#         ) != "success"
#     ]


# def run_selected_models(
#         smiles,
#         selected_models,
#         wrappers,
#         model_features=None
# ):
#     predictions = []

#     for model_name in selected_models:
#         wrapper = wrappers.get(
#             model_name
#         )

#         if wrapper is None:
#             predictions.append(
#                 {
#                     "model_name": model_name,
#                     "wavelengths": None,
#                     "spectrum": None,
#                     "lambda_max": None,
#                     "confidence": None,
#                     "uncertainty": None,
#                     "status": "failed",
#                     "error": (
#                         "No PNNL model wrapper was found for "
#                         f"{model_name}."
#                     )
#                 }
#             )

#             continue

#         try:
#             prediction = wrapper.predict(
#                 smiles=smiles,
#                 model_features=model_features
#             )

#         except Exception as error:
#             prediction = {
#                 "model_name": model_name,
#                 "wavelengths": None,
#                 "spectrum": None,
#                 "lambda_max": None,
#                 "confidence": None,
#                 "uncertainty": None,
#                 "status": "failed",
#                 "error": str(
#                     error
#                 )
#             }

#         predictions.append(
#             prediction
#         )

#     return predictions


# def average_predictions(
#         predictions,
#         target_wavelengths=None
# ):
#     successful_predictions = get_successful_predictions(
#         predictions
#     )

#     if not successful_predictions:
#         return {
#             "model_name": "Ensemble",
#             "wavelengths": None,
#             "spectrum": None,
#             "lambda_max": None,
#             "confidence": None,
#             "uncertainty": None,
#             "status": "failed",
#             "error": (
#                 "No successful model predictions "
#                 "were available."
#             )
#         }

#     aligned_spectra = []
#     prediction_weights = []
#     model_names = []

#     wavelengths = None

#     for prediction in successful_predictions:
#         (
#             aligned_wavelengths,
#             aligned_spectrum
#         ) = normalize_prediction_spectrum(
#             prediction=prediction,
#             target_wavelengths=target_wavelengths
#         )

#         if aligned_spectrum is None:
#             continue

#         if wavelengths is None:
#             wavelengths = aligned_wavelengths

#         confidence = prediction.get(
#             "confidence"
#         )

#         try:
#             confidence = (
#                 float(
#                     confidence
#                 )
#                 if confidence is not None
#                 else 1.0
#             )

#         except (
#             TypeError,
#             ValueError
#         ):
#             confidence = 1.0

#         confidence = max(
#             confidence,
#             0.01
#         )

#         aligned_spectra.append(
#             aligned_spectrum
#         )

#         prediction_weights.append(
#             confidence
#         )

#         model_names.append(
#             prediction.get(
#                 "model_name",
#                 "Unknown"
#             )
#         )

#     if not aligned_spectra:
#         return {
#             "model_name": "Ensemble",
#             "wavelengths": None,
#             "spectrum": None,
#             "lambda_max": None,
#             "confidence": None,
#             "uncertainty": None,
#             "status": "failed",
#             "error": (
#                 "No valid spectra were available "
#                 "for the ensemble."
#             )
#         }

#     aligned_spectra = np.asarray(
#         aligned_spectra,
#         dtype=np.float32
#     )

#     prediction_weights = np.asarray(
#         prediction_weights,
#         dtype=np.float32
#     )

#     prediction_weights = (
#         prediction_weights
#         / np.sum(
#             prediction_weights
#         )
#     )

#     ensemble_spectrum = np.average(
#         aligned_spectra,
#         axis=0,
#         weights=prediction_weights
#     )

#     if len(
#         aligned_spectra
#     ) == 1:
#         ensemble_uncertainty = 0.0
#     else:
#         ensemble_uncertainty = float(
#             np.mean(
#                 np.std(
#                     aligned_spectra,
#                     axis=0
#                 )
#             )
#         )

#     lambda_max = None

#     if (
#         wavelengths is not None
#         and len(
#             wavelengths
#         ) == len(
#             ensemble_spectrum
#         )
#     ):
#         lambda_max = float(
#             wavelengths[
#                 int(
#                     np.argmax(
#                         ensemble_spectrum
#                     )
#                 )
#             ]
#         )

#     ensemble_confidence = float(
#         np.clip(
#             1.0
#             / (
#                 1.0
#                 + ensemble_uncertainty
#             ),
#             0.0,
#             1.0
#         )
#     )

#     return {
#         "model_name": "Ensemble",
#         "source_models": model_names,
#         "wavelengths": (
#             None
#             if wavelengths is None
#             else wavelengths.tolist()
#         ),
#         "spectrum": ensemble_spectrum.tolist(),
#         "lambda_max": lambda_max,
#         "confidence": ensemble_confidence,
#         "uncertainty": ensemble_uncertainty,
#         "status": "success",
#         "error": None
#     }


# def run_prediction(
#         smiles,
#         routing_output,
#         model_features=None,
#         wrappers=None,
#         target_wavelengths=None,
#         use_counter_prop=True,
#         counter_prop_predictor=None
# ):
#     if wrappers is None:
#         wrappers = build_pnnl_wrappers()

#     selected_models = routing_output.get(
#         "selected_models",
#         []
#     )

#     if not selected_models:
#         selected_models = [
#             "MPNN"
#         ]

#     selected_models = [
#         str(
#             model_name
#         ).strip()
#         for model_name in selected_models
#         if str(
#             model_name
#         ).strip()
#     ]

#     if "MPNN" not in selected_models:
#         selected_models.append(
#             "MPNN"
#         )

#     selected_models = list(
#         dict.fromkeys(
#             selected_models
#         )
#     )

#     predictions = run_selected_models(
#         smiles=smiles,
#         selected_models=selected_models,
#         wrappers=wrappers,
#         model_features=model_features
#     )

#     successful_predictions = get_successful_predictions(
#         predictions
#     )

#     failed_predictions = get_failed_predictions(
#         predictions
#     )

#     ensemble_prediction = average_predictions(
#         predictions=predictions,
#         target_wavelengths=target_wavelengths
#     )

#     counter_prop_prediction = None

#     if use_counter_prop:
#         if counter_prop_predictor is None:
#             try:
#                 counter_prop_predictor = CounterPropPredictor()

#             except Exception as error:
#                 counter_prop_prediction = {
#                     "model_name": "CounterPropagation",
#                     "status": "failed",
#                     "error": str(
#                         error
#                     )
#                 }

#         if (
#             counter_prop_predictor is not None
#             and counter_prop_prediction is None
#         ):
#             counter_prop_prediction = (
#                 counter_prop_predictor.predict(
#                     model_features=model_features,
#                     predictions=predictions
#                 )
#             )

#     final_prediction = ensemble_prediction

#     if (
#         counter_prop_prediction is not None
#         and counter_prop_prediction.get(
#             "status"
#         ) == "success"
#     ):
#         final_prediction = counter_prop_prediction

#     return {
#         "smiles": smiles,
#         "routing_output": routing_output,
#         "model_predictions": predictions,
#         "successful_predictions": successful_predictions,
#         "failed_predictions": failed_predictions,
#         "ensemble_prediction": ensemble_prediction,
#         "counter_prop_prediction": counter_prop_prediction,
#         "final_prediction": final_prediction
#     }


# def save_prediction(
#         prediction_output,
#         output_file=prediction_output_file
# ):
#     output_file = Path(
#         output_file
#     )

#     with open(
#         output_file,
#         "w",
#         encoding="utf-8"
#     ) as file:
#         json.dump(
#             prediction_output,
#             file,
#             indent=4,
#             ensure_ascii=False
#         )

#     print(
#         "Prediction saved:",
#         output_file
#     )


# def main():
#     smiles = input(
#         "Enter a SMILES string: "
#     ).strip()

#     if not model_features_file.exists():
#         raise FileNotFoundError(
#             f"File not found: {model_features_file}"
#         )

#     if not routing_file.exists():
#         raise FileNotFoundError(
#             f"File not found: {routing_file}"
#         )

#     with open(
#         model_features_file,
#         "r",
#         encoding="utf-8"
#     ) as file:
#         model_features = json.load(
#             file
#         )

#     with open(
#         routing_file,
#         "r",
#         encoding="utf-8"
#     ) as file:
#         routing_output = json.load(
#             file
#         )

#     prediction_output = run_prediction(
#         smiles=smiles,
#         routing_output=routing_output,
#         model_features=model_features,
#         target_wavelengths=target_wavelengths,
#         use_counter_prop=True
#     )

#     save_prediction(
#         prediction_output
#     )

#     print(
#         json.dumps(
#             prediction_output,
#             indent=4
#         )
#     )


# if __name__ == "__main__":
#     main()