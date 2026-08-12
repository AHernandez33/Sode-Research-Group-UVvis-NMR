from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import importlib.util
import sys
import traceback


PROJECT_FOLDER = Path(__file__).resolve().parent
UVVIS_FOLDER = PROJECT_FOLDER.parent

MPNN_ADAPTER_CANDIDATES = [
    UVVIS_FOLDER / "PNNL repo integration" / "mpnn_adapter.py",
    PROJECT_FOLDER / "mpnn_adapter.py",
]

TRANSFORMER_ADAPTER_PATH = (
    UVVIS_FOLDER
    / "PNNL repo integration"
    / "transformer_adapter.py"
)

def _failed_prediction(model_name: str, error: Any) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "wavelengths": None,
        "spectrum": None,
        "lambda_max": None,
        "confidence": None,
        "uncertainty": None,
        "raw_output": None,
        "status": "failed",
        "error": str(error),
    }


def _normalize_prediction(model_name: str, result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return _failed_prediction(
            model_name,
            f"{model_name} adapter returned {type(result).__name__}, not a dictionary.",
        )

    output = {
        "model_name": result.get("model_name", model_name),
        "wavelengths": result.get("wavelengths"),
        "spectrum": result.get("spectrum"),
        "lambda_max": result.get("lambda_max"),
        "confidence": result.get("confidence"),
        "uncertainty": result.get("uncertainty"),
        "raw_output": result.get("raw_output", result.get("raw_spectrum")),
        "status": result.get("status", "failed"),
        "error": result.get("error"),
    }

    for key in (
        "checkpoint",
        "checkpoint_epoch",
        "validation_metrics",
        "model_parameters",
        "runtime_seconds",
        "device",
        "smiles",
        "command",
        "stdout",
        "stderr",
    ):
        if key in result:
            output[key] = result[key]

    if output["status"] == "success":
        if output["spectrum"] is None:
            return _failed_prediction(
                model_name,
                "Successful result did not contain a spectrum.",
            )

        try:
            spectrum_length = len(output["spectrum"])
        except TypeError:
            return _failed_prediction(model_name, "Spectrum is not a sequence.")

        if spectrum_length != 181:
            return _failed_prediction(
                model_name,
                f"Expected 181 spectrum values, found {spectrum_length}.",
            )

    return output


def _load_module(module_path: Path, module_name: str):
    module_path = Path(module_path).expanduser().resolve()

    if not module_path.exists():
        raise FileNotFoundError(f"Adapter file not found: {module_path}")

    module_folder = str(module_path.parent)

    # Dynamic imports do not automatically make sibling modules importable.
    # Adding the adapter directory fixes imports such as:
    # from original_config import command_timeout_seconds
    if module_folder not in sys.path:
        sys.path.insert(0, module_folder)

    specification = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load adapter module: {module_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@dataclass
class BaseModelWrapper:
    model_name: str

    def predict(
        self,
        smiles: str,
        model_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MPNNWrapper(BaseModelWrapper):
    adapter_path: Optional[Path] = None

    def __init__(self, adapter_path: Optional[str | Path] = None):
        super().__init__("MPNN")
        self.adapter_path = (
            Path(adapter_path).expanduser().resolve()
            if adapter_path
            else None
        )

    def _resolve_adapter(self) -> Path:
        if self.adapter_path is not None:
            if not self.adapter_path.exists():
                raise FileNotFoundError(
                    f"MPNN adapter not found: {self.adapter_path}"
                )
            return self.adapter_path

        for candidate in MPNN_ADAPTER_CANDIDATES:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "MPNN adapter was not found. Checked:\n"
            + "\n".join(str(path) for path in MPNN_ADAPTER_CANDIDATES)
        )

    def predict(
        self,
        smiles: str,
        model_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            adapter_path = self._resolve_adapter()
            module = _load_module(adapter_path, "_uvvis_mpnn_adapter")

            if hasattr(module, "predict_mpnn"):
                result = module.predict_mpnn(smiles)
            elif hasattr(module, "MPNNAdapter"):
                result = module.MPNNAdapter().predict(smiles)
            elif hasattr(module, "predict"):
                result = module.predict(smiles)
            else:
                raise AttributeError(
                    "MPNN adapter must expose predict_mpnn(smiles), "
                    "MPNNAdapter().predict(smiles), or predict(smiles)."
                )

            return _normalize_prediction(self.model_name, result)

        except Exception as error:
            output = _failed_prediction(self.model_name, error)
            output["traceback"] = traceback.format_exc()
            return output


@dataclass
class TransformerWrapper(BaseModelWrapper):
    adapter_path: Path = TRANSFORMER_ADAPTER_PATH

    def __init__(self, adapter_path: Optional[str | Path] = None):
        super().__init__("Transformer")
        self.adapter_path = (
            Path(adapter_path).expanduser().resolve()
            if adapter_path
            else TRANSFORMER_ADAPTER_PATH.resolve()
        )

    def predict(
        self,
        smiles: str,
        model_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            module = _load_module(
                self.adapter_path,
                "_uvvis_transformer_adapter",
            )

            if hasattr(module, "predict_transformer"):
                result = module.predict_transformer(smiles)
            elif hasattr(module, "TransformerAdapter"):
                result = module.TransformerAdapter().predict(smiles)
            else:
                raise AttributeError(
                    "transformer_adapter.py must expose "
                    "predict_transformer(smiles) or "
                    "TransformerAdapter().predict(smiles)."
                )

            return _normalize_prediction(self.model_name, result)

        except Exception as error:
            output = _failed_prediction(self.model_name, error)
            output["traceback"] = traceback.format_exc()
            return output


def build_model_wrappers(
    model_paths: Optional[Dict[str, str]] = None,
    model_builders: Optional[Dict[str, Any]] = None,
    preprocessors: Optional[Dict[str, Any]] = None,
    postprocessors: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    mpnn_adapter_path: Optional[str | Path] = None,
    transformer_adapter_path: Optional[str | Path] = None,
) -> Dict[str, BaseModelWrapper]:
    # Extra arguments remain accepted for compatibility with existing code.
    return {
        "MPNN": MPNNWrapper(adapter_path=mpnn_adapter_path),
        "Transformer": TransformerWrapper(
            adapter_path=transformer_adapter_path
        ),
    }


def run_selected_models(
    smiles: str,
    selected_models: Iterable[str],
    wrappers: Dict[str, BaseModelWrapper],
    model_features: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    for model_name in selected_models:
        wrapper = wrappers.get(model_name)

        if wrapper is None:
            predictions.append(
                _failed_prediction(
                    model_name,
                    f"No wrapper is registered for {model_name}.",
                )
            )
            continue

        try:
            prediction = wrapper.predict(
                smiles=smiles,
                model_features=model_features,
            )
        except Exception as error:
            prediction = _failed_prediction(model_name, error)
            prediction["traceback"] = traceback.format_exc()

        predictions.append(prediction)

    return predictions


def get_successful_predictions(
    predictions: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        prediction
        for prediction in predictions
        if prediction.get("status") == "success"
    ]


def get_failed_predictions(
    predictions: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        prediction
        for prediction in predictions
        if prediction.get("status") != "success"
    ]