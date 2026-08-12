from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_FOLDER = Path(__file__).resolve().parent
MODELS = ["MPNN", "Transformer"]

UNAVAILABLE_MODELS = {
    "SchNet": "Disabled for the prototype because no usable checkpoint is available.",
    "DTNN": "Disabled for the prototype because trained weights and normalization files are missing.",
}


def normalize_model_name(model_name: Any) -> str:
    normalized = str(model_name).strip().lower()

    model_map = {
        "mpnn": "MPNN",
        "message passing neural network": "MPNN",
        "transformer": "Transformer",
        "smiles transformer": "Transformer",
        "soltrannet": "Transformer",
        "molecule attention transformer": "Transformer",
    }

    return model_map.get(normalized, str(model_name).strip())


def clean_model_list(model_names: Any) -> List[str]:
    if model_names is None:
        return []

    if isinstance(model_names, str):
        model_names = [model_names]

    cleaned: List[str] = []

    for model_name in model_names:
        normalized = normalize_model_name(model_name)

        if normalized in MODELS and normalized not in cleaned:
            cleaned.append(normalized)

    return cleaned


def get_nested_value(
    dictionary: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    value: Any = dictionary

    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def _flatten_strings(value: Any) -> List[str]:
    output: List[str] = []

    if value is None:
        return output

    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            output.append(stripped)
        return output

    if isinstance(value, (list, tuple, set)):
        for item in value:
            output.extend(_flatten_strings(item))

    return output


def score_models(
    model_features: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores = {
        "MPNN": 2.0,
        "Transformer": 1.5,
    }

    reasons = {
        "MPNN": ["General molecular graph model."],
        "Transformer": ["Learns global patterns from the molecular representation."],
    }

    molecular = get_nested_value(model_features, "molecular_features", {}) or {}
    routing_flags = get_nested_value(model_features, "routing_flags", {}) or {}

    num_atoms = float(molecular.get("num_atoms", 0) or 0)
    num_heavy_atoms = float(molecular.get("num_heavy_atoms", 0) or 0)
    num_rings = float(molecular.get("num_rings", 0) or 0)
    num_aromatic_rings = float(molecular.get("num_aromatic_rings", 0) or 0)
    num_conjugated_bonds = float(molecular.get("num_conjugated_bonds", 0) or 0)
    molecular_weight = float(molecular.get("molecular_weight", 0) or 0)

    canonical_smiles = str(
        molecular.get(
            "canonical_smiles",
            model_features.get("smiles", ""),
        )
        or ""
    )

    smiles_length = len(canonical_smiles)

    aromatic_system = bool(routing_flags.get("aromatic_system", False))
    conjugated_system = bool(routing_flags.get("conjugated_system", False))
    flexible_molecule = bool(routing_flags.get("flexible_molecule", False))
    low_retrieval_support = bool(routing_flags.get("low_retrieval_support", False))

    complexity_level = str(
        get_nested_value(
            model_features,
            "complexity_features.complexity_level",
            "unknown",
        )
    ).strip().lower()

    recommended_raw = get_nested_value(
        model_features,
        "llm_features.recommended_models",
        [],
    )
    recommended_models = clean_model_list(_flatten_strings(recommended_raw))

    # MPNN: local graph neighborhoods and smaller/medium molecules.
    if num_heavy_atoms <= 35:
        scores["MPNN"] += 1.0
        reasons["MPNN"].append("Small or medium heavy-atom count favors graph message passing.")

    if num_atoms <= 20:
        scores["MPNN"] += 0.75
        reasons["MPNN"].append("The molecule has a compact molecular graph.")

    if num_aromatic_rings <= 1 and num_conjugated_bonds <= 4:
        scores["MPNN"] += 0.75
        reasons["MPNN"].append("Local chemical environments are likely to dominate.")

    if low_retrieval_support:
        scores["MPNN"] += 0.5
        reasons["MPNN"].append("Low retrieval support favors a learned graph representation.")

    # Transformer: larger/global/repeated/aromatic/conjugated structural patterns.
    if smiles_length >= 45:
        scores["Transformer"] += 1.0
        reasons["Transformer"].append("The longer molecular representation contains global patterns.")

    if num_aromatic_rings >= 2:
        scores["Transformer"] += 1.25
        reasons["Transformer"].append("Multiple aromatic rings favor global pattern modeling.")

    if num_conjugated_bonds >= 6 or conjugated_system:
        scores["Transformer"] += 1.0
        reasons["Transformer"].append("Extended conjugation can depend on long-range structure.")

    if num_rings >= 3:
        scores["Transformer"] += 0.75
        reasons["Transformer"].append("A multi-ring system benefits from whole-molecule context.")

    if molecular_weight >= 300:
        scores["Transformer"] += 0.5
        reasons["Transformer"].append("Higher molecular weight suggests greater structural complexity.")

    if flexible_molecule and num_heavy_atoms >= 20:
        scores["Transformer"] += 0.5
        reasons["Transformer"].append("A larger flexible structure contains nonlocal relationships.")

    if aromatic_system:
        scores["Transformer"] += 0.5
        reasons["Transformer"].append("Aromatic notation provides informative global patterns.")

    if complexity_level in {"high", "very high"}:
        scores["Transformer"] += 1.0
        reasons["Transformer"].append("High molecular complexity favors global representation learning.")

    for model_name in recommended_models:
        scores[model_name] += 0.75
        reasons[model_name].append("Recommended by the LLM analysis.")

    return scores, reasons


def calculate_routing_confidence(
    selected_models: List[str],
    scores: Dict[str, float],
) -> float:
    if not selected_models:
        return 0.0

    ranked_scores = sorted(
        (float(value) for value in scores.values()),
        reverse=True,
    )

    if len(ranked_scores) < 2:
        return 1.0

    first_score = ranked_scores[0]
    second_score = ranked_scores[1]
    denominator = max(abs(first_score), 1.0)

    return float(
        max(
            0.0,
            min(
                1.0,
                (first_score - second_score) / denominator,
            ),
        )
    )


def route_models(
    model_features: Dict[str, Any],
    max_models: int = 1,
    minimum_score: Optional[float] = None,
    available_models: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if not isinstance(model_features, dict):
        raise TypeError("model_features must be a dictionary.")

    if not isinstance(max_models, int) or max_models <= 0:
        raise ValueError("max_models must be a positive integer.")

    available = clean_model_list(
        list(available_models) if available_models is not None else MODELS
    )

    if not available:
        raise ValueError("No available models were provided.")

    scores, reasons = score_models(model_features)

    ranked_models = sorted(
        available,
        key=lambda name: scores.get(name, float("-inf")),
        reverse=True,
    )

    if minimum_score is not None:
        ranked_models = [
            name
            for name in ranked_models
            if scores.get(name, 0.0) >= minimum_score
        ]

    selected_models = ranked_models[:max_models]

    if not selected_models:
        selected_models = ["MPNN"] if "MPNN" in available else [available[0]]

    return {
        "primary_model": selected_models[0],
        "selected_models": selected_models,
        "model_scores": {
            name: float(scores.get(name, 0.0))
            for name in available
        },
        "model_reasons": {
            name: reasons.get(name, [])
            for name in available
        },
        "routing_confidence": calculate_routing_confidence(
            selected_models,
            {name: scores[name] for name in available},
        ),
        "available_models": available,
        "unavailable_models": UNAVAILABLE_MODELS,
    }


def main() -> None:
    features_file = PROJECT_FOLDER / "latest_model_features.json"
    output_file = PROJECT_FOLDER / "latest_routing_output.json"

    if not features_file.exists():
        raise FileNotFoundError(f"File not found: {features_file}")

    with open(features_file, "r", encoding="utf-8") as file:
        model_features = json.load(file)

    routing_output = route_models(
        model_features=model_features,
        max_models=1,
        available_models=MODELS,
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(routing_output, file, indent=4, ensure_ascii=False)

    print(json.dumps(routing_output, indent=4, ensure_ascii=False))
    print("Routing output saved:", output_file)


if __name__ == "__main__":
    main()



# from pathlib import Path
# import json


# router_folder = Path(
#     __file__
# ).resolve().parent

# model_features_file = (
#     router_folder
#     / "latest_model_features.json"
# )

# routing_output_file = (
#     router_folder
#     / "latest_routing_output.json"
# )


# ALL_MODELS = [
#     "DTNN",
#     "MPNN",
#     "SchNet",
#     "Transformer"
# ]

# # Only models that currently work should be listed here.
# # Add SchNet and Transformer after their adapters are working.
# AVAILABLE_MODELS = [
#     "MPNN"
#     "Transformer"
# ]


# def normalize_model_name(model_name):
#     normalized_name = str(model_name).strip().lower()

#     model_map = {
#         "dtnn": "DTNN",
#         "deep tensor neural network": "DTNN",
#         "mpnn": "MPNN",
#         "message passing neural network": "MPNN",
#         "schnet": "SchNet",
#         "sch net": "SchNet",
#         "transformer": "Transformer",
#         "smiles transformer": "Transformer"
#     }

#     return model_map.get(
#         normalized_name,
#         str(model_name).strip()
#     )


# def flatten_values(values):
#     if values is None:
#         return []

#     if isinstance(values, (list, tuple, set)):
#         flattened = []

#         for value in values:
#             flattened.extend(
#                 flatten_values(value)
#             )

#         return flattened

#     return [values]


# def clean_model_list(model_names):
#     cleaned_models = []

#     for model_name in flatten_values(model_names):
#         normalized_name = normalize_model_name(model_name)

#         if (
#             normalized_name in ALL_MODELS
#             and normalized_name not in cleaned_models
#         ):
#             cleaned_models.append(normalized_name)

#     return cleaned_models


# def get_nested_value(dictionary, path, default=None):
#     value = dictionary

#     for key in path.split("."):
#         if not isinstance(value, dict) or key not in value:
#             return default

#         value = value[key]

#     return value


# def safe_float(value, default=0.0):
#     try:
#         return float(value)

#     except (TypeError, ValueError):
#         return float(default)


# def score_models(model_features):
#     scores = {
#         "DTNN": 0.0,
#         "MPNN": 0.0,
#         "SchNet": 0.0,
#         "Transformer": 0.0
#     }

#     reasons = {
#         "DTNN": [],
#         "MPNN": [],
#         "SchNet": [],
#         "Transformer": []
#     }

#     num_atoms = safe_float(
#         get_nested_value(
#             model_features,
#             "molecular_features.num_atoms",
#             0
#         )
#     )

#     num_heavy_atoms = safe_float(
#         get_nested_value(
#             model_features,
#             "molecular_features.num_heavy_atoms",
#             0
#         )
#     )

#     num_rings = safe_float(
#         get_nested_value(
#             model_features,
#             "molecular_features.num_rings",
#             0
#         )
#     )

#     num_aromatic_rings = safe_float(
#         get_nested_value(
#             model_features,
#             "molecular_features.num_aromatic_rings",
#             0
#         )
#     )

#     molecular_weight = safe_float(
#         get_nested_value(
#             model_features,
#             "molecular_features.molecular_weight",
#             0
#         )
#     )

#     aromatic_system = bool(
#         get_nested_value(
#             model_features,
#             "routing_flags.aromatic_system",
#             False
#         )
#     )

#     conjugated_system = bool(
#         get_nested_value(
#             model_features,
#             "routing_flags.conjugated_system",
#             False
#         )
#     )

#     flexible_molecule = bool(
#         get_nested_value(
#             model_features,
#             "routing_flags.flexible_molecule",
#             False
#         )
#     )

#     contains_uncommon_elements = bool(
#         get_nested_value(
#             model_features,
#             "routing_flags.contains_uncommon_elements",
#             False
#         )
#     )

#     low_retrieval_support = bool(
#         get_nested_value(
#             model_features,
#             "routing_flags.low_retrieval_support",
#             False
#         )
#     )

#     complexity_level = str(
#         get_nested_value(
#             model_features,
#             "complexity_features.complexity_level",
#             "unknown"
#         )
#     ).strip().lower()

#     llm_recommended_models = clean_model_list(
#         get_nested_value(
#             model_features,
#             "llm_features.recommended_models",
#             []
#         )
#     )

#     scores["MPNN"] += 2.0
#     reasons["MPNN"].append(
#         "General molecular graph model."
#     )

#     scores["Transformer"] += 1.5
#     reasons["Transformer"].append(
#         "Can operate from a SMILES representation."
#     )

#     if num_heavy_atoms <= 35:
#         scores["DTNN"] += 1.5
#         reasons["DTNN"].append(
#             "The molecule is relatively small."
#         )

#     if num_heavy_atoms >= 20:
#         scores["MPNN"] += 1.0
#         reasons["MPNN"].append(
#             "The graph contains enough atoms for message passing."
#         )

#     if aromatic_system:
#         scores["MPNN"] += 1.5
#         scores["Transformer"] += 1.0

#         reasons["MPNN"].append(
#             "The molecule contains an aromatic system."
#         )

#         reasons["Transformer"].append(
#             "SMILES tokens encode aromatic patterns."
#         )

#     if conjugated_system:
#         scores["MPNN"] += 1.5
#         scores["SchNet"] += 1.0
#         scores["Transformer"] += 1.0

#         reasons["MPNN"].append(
#             "The molecule contains conjugated bonds."
#         )

#         reasons["SchNet"].append(
#             "Geometry-aware interactions may help with conjugation."
#         )

#         reasons["Transformer"].append(
#             "The SMILES sequence contains conjugation information."
#         )

#     if num_rings >= 2:
#         scores["SchNet"] += 1.5
#         scores["MPNN"] += 1.0

#         reasons["SchNet"].append(
#             "Multiple rings may benefit from three-dimensional modeling."
#         )

#         reasons["MPNN"].append(
#             "Multiple rings provide informative graph structure."
#         )

#     if num_aromatic_rings >= 2:
#         scores["Transformer"] += 0.5
#         reasons["Transformer"].append(
#             "Multiple aromatic rings produce informative SMILES patterns."
#         )

#     if flexible_molecule:
#         scores["SchNet"] += 1.0
#         scores["MPNN"] += 0.5

#         reasons["SchNet"].append(
#             "A flexible molecule may benefit from geometry-aware modeling."
#         )

#         reasons["MPNN"].append(
#             "Graph message passing can represent flexible connectivity."
#         )

#     if molecular_weight >= 300:
#         scores["MPNN"] += 0.5
#         scores["SchNet"] += 0.5

#         reasons["MPNN"].append(
#             "The larger molecular graph may benefit from message passing."
#         )

#         reasons["SchNet"].append(
#             "The larger molecule may benefit from geometry-aware interactions."
#         )

#     if num_atoms <= 15:
#         scores["DTNN"] += 1.0
#         reasons["DTNN"].append(
#             "The molecule has a small atom count."
#         )

#     if contains_uncommon_elements:
#         scores["MPNN"] += 0.5
#         scores["Transformer"] -= 0.5

#         reasons["MPNN"].append(
#             "Graph features can explicitly represent uncommon elements."
#         )

#     if low_retrieval_support:
#         scores["MPNN"] += 0.5
#         scores["SchNet"] += 0.5

#         reasons["MPNN"].append(
#             "Low retrieval support favors learned molecular structure."
#         )

#         reasons["SchNet"].append(
#             "Low retrieval support may favor learned geometry."
#         )

#     if complexity_level in ["high", "very high"]:
#         scores["MPNN"] += 1.0
#         scores["SchNet"] += 1.0

#         reasons["MPNN"].append(
#             "High molecular complexity favors a graph model."
#         )

#         reasons["SchNet"].append(
#             "High molecular complexity may benefit from geometry-aware modeling."
#         )

#     for model_name in llm_recommended_models:
#         scores[model_name] += 1.0
#         reasons[model_name].append(
#             "Recommended by the LLM analysis."
#         )

#     return scores, reasons


# def calculate_routing_confidence(ranked_models, scores):
#     if not ranked_models:
#         return 0.0

#     first_score = float(
#         scores.get(
#             ranked_models[0],
#             0.0
#         )
#     )

#     if len(ranked_models) == 1:
#         return 1.0

#     second_score = float(
#         scores.get(
#             ranked_models[1],
#             0.0
#         )
#     )

#     denominator = max(
#         abs(first_score),
#         1.0
#     )

#     confidence = (
#         first_score
#         - second_score
#     ) / denominator

#     return float(
#         max(
#             0.0,
#             min(
#                 1.0,
#                 confidence
#             )
#         )
#     )


# def route_models(
#         model_features,
#         max_models=1,
#         minimum_score=None,
#         available_models=None
# ):
#     if not isinstance(model_features, dict):
#         raise TypeError(
#             "model_features must be a dictionary."
#         )

#     if not isinstance(max_models, int):
#         raise TypeError(
#             "max_models must be an integer."
#         )

#     if max_models <= 0:
#         raise ValueError(
#             "max_models must be greater than zero."
#         )

#     if available_models is None:
#         available_models = AVAILABLE_MODELS

#     available_models = clean_model_list(
#         available_models
#     )

#     if not available_models:
#         raise ValueError(
#             "No available models were provided."
#         )

#     scores, reasons = score_models(
#         model_features
#     )

#     ranked_models = sorted(
#         available_models,
#         key=lambda model_name: scores.get(
#             model_name,
#             float("-inf")
#         ),
#         reverse=True
#     )

#     if minimum_score is not None:
#         ranked_models = [
#             model_name
#             for model_name in ranked_models
#             if scores.get(
#                 model_name,
#                 0.0
#             ) >= minimum_score
#         ]

#     if not ranked_models:
#         if "MPNN" in available_models:
#             ranked_models = ["MPNN"]
#         else:
#             ranked_models = [available_models[0]]

#     selected_models = ranked_models[:max_models]

#     return {
#         "primary_model": selected_models[0],
#         "selected_models": selected_models,
#         "ranked_models": ranked_models,
#         "available_models": available_models,
#         "model_scores": {
#             model_name: float(
#                 scores.get(
#                     model_name,
#                     0.0
#                 )
#             )
#             for model_name in ALL_MODELS
#         },
#         "model_reasons": {
#             model_name: reasons.get(
#                 model_name,
#                 []
#             )
#             for model_name in ALL_MODELS
#         },
#         "routing_confidence": calculate_routing_confidence(
#             ranked_models=ranked_models,
#             scores=scores
#         )
#     }


# def main():
#     if not model_features_file.exists():
#         raise FileNotFoundError(
#             f"File not found: {model_features_file}"
#         )

#     with open(
#         model_features_file,
#         "r",
#         encoding="utf-8"
#     ) as input_file:
#         model_features = json.load(
#             input_file
#         )

#     routing_output = route_models(
#         model_features=model_features,
#         max_models=1,
#         available_models=AVAILABLE_MODELS
#     )

#     with open(
#         routing_output_file,
#         "w",
#         encoding="utf-8"
#     ) as output_file:
#         json.dump(
#             routing_output,
#             output_file,
#             indent=4,
#             ensure_ascii=False
#         )

#     print(
#         json.dumps(
#             routing_output,
#             indent=4,
#             ensure_ascii=False
#         )
#     )

#     print()
#     print(
#         "Routing output saved:"
#     )
#     print(
#         routing_output_file
#     )


# if __name__ == "__main__":
#     main()