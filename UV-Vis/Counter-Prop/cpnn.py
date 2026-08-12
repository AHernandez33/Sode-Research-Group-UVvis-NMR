# counter_prop.py

import os
import time
import json
import math
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StandardScaler:
    def __init__(
        self,
        epsilon=1e-8
    ):
        self.epsilon = epsilon
        self.mean = None
        self.standard_deviation = None
        self.fitted = False

    def fit(
        self,
        values
    ):
        values = np.asarray(
            values,
            dtype=np.float32
        )

        if values.ndim == 1:
            values = values.reshape(
                -1,
                1
            )

        self.mean = np.mean(
            values,
            axis=0
        ).astype(
            np.float32
        )

        self.standard_deviation = np.std(
            values,
            axis=0
        ).astype(
            np.float32
        )

        self.standard_deviation[
            self.standard_deviation < self.epsilon
        ] = 1.0

        self.fitted = True

        return self

    def transform(
        self,
        values
    ):
        if not self.fitted:
            raise RuntimeError(
                "The scaler must be fitted before transform()."
            )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        return (
            values - self.mean
        ) / self.standard_deviation

    def inverse_transform(
        self,
        values
    ):
        if not self.fitted:
            raise RuntimeError(
                "The scaler must be fitted before inverse_transform()."
            )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        return (
            values * self.standard_deviation
        ) + self.mean

    def state_dict(
        self
    ):
        return {
            "epsilon": self.epsilon,
            "mean": (
                None
                if self.mean is None
                else self.mean.tolist()
            ),
            "standard_deviation": (
                None
                if self.standard_deviation is None
                else self.standard_deviation.tolist()
            ),
            "fitted": self.fitted
        }

    def load_state_dict(
        self,
        state
    ):
        self.epsilon = float(
            state.get(
                "epsilon",
                1e-8
            )
        )

        mean = state.get(
            "mean"
        )

        standard_deviation = state.get(
            "standard_deviation"
        )

        self.mean = (
            None
            if mean is None
            else np.asarray(
                mean,
                dtype=np.float32
            )
        )

        self.standard_deviation = (
            None
            if standard_deviation is None
            else np.asarray(
                standard_deviation,
                dtype=np.float32
            )
        )

        self.fitted = bool(
            state.get(
                "fitted",
                self.mean is not None
            )
        )


class CounterPropagationDataset(
    Dataset
):
    def __init__(
        self,
        input_features,
        target_spectra,
        sample_weights=None
    ):
        input_features = np.asarray(
            input_features,
            dtype=np.float32
        )

        target_spectra = np.asarray(
            target_spectra,
            dtype=np.float32
        )

        if input_features.ndim != 2:
            raise ValueError(
                "input_features must be a two-dimensional array."
            )

        if target_spectra.ndim != 2:
            raise ValueError(
                "target_spectra must be a two-dimensional array."
            )

        if len(
            input_features
        ) != len(
            target_spectra
        ):
            raise ValueError(
                "input_features and target_spectra must have the same number "
                "of samples."
            )

        if sample_weights is None:
            sample_weights = np.ones(
                len(
                    input_features
                ),
                dtype=np.float32
            )

        sample_weights = np.asarray(
            sample_weights,
            dtype=np.float32
        ).reshape(
            -1
        )

        if len(
            sample_weights
        ) != len(
            input_features
        ):
            raise ValueError(
                "sample_weights must have one value for each sample."
            )

        self.input_features = torch.tensor(
            input_features,
            dtype=torch.float32
        )

        self.target_spectra = torch.tensor(
            target_spectra,
            dtype=torch.float32
        )

        self.sample_weights = torch.tensor(
            sample_weights,
            dtype=torch.float32
        )

    def __len__(
        self
    ):
        return len(
            self.input_features
        )

    def __getitem__(
        self,
        index
    ):
        return (
            self.input_features[
                index
            ],
            self.target_spectra[
                index
            ],
            self.sample_weights[
                index
            ]
        )


class CounterPropagationNetwork(
    nn.Module
):
    def __init__(
        self,
        input_size,
        output_size,
        num_prototypes=64,
        hidden_size=128,
        dropout=0.10,
        temperature=0.35,
        use_residual=True
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "input_size must be greater than zero."
            )

        if output_size <= 0:
            raise ValueError(
                "output_size must be greater than zero."
            )

        if num_prototypes <= 0:
            raise ValueError(
                "num_prototypes must be greater than zero."
            )

        self.input_size = int(
            input_size
        )
        self.output_size = int(
            output_size
        )
        self.num_prototypes = int(
            num_prototypes
        )
        self.hidden_size = int(
            hidden_size
        )
        self.dropout_rate = float(
            dropout
        )
        self.temperature = float(
            temperature
        )
        self.use_residual = bool(
            use_residual
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                self.input_size,
                self.hidden_size
            ),
            nn.LayerNorm(
                self.hidden_size
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout_rate
            ),
            nn.Linear(
                self.hidden_size,
                self.hidden_size
            ),
            nn.LayerNorm(
                self.hidden_size
            ),
            nn.GELU()
        )

        self.prototypes = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.hidden_size
            )
        )

        self.prototype_outputs = nn.Parameter(
            torch.empty(
                self.num_prototypes,
                self.output_size
            )
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout_rate
            ),
            nn.Linear(
                self.hidden_size,
                self.output_size
            )
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                max(
                    16,
                    self.hidden_size // 2
                )
            ),
            nn.GELU(),
            nn.Linear(
                max(
                    16,
                    self.hidden_size // 2
                ),
                1
            ),
            nn.Softplus()
        )

        self.reset_parameters()

    def reset_parameters(
        self
    ):
        nn.init.normal_(
            self.prototypes,
            mean=0.0,
            std=0.05
        )

        nn.init.normal_(
            self.prototype_outputs,
            mean=0.0,
            std=0.05
        )

    def calculate_distances(
        self,
        encoded_features
    ):
        encoded_squared = torch.sum(
            encoded_features ** 2,
            dim=1,
            keepdim=True
        )

        prototype_squared = torch.sum(
            self.prototypes ** 2,
            dim=1
        ).unsqueeze(
            0
        )

        cross_term = encoded_features @ self.prototypes.transpose(
            0,
            1
        )

        distances = (
            encoded_squared
            + prototype_squared
            - (
                2.0 * cross_term
            )
        )

        return torch.clamp(
            distances,
            min=0.0
        )

    def calculate_activations(
        self,
        distances
    ):
        temperature = max(
            self.temperature,
            1e-6
        )

        return torch.softmax(
            -distances / temperature,
            dim=1
        )

    def forward(
        self,
        input_features,
        return_details=False
    ):
        encoded_features = self.encoder(
            input_features
        )

        distances = self.calculate_distances(
            encoded_features
        )

        activations = self.calculate_activations(
            distances
        )

        prototype_prediction = activations @ self.prototype_outputs

        if self.use_residual:
            residual_prediction = self.residual_head(
                encoded_features
            )

            spectrum = (
                prototype_prediction
                + residual_prediction
            )
        else:
            residual_prediction = torch.zeros_like(
                prototype_prediction
            )

            spectrum = prototype_prediction

        uncertainty = self.uncertainty_head(
            encoded_features
        ).squeeze(
            -1
        )

        if return_details:
            winning_indices = torch.argmin(
                distances,
                dim=1
            )

            return {
                "spectrum": spectrum,
                "uncertainty": uncertainty,
                "encoded_features": encoded_features,
                "distances": distances,
                "activations": activations,
                "winning_indices": winning_indices,
                "prototype_prediction": prototype_prediction,
                "residual_prediction": residual_prediction
            }

        return spectrum


class CounterPropagationLoss(
    nn.Module
):
    def __init__(
        self,
        reconstruction_weight=1.0,
        spectral_angle_weight=0.20,
        smoothness_weight=0.05,
        prototype_weight=0.01,
        uncertainty_weight=0.05,
        epsilon=1e-8
    ):
        super().__init__()

        self.reconstruction_weight = float(
            reconstruction_weight
        )
        self.spectral_angle_weight = float(
            spectral_angle_weight
        )
        self.smoothness_weight = float(
            smoothness_weight
        )
        self.prototype_weight = float(
            prototype_weight
        )
        self.uncertainty_weight = float(
            uncertainty_weight
        )
        self.epsilon = float(
            epsilon
        )

    def weighted_mean(
        self,
        values,
        sample_weights
    ):
        sample_weights = sample_weights.reshape(
            -1
        )

        return torch.sum(
            values * sample_weights
        ) / torch.clamp(
            torch.sum(
                sample_weights
            ),
            min=self.epsilon
        )

    def forward(
        self,
        model_output,
        target_spectra,
        sample_weights=None
    ):
        predicted_spectra = model_output[
            "spectrum"
        ]

        if sample_weights is None:
            sample_weights = torch.ones(
                predicted_spectra.shape[0],
                dtype=predicted_spectra.dtype,
                device=predicted_spectra.device
            )

        per_sample_mse = torch.mean(
            (
                predicted_spectra
                - target_spectra
            ) ** 2,
            dim=1
        )

        reconstruction_loss = self.weighted_mean(
            per_sample_mse,
            sample_weights
        )

        predicted_norm = torch.linalg.norm(
            predicted_spectra,
            dim=1
        )

        target_norm = torch.linalg.norm(
            target_spectra,
            dim=1
        )

        dot_product = torch.sum(
            predicted_spectra * target_spectra,
            dim=1
        )

        cosine_similarity = dot_product / torch.clamp(
            predicted_norm * target_norm,
            min=self.epsilon
        )

        cosine_similarity = torch.clamp(
            cosine_similarity,
            min=-1.0,
            max=1.0
        )

        spectral_angle = torch.acos(
            cosine_similarity
        )

        spectral_angle_loss = self.weighted_mean(
            spectral_angle,
            sample_weights
        )

        if predicted_spectra.shape[1] >= 3:
            predicted_second_difference = (
                predicted_spectra[
                    :,
                    2:
                ]
                - (
                    2.0
                    * predicted_spectra[
                        :,
                        1:-1
                    ]
                )
                + predicted_spectra[
                    :,
                    :-2
                ]
            )

            target_second_difference = (
                target_spectra[
                    :,
                    2:
                ]
                - (
                    2.0
                    * target_spectra[
                        :,
                        1:-1
                    ]
                )
                + target_spectra[
                    :,
                    :-2
                ]
            )

            per_sample_smoothness = torch.mean(
                (
                    predicted_second_difference
                    - target_second_difference
                ) ** 2,
                dim=1
            )

            smoothness_loss = self.weighted_mean(
                per_sample_smoothness,
                sample_weights
            )
        else:
            smoothness_loss = torch.tensor(
                0.0,
                dtype=predicted_spectra.dtype,
                device=predicted_spectra.device
            )

        minimum_distances = torch.min(
            model_output[
                "distances"
            ],
            dim=1
        ).values

        prototype_loss = self.weighted_mean(
            minimum_distances,
            sample_weights
        )

        predicted_uncertainty = model_output[
            "uncertainty"
        ]

        target_uncertainty = torch.sqrt(
            per_sample_mse.detach()
            + self.epsilon
        )

        uncertainty_loss = self.weighted_mean(
            (
                predicted_uncertainty
                - target_uncertainty
            ) ** 2,
            sample_weights
        )

        total_loss = (
            self.reconstruction_weight
            * reconstruction_loss
            + self.spectral_angle_weight
            * spectral_angle_loss
            + self.smoothness_weight
            * smoothness_loss
            + self.prototype_weight
            * prototype_loss
            + self.uncertainty_weight
            * uncertainty_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "spectral_angle_loss": spectral_angle_loss,
            "smoothness_loss": smoothness_loss,
            "prototype_loss": prototype_loss,
            "uncertainty_loss": uncertainty_loss
        }


def set_random_seed(
    seed=42
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


def get_device(
    device=None
):
    if device is not None:
        return torch.device(
            device
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def normalize_spectrum(
    spectrum,
    epsilon=1e-8
):
    spectrum = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    maximum = float(
        np.max(
            np.abs(
                spectrum
            )
        )
    )

    if maximum < epsilon:
        return spectrum

    return spectrum / maximum


def interpolate_spectrum(
    wavelengths,
    spectrum,
    target_wavelengths
):
    wavelengths = np.asarray(
        wavelengths,
        dtype=np.float32
    ).reshape(
        -1
    )

    spectrum = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    target_wavelengths = np.asarray(
        target_wavelengths,
        dtype=np.float32
    ).reshape(
        -1
    )

    if len(
        wavelengths
    ) != len(
        spectrum
    ):
        raise ValueError(
            "wavelengths and spectrum must have the same length."
        )

    if len(
        wavelengths
    ) < 2:
        raise ValueError(
            "At least two wavelength points are required."
        )

    order = np.argsort(
        wavelengths
    )

    wavelengths = wavelengths[
        order
    ]

    spectrum = spectrum[
        order
    ]

    return np.interp(
        target_wavelengths,
        wavelengths,
        spectrum
    ).astype(
        np.float32
    )


def flatten_numeric_values(
    value
):
    flattened_values = []

    if value is None:
        return flattened_values

    if isinstance(
        value,
        bool
    ):
        flattened_values.append(
            float(
                value
            )
        )

        return flattened_values

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        numeric_value = float(
            value
        )

        if math.isfinite(
            numeric_value
        ):
            flattened_values.append(
                numeric_value
            )

        return flattened_values

    if isinstance(
        value,
        np.ndarray
    ):
        return flatten_numeric_values(
            value.tolist()
        )

    if isinstance(
        value,
        dict
    ):
        for key in sorted(
            value.keys()
        ):
            flattened_values.extend(
                flatten_numeric_values(
                    value[
                        key
                    ]
                )
            )

        return flattened_values

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):
        for item in value:
            flattened_values.extend(
                flatten_numeric_values(
                    item
                )
            )

        return flattened_values

    return flattened_values


def extract_feature_by_path(
    data,
    path,
    default=0.0
):
    current_value = data

    for key in path.split(
        "."
    ):
        if not isinstance(
            current_value,
            dict
        ):
            return default

        if key not in current_value:
            return default

        current_value = current_value[
            key
        ]

    if isinstance(
        current_value,
        bool
    ):
        return float(
            current_value
        )

    if isinstance(
        current_value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        current_value = float(
            current_value
        )

        if math.isfinite(
            current_value
        ):
            return current_value

    return default


DEFAULT_FEATURE_PATHS = [
    "molecular_features.molecular_weight",
    "molecular_features.logp",
    "molecular_features.num_atoms",
    "molecular_features.num_heavy_atoms",
    "molecular_features.num_aromatic_atoms",
    "molecular_features.aromatic_fraction",
    "molecular_features.num_rings",
    "molecular_features.num_aromatic_rings",
    "molecular_features.num_rotatable_bonds",
    "molecular_features.num_hydrogen_bond_acceptors",
    "molecular_features.num_hydrogen_bond_donors",
    "molecular_features.topological_polar_surface_area",
    "molecular_features.formal_charge",
    "molecular_features.num_heteroatoms",
    "molecular_features.conjugated_bond_fraction",
    "retrieval_features.highest_similarity",
    "retrieval_features.average_similarity",
    "retrieval_features.num_retrieved_molecules",
    "support_features.structural_support_score",
    "support_features.data_support_score",
    "complexity_features.complexity_score",
    "complexity_features.conjugation_score",
    "complexity_features.aromaticity_score",
    "routing_flags.requires_3d_model",
    "routing_flags.requires_graph_model",
    "routing_flags.requires_transformer",
    "routing_flags.requires_uncertainty_warning"
]


def vectorize_model_features(
    model_features,
    feature_paths=None,
    include_all_numeric=False
):
    if feature_paths is None:
        feature_paths = DEFAULT_FEATURE_PATHS

    vector = [
        extract_feature_by_path(
            model_features,
            path,
            default=0.0
        )
        for path in feature_paths
    ]

    if include_all_numeric:
        vector.extend(
            flatten_numeric_values(
                model_features
            )
        )

    return np.asarray(
        vector,
        dtype=np.float32
    )


def find_prediction_value(
    prediction,
    possible_keys,
    default=None
):
    for key in possible_keys:
        if key in prediction:
            value = prediction[
                key
            ]

            if value is not None:
                return value

    return default


def prediction_to_spectrum(
    prediction,
    target_wavelengths,
    normalize=True
):
    if not isinstance(
        prediction,
        dict
    ):
        raise TypeError(
            "Each model prediction must be a dictionary."
        )

    status = prediction.get(
        "status",
        "success"
    )

    if status != "success":
        raise ValueError(
            prediction.get(
                "error",
                "The model prediction failed."
            )
        )

    spectrum = find_prediction_value(
        prediction,
        [
            "spectrum",
            "absorbance",
            "absorbance_values",
            "intensity",
            "intensities",
            "y",
            "y_values",
            "prediction"
        ]
    )

    wavelengths = find_prediction_value(
        prediction,
        [
            "wavelengths",
            "wavelength",
            "x",
            "x_values",
            "x_nm"
        ]
    )

    if spectrum is None:
        raise ValueError(
            "The model prediction does not contain a spectrum."
        )

    spectrum = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    target_wavelengths = np.asarray(
        target_wavelengths,
        dtype=np.float32
    ).reshape(
        -1
    )

    if wavelengths is None:
        if len(
            spectrum
        ) != len(
            target_wavelengths
        ):
            raise ValueError(
                "A prediction without wavelengths must already match the "
                "target wavelength grid."
            )

        aligned_spectrum = spectrum
    else:
        aligned_spectrum = interpolate_spectrum(
            wavelengths=wavelengths,
            spectrum=spectrum,
            target_wavelengths=target_wavelengths
        )

    if normalize:
        aligned_spectrum = normalize_spectrum(
            aligned_spectrum
        )

    return aligned_spectrum.astype(
        np.float32
    )


def build_counter_prop_input(
    model_features,
    predictions,
    target_wavelengths,
    model_order=None,
    feature_paths=None,
    include_all_numeric=False,
    include_prediction_metadata=True,
    normalize_predictions=True
):
    if model_order is None:
        model_order = [
            prediction.get(
                "model_name",
                f"model_{index}"
            )
            for index, prediction in enumerate(
                predictions
            )
        ]

    prediction_lookup = {
        prediction.get(
            "model_name",
            f"model_{index}"
        ): prediction
        for index, prediction in enumerate(
            predictions
        )
    }

    feature_vector = vectorize_model_features(
        model_features=model_features,
        feature_paths=feature_paths,
        include_all_numeric=include_all_numeric
    )

    input_parts = [
        feature_vector
    ]

    for model_name in model_order:
        prediction = prediction_lookup.get(
            model_name
        )

        if prediction is None:
            aligned_spectrum = np.zeros(
                len(
                    target_wavelengths
                ),
                dtype=np.float32
            )

            status_value = 0.0
            confidence_value = 0.0
            uncertainty_value = 1.0
        else:
            try:
                aligned_spectrum = prediction_to_spectrum(
                    prediction=prediction,
                    target_wavelengths=target_wavelengths,
                    normalize=normalize_predictions
                )

                status_value = 1.0
            except Exception:
                aligned_spectrum = np.zeros(
                    len(
                        target_wavelengths
                    ),
                    dtype=np.float32
                )

                status_value = 0.0

            confidence_value = float(
                prediction.get(
                    "confidence",
                    0.0
                )
                or 0.0
            )

            uncertainty_value = float(
                prediction.get(
                    "uncertainty",
                    0.0
                )
                or 0.0
            )

        input_parts.append(
            aligned_spectrum
        )

        if include_prediction_metadata:
            input_parts.append(
                np.asarray(
                    [
                        status_value,
                        confidence_value,
                        uncertainty_value
                    ],
                    dtype=np.float32
                )
            )

    return np.concatenate(
        input_parts
    ).astype(
        np.float32
    )


def calculate_spectral_metrics(
    predicted_spectra,
    target_spectra,
    epsilon=1e-8
):
    predicted_spectra = np.asarray(
        predicted_spectra,
        dtype=np.float32
    )

    target_spectra = np.asarray(
        target_spectra,
        dtype=np.float32
    )

    difference = (
        predicted_spectra
        - target_spectra
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

    predicted_flat = predicted_spectra.reshape(
        predicted_spectra.shape[0],
        -1
    )

    target_flat = target_spectra.reshape(
        target_spectra.shape[0],
        -1
    )

    dot_product = np.sum(
        predicted_flat * target_flat,
        axis=1
    )

    denominator = (
        np.linalg.norm(
            predicted_flat,
            axis=1
        )
        * np.linalg.norm(
            target_flat,
            axis=1
        )
    )

    cosine_similarity = dot_product / np.maximum(
        denominator,
        epsilon
    )

    cosine_similarity = np.clip(
        cosine_similarity,
        -1.0,
        1.0
    )

    spectral_angle = np.arccos(
        cosine_similarity
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_cosine_similarity": float(
            np.mean(
                cosine_similarity
            )
        ),
        "mean_spectral_angle": float(
            np.mean(
                spectral_angle
            )
        )
    }


class CounterPropagationTrainer:
    def __init__(
        self,
        model,
        device=None,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_clip=5.0,
        loss_function=None
    ):
        self.device = get_device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.learning_rate = float(
            learning_rate
        )
        self.weight_decay = float(
            weight_decay
        )
        self.gradient_clip = float(
            gradient_clip
        )

        self.loss_function = (
            loss_function
            if loss_function is not None
            else CounterPropagationLoss()
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def train_epoch(
        self,
        data_loader
    ):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for (
            input_features,
            target_spectra,
            sample_weights
        ) in data_loader:
            input_features = input_features.to(
                self.device
            )

            target_spectra = target_spectra.to(
                self.device
            )

            sample_weights = sample_weights.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            model_output = self.model(
                input_features,
                return_details=True
            )

            loss_output = self.loss_function(
                model_output=model_output,
                target_spectra=target_spectra,
                sample_weights=sample_weights
            )

            loss = loss_output[
                "loss"
            ]

            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

            self.optimizer.step()

            batch_size = input_features.shape[0]

            total_loss += float(
                loss.detach().cpu().item()
            ) * batch_size

            total_samples += batch_size

        return total_loss / max(
            total_samples,
            1
        )

    def evaluate(
        self,
        data_loader
    ):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for (
                input_features,
                target_spectra,
                sample_weights
            ) in data_loader:
                input_features = input_features.to(
                    self.device
                )

                target_spectra = target_spectra.to(
                    self.device
                )

                sample_weights = sample_weights.to(
                    self.device
                )

                model_output = self.model(
                    input_features,
                    return_details=True
                )

                loss_output = self.loss_function(
                    model_output=model_output,
                    target_spectra=target_spectra,
                    sample_weights=sample_weights
                )

                batch_size = input_features.shape[0]

                total_loss += float(
                    loss_output[
                        "loss"
                    ].detach().cpu().item()
                ) * batch_size

                total_samples += batch_size

                predictions.append(
                    model_output[
                        "spectrum"
                    ].detach().cpu().numpy()
                )

                targets.append(
                    target_spectra.detach().cpu().numpy()
                )

        predicted_spectra = np.concatenate(
            predictions,
            axis=0
        )

        target_spectra = np.concatenate(
            targets,
            axis=0
        )

        metrics = calculate_spectral_metrics(
            predicted_spectra=predicted_spectra,
            target_spectra=target_spectra
        )

        metrics[
            "loss"
        ] = total_loss / max(
            total_samples,
            1
        )

        return metrics

    def fit(
        self,
        train_loader,
        validation_loader=None,
        epochs=100,
        patience=15,
        minimum_improvement=1e-5,
        checkpoint_path=None,
        verbose=True
    ):
        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_rmse": [],
            "validation_mae": [],
            "validation_cosine_similarity": []
        }

        best_validation_loss = float(
            "inf"
        )

        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(
            1,
            epochs + 1
        ):
            train_loss = self.train_epoch(
                train_loader
            )

            history[
                "train_loss"
            ].append(
                train_loss
            )

            if validation_loader is not None:
                validation_metrics = self.evaluate(
                    validation_loader
                )

                validation_loss = validation_metrics[
                    "loss"
                ]

                history[
                    "validation_loss"
                ].append(
                    validation_loss
                )

                history[
                    "validation_rmse"
                ].append(
                    validation_metrics[
                        "rmse"
                    ]
                )

                history[
                    "validation_mae"
                ].append(
                    validation_metrics[
                        "mae"
                    ]
                )

                history[
                    "validation_cosine_similarity"
                ].append(
                    validation_metrics[
                        "mean_cosine_similarity"
                    ]
                )

                improved = (
                    validation_loss
                    < (
                        best_validation_loss
                        - minimum_improvement
                    )
                )

                if improved:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    if checkpoint_path is not None:
                        torch.save(
                            {
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "epoch": epoch,
                                "validation_loss": validation_loss
                            },
                            checkpoint_path
                        )
                else:
                    epochs_without_improvement += 1

                if verbose:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Train loss: {train_loss:.6f} | "
                        f"Validation loss: {validation_loss:.6f} | "
                        f"RMSE: {validation_metrics['rmse']:.6f} | "
                        f"Cosine: "
                        f"{validation_metrics['mean_cosine_similarity']:.6f}"
                    )

                if (
                    patience is not None
                    and patience > 0
                    and epochs_without_improvement >= patience
                ):
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch}. "
                            f"Best epoch: {best_epoch}."
                        )

                    break
            else:
                if verbose:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Train loss: {train_loss:.6f}"
                    )

        return history


class CounterPropagationPredictor:
    def __init__(
        self,
        model,
        input_scaler=None,
        output_scaler=None,
        wavelengths=None,
        model_order=None,
        feature_paths=None,
        include_all_numeric=False,
        include_prediction_metadata=True,
        normalize_predictions=True,
        device=None
    ):
        self.device = get_device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.model.eval()

        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        self.wavelengths = (
            None
            if wavelengths is None
            else np.asarray(
                wavelengths,
                dtype=np.float32
            )
        )

        self.model_order = model_order
        self.feature_paths = feature_paths
        self.include_all_numeric = include_all_numeric
        self.include_prediction_metadata = include_prediction_metadata
        self.normalize_predictions = normalize_predictions

    def predict_vector(
        self,
        input_vector
    ):
        input_vector = np.asarray(
            input_vector,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        if self.input_scaler is not None:
            input_vector = self.input_scaler.transform(
                input_vector
            )

        input_tensor = torch.tensor(
            input_vector,
            dtype=torch.float32,
            device=self.device
        )

        start_time = time.time()

        with torch.no_grad():
            model_output = self.model(
                input_tensor,
                return_details=True
            )

        spectrum = model_output[
            "spectrum"
        ].detach().cpu().numpy()

        uncertainty = model_output[
            "uncertainty"
        ].detach().cpu().numpy()

        winning_index = model_output[
            "winning_indices"
        ].detach().cpu().numpy()

        activations = model_output[
            "activations"
        ].detach().cpu().numpy()

        if self.output_scaler is not None:
            spectrum = self.output_scaler.inverse_transform(
                spectrum
            )

        spectrum = spectrum[
            0
        ].astype(
            np.float32
        )

        uncertainty_value = float(
            uncertainty[
                0
            ]
        )

        winning_prototype = int(
            winning_index[
                0
            ]
        )

        maximum_activation = float(
            np.max(
                activations[
                    0
                ]
            )
        )

        lambda_max = None

        if (
            self.wavelengths is not None
            and len(
                self.wavelengths
            ) == len(
                spectrum
            )
        ):
            lambda_max = float(
                self.wavelengths[
                    int(
                        np.argmax(
                            spectrum
                        )
                    )
                ]
            )

        confidence = float(
            np.clip(
                maximum_activation
                * (
                    1.0
                    / (
                        1.0
                        + uncertainty_value
                    )
                ),
                0.0,
                1.0
            )
        )

        return {
            "model_name": "CounterPropagation",
            "wavelengths": (
                None
                if self.wavelengths is None
                else self.wavelengths.tolist()
            ),
            "spectrum": spectrum.tolist(),
            "lambda_max": lambda_max,
            "confidence": confidence,
            "uncertainty": uncertainty_value,
            "winning_prototype": winning_prototype,
            "prototype_activation": maximum_activation,
            "status": "success",
            "error": None,
            "runtime_seconds": float(
                time.time() - start_time
            ),
            "device": str(
                self.device
            )
        }

    def predict(
        self,
        model_features,
        predictions
    ):
        try:
            if self.wavelengths is None:
                raise ValueError(
                    "wavelengths are required to build the counter-propagation "
                    "input."
                )

            input_vector = build_counter_prop_input(
                model_features=model_features,
                predictions=predictions,
                target_wavelengths=self.wavelengths,
                model_order=self.model_order,
                feature_paths=self.feature_paths,
                include_all_numeric=self.include_all_numeric,
                include_prediction_metadata=self.include_prediction_metadata,
                normalize_predictions=self.normalize_predictions
            )

            return self.predict_vector(
                input_vector
            )

        except Exception as error:
            return {
                "model_name": "CounterPropagation",
                "wavelengths": (
                    None
                    if self.wavelengths is None
                    else self.wavelengths.tolist()
                ),
                "spectrum": None,
                "lambda_max": None,
                "confidence": None,
                "uncertainty": None,
                "winning_prototype": None,
                "prototype_activation": None,
                "status": "failed",
                "error": str(
                    error
                ),
                "traceback": traceback.format_exc(),
                "runtime_seconds": 0.0,
                "device": str(
                    self.device
                )
            }


def save_counter_prop(
    path,
    model,
    input_scaler=None,
    output_scaler=None,
    wavelengths=None,
    model_order=None,
    feature_paths=None,
    include_all_numeric=False,
    include_prediction_metadata=True,
    normalize_predictions=True,
    extra_metadata=None
):
    os.makedirs(
        os.path.dirname(
            path
        )
        or ".",
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": model.input_size,
            "output_size": model.output_size,
            "num_prototypes": model.num_prototypes,
            "hidden_size": model.hidden_size,
            "dropout": model.dropout_rate,
            "temperature": model.temperature,
            "use_residual": model.use_residual
        },
        "input_scaler": (
            None
            if input_scaler is None
            else input_scaler.state_dict()
        ),
        "output_scaler": (
            None
            if output_scaler is None
            else output_scaler.state_dict()
        ),
        "wavelengths": (
            None
            if wavelengths is None
            else np.asarray(
                wavelengths,
                dtype=np.float32
            ).tolist()
        ),
        "model_order": model_order,
        "feature_paths": feature_paths,
        "include_all_numeric": include_all_numeric,
        "include_prediction_metadata": include_prediction_metadata,
        "normalize_predictions": normalize_predictions,
        "extra_metadata": extra_metadata
    }

    torch.save(
        checkpoint,
        path
    )


def load_counter_prop(
    path,
    device=None
):
    device = get_device(
        device
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model_config = checkpoint[
        "model_config"
    ]

    model = CounterPropagationNetwork(
        input_size=model_config[
            "input_size"
        ],
        output_size=model_config[
            "output_size"
        ],
        num_prototypes=model_config.get(
            "num_prototypes",
            64
        ),
        hidden_size=model_config.get(
            "hidden_size",
            128
        ),
        dropout=model_config.get(
            "dropout",
            0.10
        ),
        temperature=model_config.get(
            "temperature",
            0.35
        ),
        use_residual=model_config.get(
            "use_residual",
            True
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    input_scaler = None

    if checkpoint.get(
        "input_scaler"
    ) is not None:
        input_scaler = StandardScaler()
        input_scaler.load_state_dict(
            checkpoint[
                "input_scaler"
            ]
        )

    output_scaler = None

    if checkpoint.get(
        "output_scaler"
    ) is not None:
        output_scaler = StandardScaler()
        output_scaler.load_state_dict(
            checkpoint[
                "output_scaler"
            ]
        )

    predictor = CounterPropagationPredictor(
        model=model,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        wavelengths=checkpoint.get(
            "wavelengths"
        ),
        model_order=checkpoint.get(
            "model_order"
        ),
        feature_paths=checkpoint.get(
            "feature_paths"
        ),
        include_all_numeric=checkpoint.get(
            "include_all_numeric",
            False
        ),
        include_prediction_metadata=checkpoint.get(
            "include_prediction_metadata",
            True
        ),
        normalize_predictions=checkpoint.get(
            "normalize_predictions",
            True
        ),
        device=device
    )

    return {
        "model": model,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "predictor": predictor,
        "checkpoint": checkpoint
    }


def split_indices(
    num_samples,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=42
):
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than zero."
        )

    if not (
        0.0 <= validation_fraction < 1.0
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if not (
        0.0 <= test_fraction < 1.0
    ):
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    if (
        validation_fraction
        + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "validation_fraction and test_fraction must sum to less than 1."
        )

    generator = np.random.default_rng(
        seed
    )

    indices = np.arange(
        num_samples
    )

    generator.shuffle(
        indices
    )

    num_test = int(
        round(
            num_samples
            * test_fraction
        )
    )

    num_validation = int(
        round(
            num_samples
            * validation_fraction
        )
    )

    test_indices = indices[
        :num_test
    ]

    validation_indices = indices[
        num_test:
        num_test + num_validation
    ]

    train_indices = indices[
        num_test + num_validation:
    ]

    return (
        train_indices,
        validation_indices,
        test_indices
    )


def create_data_loaders(
    input_features,
    target_spectra,
    sample_weights=None,
    batch_size=1,
    validation_fraction=0.10,
    test_fraction=0.10,
    seed=42,
    num_workers=0
):
    input_features = np.asarray(
        input_features,
        dtype=np.float32
    )

    target_spectra = np.asarray(
        target_spectra,
        dtype=np.float32
    )

    if sample_weights is None:
        sample_weights = np.ones(
            len(
                input_features
            ),
            dtype=np.float32
        )

    sample_weights = np.asarray(
        sample_weights,
        dtype=np.float32
    )

    (
        train_indices,
        validation_indices,
        test_indices
    ) = split_indices(
        num_samples=len(
            input_features
        ),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed
    )

    train_dataset = CounterPropagationDataset(
        input_features=input_features[
            train_indices
        ],
        target_spectra=target_spectra[
            train_indices
        ],
        sample_weights=sample_weights[
            train_indices
        ]
    )

    validation_dataset = CounterPropagationDataset(
        input_features=input_features[
            validation_indices
        ],
        target_spectra=target_spectra[
            validation_indices
        ],
        sample_weights=sample_weights[
            validation_indices
        ]
    )

    test_dataset = CounterPropagationDataset(
        input_features=input_features[
            test_indices
        ],
        target_spectra=target_spectra[
            test_indices
        ],
        sample_weights=sample_weights[
            test_indices
        ]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices
    }


def train_counter_prop(
    input_features,
    target_spectra,
    wavelengths,
    model_order=None,
    feature_paths=None,
    checkpoint_path="models/counter_prop_best.pth",
    final_model_path="models/counter_prop.pth",
    num_prototypes=64,
    hidden_size=128,
    dropout=0.05,
    temperature=0.50,
    use_residual=True,
    batch_size=1,
    epochs=100,
    learning_rate=0.0001,
    weight_decay=1e-5,
    validation_fraction=0.10,
    test_fraction=0.10,
    patience=20,
    standardize_inputs=True,
    standardize_outputs=False,
    sample_weights=None,
    device=None,
    seed=42
):
    set_random_seed(
        seed
    )

    input_features = np.asarray(
        input_features,
        dtype=np.float32
    )

    target_spectra = np.asarray(
        target_spectra,
        dtype=np.float32
    )

    input_scaler = None

    if standardize_inputs:
        input_scaler = StandardScaler()
        input_scaler.fit(
            input_features
        )

        transformed_inputs = input_scaler.transform(
            input_features
        )
    else:
        transformed_inputs = input_features

    output_scaler = None

    if standardize_outputs:
        output_scaler = StandardScaler()
        output_scaler.fit(
            target_spectra
        )

        transformed_targets = output_scaler.transform(
            target_spectra
        )
    else:
        transformed_targets = target_spectra

    loaders = create_data_loaders(
        input_features=transformed_inputs,
        target_spectra=transformed_targets,
        sample_weights=sample_weights,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed
    )

    model = CounterPropagationNetwork(
        input_size=transformed_inputs.shape[
            1
        ],
        output_size=transformed_targets.shape[
            1
        ],
        num_prototypes=num_prototypes,
        hidden_size=hidden_size,
        dropout=dropout,
        temperature=temperature,
        use_residual=use_residual
    )

    trainer = CounterPropagationTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    history = trainer.fit(
        train_loader=loaders[
            "train_loader"
        ],
        validation_loader=loaders[
            "validation_loader"
        ],
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        verbose=True
    )

    if os.path.exists(
        checkpoint_path
    ):
        best_checkpoint = torch.load(
            checkpoint_path,
            map_location=trainer.device,
            weights_only=False
        )

        model.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )

    test_metrics = trainer.evaluate(
        loaders[
            "test_loader"
        ]
    )

    save_counter_prop(
        path=final_model_path,
        model=model,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        wavelengths=wavelengths,
        model_order=model_order,
        feature_paths=feature_paths,
        extra_metadata={
            "test_metrics": test_metrics,
            "history": history,
            "seed": seed
        }
    )

    return {
        "model": model,
        "trainer": trainer,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "history": history,
        "test_metrics": test_metrics,
        "loaders": loaders,
        "model_path": final_model_path
    }


def main():
    print(
        "counter_prop.py provides the counter-propagation network, training, "
        "saving, loading, and inference utilities."
    )

    print(
        "Import this module from your training script, predict.py, or agent.py."
    )


if __name__ == "__main__":
    main()