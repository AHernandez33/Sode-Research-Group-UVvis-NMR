from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


project_folder = Path(__file__).resolve().parent

prediction_file = (
    project_folder
    / "latest_prediction.json"
)

plot_file = (
    project_folder
    / "latest_mpnn_spectrum.png"
)

csv_file = (
    project_folder
    / "latest_mpnn_spectrum.csv"
)


def get_mpnn_prediction(prediction_output):
    model_predictions = prediction_output.get(
        "model_predictions",
        []
    )

    for prediction in model_predictions:
        model_name = str(
            prediction.get(
                "model_name",
                ""
            )
        ).strip().lower()

        if (
            model_name == "mpnn"
            and prediction.get(
                "status"
            ) == "success"
        ):
            return prediction

    final_prediction = prediction_output.get(
        "final_prediction",
        {}
    )

    final_model_name = str(
        final_prediction.get(
            "model_name",
            ""
        )
    ).strip().lower()

    if (
        final_model_name == "mpnn"
        and final_prediction.get(
            "status"
        ) == "success"
    ):
        return final_prediction

    raise ValueError(
        "No successful MPNN prediction was found in "
        "latest_prediction.json."
    )


def main():
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    with open(
        prediction_file,
        "r",
        encoding="utf-8"
    ) as file:
        prediction_output = json.load(
            file
        )

    prediction = get_mpnn_prediction(
        prediction_output
    )

    wavelengths = np.asarray(
        prediction.get(
            "wavelengths"
        ),
        dtype=np.float32
    )

    spectrum = np.asarray(
        prediction.get(
            "spectrum"
        ),
        dtype=np.float32
    )

    if wavelengths.ndim != 1:
        wavelengths = wavelengths.reshape(
            -1
        )

    if spectrum.ndim != 1:
        spectrum = spectrum.reshape(
            -1
        )

    if wavelengths.shape != spectrum.shape:
        raise ValueError(
            "The wavelength and spectrum arrays have different lengths."
        )

    if len(
        wavelengths
    ) != 181:
        raise ValueError(
            "Expected 181 wavelength values, "
            f"but found {len(wavelengths)}."
        )

    lambda_max = prediction.get(
        "lambda_max"
    )

    if lambda_max is None:
        lambda_max = float(
            wavelengths[
                int(
                    np.argmax(
                        spectrum
                    )
                )
            ]
        )

    smiles = prediction_output.get(
        "smiles",
        ""
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "wavelength_nm",
                "normalized_absorbance"
            ]
        )

        for wavelength, absorbance in zip(
            wavelengths,
            spectrum
        ):
            writer.writerow(
                [
                    float(
                        wavelength
                    ),
                    float(
                        absorbance
                    )
                ]
            )

    lambda_max_index = int(
        np.argmax(
            spectrum
        )
    )

    lambda_max_absorbance = float(
        spectrum[
            lambda_max_index
        ]
    )

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.plot(
        wavelengths,
        spectrum,
        linewidth=2
    )

    plt.scatter(
        [
            lambda_max
        ],
        [
            lambda_max_absorbance
        ]
    )

    plt.annotate(
        f"λmax = {lambda_max:.1f} nm",
        xy=(
            lambda_max,
            lambda_max_absorbance
        ),
        xytext=(
            15,
            -25
        ),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->"
        }
    )

    plt.title(
        f"MPNN Predicted UV-Vis Spectrum\nSMILES: {smiles}"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Normalized absorbance"
    )

    plt.xlim(
        float(
            wavelengths.min()
        ),
        float(
            wavelengths.max()
        )
    )

    plt.ylim(
        0,
        max(
            1.05,
            float(
                spectrum.max()
            )
            * 1.05
        )
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        plot_file,
        dpi=300
    )

    plt.show()

    print(
        "MPNN spectrum plot saved:",
        plot_file
    )

    print(
        "MPNN spectrum CSV saved:",
        csv_file
    )

    print(
        "Lambda max:",
        lambda_max
    )


if __name__ == "__main__":
    main()