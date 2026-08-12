from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


project_folder = Path(__file__).resolve().parent

prediction_file = (
    project_folder
    / "latest_transformer_prediction.json"
)

plot_file = (
    project_folder
    / "latest_transformer_spectrum.png"
)

csv_file = (
    project_folder
    / "latest_transformer_spectrum.csv"
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
        prediction = json.load(file)

    if prediction.get("status") != "success":
        raise ValueError(
            "The Transformer prediction was not successful. "
            f"Error: {prediction.get('error')}"
        )

    wavelengths = np.asarray(
        prediction["wavelengths"],
        dtype=np.float32
    )

    spectrum = np.asarray(
        prediction["spectrum"],
        dtype=np.float32
    )

    if wavelengths.shape != spectrum.shape:
        raise ValueError(
            "The wavelength and spectrum arrays have different lengths."
        )

    lambda_max = prediction.get("lambda_max")
    smiles = prediction.get("smiles", "")

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

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
                    float(wavelength),
                    float(absorbance)
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

    if lambda_max is not None:
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
        f"Predicted UV-Vis Spectrum\nSMILES: {smiles}"
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
        "Spectrum plot saved:",
        plot_file
    )

    print(
        "Spectrum CSV saved:",
        csv_file
    )

    print(
        "Lambda max:",
        lambda_max
    )


if __name__ == "__main__":
    main()