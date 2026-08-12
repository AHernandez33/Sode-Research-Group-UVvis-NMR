import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent

prediction_file = PROJECT_FOLDER / "latest_prediction.json"

png_file = PROJECT_FOLDER / "latest_final_spectrum.png"
csv_file = PROJECT_FOLDER / "latest_final_spectrum.csv"


def main():

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_file}"
        )

    with open(prediction_file, "r", encoding="utf-8") as file:
        prediction = json.load(file)

    final_prediction = prediction.get("final_prediction", {})

    if final_prediction.get("status") != "success":
        raise RuntimeError(
            f"Prediction failed:\n{final_prediction.get('error')}"
        )

    wavelengths = final_prediction.get("wavelengths")
    spectrum = final_prediction.get("spectrum")

    if wavelengths is None:
        wavelengths = list(range(220, 401))

    wavelengths = np.asarray(
        wavelengths,
        dtype=float
    )

    spectrum = np.asarray(
        spectrum,
        dtype=float
    )

    if len(wavelengths) != len(spectrum):
        raise ValueError(
            "Spectrum length does not match wavelength length."
        )

    lambda_max = float(
        wavelengths[np.argmax(spectrum)]
    )

    confidence = final_prediction.get(
        "confidence"
    )

    uncertainty = final_prediction.get(
        "uncertainty"
    )

    model_name = final_prediction.get(
        "model_name",
        "Unknown"
    )

    ##################################################
    # Save CSV
    ##################################################

    df = pd.DataFrame(
        {
            "Wavelength (nm)": wavelengths,
            "Absorbance": spectrum
        }
    )

    df.to_csv(
        csv_file,
        index=False
    )

    ##################################################
    # Plot
    ##################################################

    plt.figure(figsize=(10, 6))

    plt.plot(
        wavelengths,
        spectrum,
        linewidth=2,
        label=model_name
    )

    plt.scatter(
        [lambda_max],
        [np.max(spectrum)],
        s=60,
        color="red",
        zorder=10
    )

    plt.axvline(
        lambda_max,
        linestyle="--",
        linewidth=1.5
    )

    plt.text(
        lambda_max + 2,
        np.max(spectrum),
        f"λmax = {lambda_max:.1f} nm"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")

    plt.title(
        f"Predicted UV-Vis Spectra ({model_name})"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        png_file,
        dpi=300
    )

    plt.show()

    ##################################################
    # Summary
    ##################################################

    print()

    print("Final UV-Vis Prediction")
    print("-----------------------")

    print("Model:", model_name)

    print("Lambda max:", lambda_max)

    print("Confidence:", confidence)

    print("Uncertainty:", uncertainty)

    print()

    print("Spectrum CSV:")
    print(csv_file)

    print()

    print("Spectrum image:")
    print(png_file)


if __name__ == "__main__":
    main()