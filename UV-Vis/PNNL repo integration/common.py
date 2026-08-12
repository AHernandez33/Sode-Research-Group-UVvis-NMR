from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import ast


def ensure_exists(
        path,
        description
):
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    return path


def run_command(
        command,
        working_folder,
        timeout_seconds=3600
):
    result = subprocess.run(
        [
            str(
                value
            )
            for value in command
        ],
        cwd=str(
            working_folder
        ),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            "PNNL command failed.\n"
            f"Command: {' '.join(str(value) for value in command)}\n"
            f"Standard output:\n{result.stdout}\n"
            f"Standard error:\n{result.stderr}"
        )

    return result


def normalize_spectrum(
        spectrum
):
    spectrum = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    if spectrum.size == 0:
        raise ValueError(
            "The predicted spectrum is empty."
        )

    minimum = float(
        np.min(
            spectrum
        )
    )

    maximum = float(
        np.max(
            spectrum
        )
    )

    if maximum == minimum:
        return np.zeros_like(
            spectrum,
            dtype=np.float32
        )

    return (
        (
            spectrum
            - minimum
        )
        /
        (
            maximum
            - minimum
        )
    ).astype(
        np.float32
    )


def calculate_lambda_max(
        wavelengths,
        spectrum
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

    if len(
        wavelengths
    ) != len(
        spectrum
    ):
        return None

    return float(
        wavelengths[
            int(
                np.argmax(
                    spectrum
                )
            )
        ]
    )


def read_spectrum_csv(
        output_file,
        target_wavelengths,
        smiles=None
):
    output_file = ensure_exists(
        output_file,
        "Prediction output file"
    )

    target_wavelengths = [
        int(
            wavelength
        )
        for wavelength in target_wavelengths
    ]

    dataframe = pd.read_csv(
        output_file
    )

    if not dataframe.empty:
        wavelength_columns = [
            str(
                wavelength
            )
            for wavelength in target_wavelengths
        ]

        if all(
            column in dataframe.columns
            for column in wavelength_columns
        ):
            row = dataframe.iloc[
                0
            ]

            spectrum = [
                float(
                    row[
                        column
                    ]
                )
                for column in wavelength_columns
            ]

            return target_wavelengths, spectrum

        for spectrum_column in [
            "spectrum",
            "prediction",
            "predictions",
            "absorbance",
            "absorbance_values",
            "intensity",
            "intensities"
        ]:
            if spectrum_column in dataframe.columns:
                spectrum = parse_list_value(
                    dataframe.iloc[
                        0
                    ][
                        spectrum_column
                    ]
                )

                if len(
                    spectrum
                ) != len(
                    target_wavelengths
                ):
                    raise ValueError(
                        f"The predicted spectrum contains "
                        f"{len(spectrum)} values, but "
                        f"{len(target_wavelengths)} were expected."
                    )

                return target_wavelengths, spectrum

    dataframe_no_header = pd.read_csv(
        output_file,
        header=None
    )

    if dataframe_no_header.empty:
        raise ValueError(
            f"The prediction output is empty: {output_file}"
        )

    row = dataframe_no_header.iloc[
        0
    ].tolist()

    if (
        len(
            row
        ) == len(
            target_wavelengths
        ) + 1
    ):
        first_value = str(
            row[
                0
            ]
        ).strip()

        if (
            smiles is None
            or first_value == str(
                smiles
            ).strip()
            or first_value.lower() == "smiles"
        ):
            row = row[
                1:
            ]

    numeric_spectrum = []

    for value in row:
        try:
            numeric_spectrum.append(
                float(
                    value
                )
            )
        except (
            TypeError,
            ValueError
        ):
            continue

    if len(
        numeric_spectrum
    ) == len(
        target_wavelengths
    ):
        return (
            target_wavelengths,
            numeric_spectrum
        )

    header_values = list(
        dataframe.columns
    )

    if (
        len(
            header_values
        ) == len(
            target_wavelengths
        ) + 1
        and str(
            header_values[
                0
            ]
        ).strip().lower() == "smiles"
    ):
        header_values = header_values[
            1:
        ]

    numeric_header = []

    for value in header_values:
        try:
            numeric_header.append(
                float(
                    value
                )
            )
        except (
            TypeError,
            ValueError
        ):
            continue

    if len(
        numeric_header
    ) == len(
        target_wavelengths
    ):
        return (
            target_wavelengths,
            numeric_header
        )

    raise ValueError(
        "The output CSV format could not be recognized. "
        f"Expected {len(target_wavelengths)} spectrum values. "
        f"Found {len(numeric_spectrum)} numeric row values and "
        f"{len(numeric_header)} numeric header values."
    )

def parse_list_value(
        value
):
    if isinstance(
        value,
        list
    ):
        return value

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    value = str(
        value
    ).strip()

    if not value:
        return []

    parsed_value = ast.literal_eval(
        value
    )

    if not isinstance(
        parsed_value,
        (
            list,
            tuple
        )
    ):
        raise ValueError(
            "The spectrum value is not a list."
        )

    return list(
        parsed_value
    )

def build_prediction_result(
        model_name,
        wavelengths,
        spectrum,
        raw_output_file=None,
        stdout=None,
        normalize=True
):
    wavelengths = [
        float(
            value
        )
        for value in wavelengths
    ]

    spectrum = np.asarray(
        spectrum,
        dtype=np.float32
    ).reshape(
        -1
    )

    if normalize:
        spectrum = normalize_spectrum(
            spectrum
        )

    return {
        "model_name": model_name,
        "wavelengths": wavelengths,
        "spectrum": spectrum.tolist(),
        "lambda_max": calculate_lambda_max(
            wavelengths,
            spectrum
        ),
        "confidence": None,
        "uncertainty": None,
        "status": "success",
        "error": None,
        "raw_output_file": (
            None
            if raw_output_file is None
            else str(
                raw_output_file
            )
        ),
        "stdout": stdout
    }


def build_failed_result(
        model_name,
        error
):
    return {
        "model_name": model_name,
        "wavelengths": None,
        "spectrum": None,
        "lambda_max": None,
        "confidence": None,
        "uncertainty": None,
        "status": "failed",
        "error": str(
            error
        )
    }