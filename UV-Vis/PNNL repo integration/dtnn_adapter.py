import tempfile
from pathlib import Path
import pandas as pd

from original_config import command_timeout_seconds
from original_config import dtnn_folder
from original_config import dtnn_predict_file
from original_config import python_executable
from original_config import wavelengths

from common import build_failed_result
from common import build_prediction_result
from common import ensure_exists
from common import read_spectrum_csv
from common import run_command


def build_dtnn_command(
        input_file,
        output_file
):
    return [
        python_executable,
        dtnn_predict_file,
        "--input",
        input_file,
        "--output",
        output_file
    ]


def predict_dtnn(
        smiles,
        normalize=True
):
    try:
        ensure_exists(
            dtnn_folder,
            "PNNL DTNN folder"
        )

        ensure_exists(
            dtnn_predict_file,
            "PNNL DTNN predict.py"
        )

        with tempfile.TemporaryDirectory(
            prefix="pnnl_dtnn_"
        ) as temporary_directory:
            temporary_directory = Path(
                temporary_directory
            )

            input_file = (
                temporary_directory
                / "smiles.csv"
            )

            output_file = (
                temporary_directory
                / "uv_preds.csv"
            )

            pd.DataFrame(
                {
                    "smiles": [
                        smiles
                    ]
                }
            ).to_csv(
                input_file,
                index=False
            )

            command = build_dtnn_command(
                input_file=input_file,
                output_file=output_file
            )

            result = run_command(
                command=command,
                working_folder=dtnn_folder,
                timeout_seconds=command_timeout_seconds
            )

            output_wavelengths, spectrum = read_spectrum_csv(
                output_file=output_file,
                target_wavelengths=wavelengths,
                smiles=smiles
            )

            return build_prediction_result(
                model_name="DTNN",
                wavelengths=output_wavelengths,
                spectrum=spectrum,
                raw_output_file=output_file,
                stdout=result.stdout,
                normalize=normalize
            )

    except Exception as error:
        return build_failed_result(
            model_name="DTNN",
            error=(
                f"{error}\n"
                "Open UVvis-DTNN/predict.py and run "
                "'python predict.py --help'. Then edit build_dtnn_command() "
                "so its arguments match the original script."
            )
        )


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    prediction = predict_dtnn(
        smiles
    )

    print(
        prediction
    )


if __name__ == "__main__":
    main()