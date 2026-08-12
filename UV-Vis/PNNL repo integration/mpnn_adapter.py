import tempfile
from pathlib import Path
import pandas as pd

from original_config import command_timeout_seconds
from original_config import mpnn_checkpoint_folder
from original_config import mpnn_folder
from original_config import mpnn_predict_file
from original_config import python_executable
from original_config import wavelengths

from common import build_failed_result
from common import build_prediction_result
from common import ensure_exists
from common import read_spectrum_csv
from common import run_command


def predict_mpnn(
        smiles,
        checkpoint_folder=mpnn_checkpoint_folder,
        normalize=True
):
    try:
        ensure_exists(
            mpnn_folder,
            "PNNL MPNN folder"
        )

        ensure_exists(
            mpnn_predict_file,
            "PNNL MPNN predict.py"
        )

        ensure_exists(
            checkpoint_folder,
            "PNNL MPNN checkpoint folder"
        )

        with tempfile.TemporaryDirectory(
            prefix="pnnl_mpnn_"
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

            command = [
                python_executable,
                mpnn_predict_file,
                "--test_path",
                input_file,
                "--checkpoint_dir",
                checkpoint_folder,
                "--preds_path",
                output_file
            ]

            result = run_command(
                command=command,
                working_folder=mpnn_folder,
                timeout_seconds=command_timeout_seconds
            )

            output_wavelengths, spectrum = read_spectrum_csv(
                output_file=output_file,
                target_wavelengths=wavelengths,
                smiles=smiles
            )

            return build_prediction_result(
                model_name="MPNN",
                wavelengths=output_wavelengths,
                spectrum=spectrum,
                raw_output_file=output_file,
                stdout=result.stdout,
                normalize=normalize
            )

    except Exception as error:
        return build_failed_result(
            model_name="MPNN",
            error=error
        )


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    prediction = predict_mpnn(
        smiles
    )

    print(
        prediction
    )


if __name__ == "__main__":
    main()