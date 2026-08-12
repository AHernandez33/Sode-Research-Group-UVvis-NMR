import tempfile
from pathlib import Path
import pandas as pd

from original_config import command_timeout_seconds
from original_config import python_executable
from original_config import transformer_folder
from original_config import transformer_model_file
from original_config import transformer_predict_file
from original_config import wavelengths

from common import build_failed_result
from common import build_prediction_result
from common import ensure_exists
from common import read_spectrum_csv
from common import run_command


def build_transformer_command(
        input_file,
        output_file,
        model_file
):
    return [
        str(
            Path(
                python_executable
            ).resolve()
        ),
        str(
            Path(
                transformer_predict_file
            ).resolve()
        ),
        "-m",
        str(
            Path(
                model_file
            ).resolve()
        ),
        "-i",
        str(
            Path(
                input_file
            ).resolve()
        ),
        "-o",
        str(
            Path(
                output_file
            ).resolve()
        ),
        "--twod"
    ]


def predict_transformer(
        smiles,
        model_file=transformer_model_file,
        normalize=True
):
    try:
        ensure_exists(
            Path(
                python_executable
            ),
            "Python executable"
        )

        ensure_exists(
            transformer_folder,
            "PNNL Transformer folder"
        )

        ensure_exists(
            transformer_predict_file,
            "PNNL Transformer predict.py"
        )

        if model_file is None:
            raise ValueError(
                "Set transformer_model_file in original_config.py "
                "to the exact trained .model file."
            )

        ensure_exists(
            model_file,
            "PNNL Transformer model file"
        )

        with tempfile.TemporaryDirectory(
            prefix="pnnl_transformer_"
        ) as temporary_directory:
            temporary_directory = Path(
                temporary_directory
            )

            input_file = (
                temporary_directory
                / "input.csv"
            )

            output_file = (
                temporary_directory
                / "predictions.csv"
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

            command = build_transformer_command(
                input_file=input_file,
                output_file=output_file,
                model_file=model_file
            )

            print()
            print(
                "Transformer command:"
            )

            print(
                command
            )

            print(
                "Transformer working folder:",
                Path(
                    transformer_folder
                ).resolve()
            )

            result = run_command(
                command=command,
                working_folder=str(
                    Path(
                        transformer_folder
                    ).resolve()
                ),
                timeout_seconds=command_timeout_seconds
            )

            output_wavelengths, spectrum = read_spectrum_csv(
                output_file=output_file,
                target_wavelengths=wavelengths,
                smiles=smiles
            )

            return build_prediction_result(
                model_name="transformer",
                wavelengths=output_wavelengths,
                spectrum=spectrum,
                raw_output_file=output_file,
                stdout=result.stdout,
                normalize=normalize
            )

    except Exception as error:
        return build_failed_result(
            model_name="transformer",
            error=error
        )


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    prediction = predict_transformer(
        smiles
    )

    print(
        prediction
    )


if __name__ == "__main__":
    main()

# import tempfile
# from pathlib import Path
# import pandas as pd

# from original_config import command_timeout_seconds
# from original_config import python_executable
# from original_config import transformer_folder
# from original_config import transformer_model_file
# from original_config import transformer_predict_file
# from original_config import wavelengths

# from common import build_failed_result
# from common import build_prediction_result
# from common import ensure_exists
# from common import read_spectrum_csv
# from common import run_command


# def build_transformer_command(
#         input_file,
#         output_file,
#         model_file
# ):
#     return [
#         python_executable,
#         transformer_predict_file,
#         "-m",
#         model_file,
#         "-i",
#         input_file,
#         "-o",
#         output_file,
#         "--twod"
#     ]


# def predict_transformer(
#         smiles,
#         model_file=transformer_model_file,
#         normalize=True
# ):
#     try:
#         ensure_exists(
#             transformer_folder,
#             "PNNL Transformer folder"
#         )

#         ensure_exists(
#             transformer_predict_file,
#             "PNNL Transformer predict.py"
#         )

#         if model_file is None:
#             raise ValueError(
#                 "Set transformer_model_file in original_config.py "
#                 "to the exact trained .model file."
#             )

#         ensure_exists(
#             model_file,
#             "PNNL Transformer model file"
#         )

#         with tempfile.TemporaryDirectory(
#             prefix="pnnl_transformer_"
#         ) as temporary_directory:
#             temporary_directory = Path(
#                 temporary_directory
#             )

#             input_file = (
#                 temporary_directory
#                 / "input.csv"
#             )

#             output_file = (
#                 temporary_directory
#                 / "predictions.csv"
#             )

#             pd.DataFrame(
#                 {
#                     "smiles": [
#                         smiles
#                     ]
#                 }
#             ).to_csv(
#                 input_file,
#                 index=False
#             )

#             command = build_transformer_command(
#                 input_file=input_file,
#                 output_file=output_file,
#                 model_file=model_file
#             )

#             result = run_command(
#                 command=command,
#                 working_folder=transformer_folder,
#                 timeout_seconds=command_timeout_seconds
#             )

#             output_wavelengths, spectrum = read_spectrum_csv(
#                 output_file=output_file,
#                 target_wavelengths=wavelengths,
#                 smiles=smiles
#             )

#             return build_prediction_result(
#                 model_name="transformer",
#                 wavelengths=output_wavelengths,
#                 spectrum=spectrum,
#                 raw_output_file=output_file,
#                 stdout=result.stdout,
#                 normalize=normalize
#             )

#     except Exception as error:
#         return build_failed_result(
#             model_name="transformer",
#             error=error
#         )


# def main():
#     smiles = input(
#         "Enter a SMILES string: "
#     ).strip()

#     prediction = predict_transformer(
#         smiles
#     )

#     print(
#         prediction
#     )


# if __name__ == "__main__":
#     main()