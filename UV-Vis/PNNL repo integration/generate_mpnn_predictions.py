from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


agent_folder = Path(
    __file__
).resolve().parent

uv_vis_folder = agent_folder.parent

integration_folder = (
    uv_vis_folder
    / "PNNL repo integration"
)

if str(
    integration_folder
) not in sys.path:
    sys.path.insert(
        0,
        str(
            integration_folder
        )
    )

from common import read_spectrum_csv
from common import run_command
from original_config import command_timeout_seconds
from original_config import mpnn_checkpoint_folder
from original_config import mpnn_folder
from original_config import mpnn_predict_file
from original_config import python_executable
from original_config import wavelengths


input_file = (
    uv_vis_folder
    / "UV_vis_merged.csv"
)

temporary_input_file = (
    agent_folder
    / "mpnn_batch_input.csv"
)

raw_prediction_file = (
    agent_folder
    / "mpnn_batch_raw_predictions.csv"
)

output_file = (
    agent_folder
    / "base_predictions_mpnn.csv"
)

failure_file = (
    agent_folder
    / "base_predictions_mpnn_failures.csv"
)


smiles_column_candidates = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "Canonical_SMILES"
]

molecule_id_column_candidates = [
    "molecule_id",
    "Molecule_ID",
    "compound_id",
    "id"
]


def find_column(
        dataframe,
        candidates,
        required=True
):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise ValueError(
            "None of the expected columns were found: "
            f"{candidates}. Available columns: "
            f"{list(dataframe.columns)}"
        )

    return None


def validate_input_dataframe(
        dataframe,
        smiles_column
):
    if dataframe.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    if dataframe[
        smiles_column
    ].astype(
        str
    ).str.strip().eq(
        ""
    ).all():
        raise ValueError(
            "The SMILES column contains no usable values."
        )


def build_batch_input(
        dataframe,
        smiles_column
):
    batch_input_df = pd.DataFrame(
        {
            "smiles": dataframe[
                smiles_column
            ].astype(
                str
            ).str.strip()
        }
    )

    batch_input_df.to_csv(
        temporary_input_file,
        index=False
    )


def run_batch_prediction():
    command = [
        python_executable,
        mpnn_predict_file,
        "--test_path",
        temporary_input_file,
        "--checkpoint_dir",
        mpnn_checkpoint_folder,
        "--preds_path",
        raw_prediction_file
    ]

    return run_command(
        command=command,
        working_folder=mpnn_folder,
        timeout_seconds=command_timeout_seconds
    )


def read_raw_predictions(
        expected_rows
):
    prediction_df = pd.read_csv(
        raw_prediction_file,
        header=None
    )

    if prediction_df.empty:
        raise ValueError(
            "The MPNN prediction output is empty."
        )

    if prediction_df.shape[
        1
    ] == len(
        wavelengths
    ) + 1:
        first_column = prediction_df.iloc[
            :,
            0
        ]

        prediction_df = prediction_df.iloc[
            :,
            1:
        ]

    if prediction_df.shape[
        1
    ] != len(
        wavelengths
    ):
        raise ValueError(
            "The MPNN prediction output has "
            f"{prediction_df.shape[1]} columns, but "
            f"{len(wavelengths)} were expected."
        )

    if len(
        prediction_df
    ) != expected_rows:
        raise ValueError(
            "The MPNN prediction output has "
            f"{len(prediction_df)} rows, but "
            f"{expected_rows} were expected."
        )

    prediction_df = prediction_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if prediction_df.isna().any().any():
        raise ValueError(
            "The MPNN prediction output contains non-numeric values."
        )

    prediction_df.columns = [
        f"MPNN_{int(wavelength)}"
        for wavelength in wavelengths
    ]

    return prediction_df.reset_index(
        drop=True
    )


def build_output_dataframe(
        input_df,
        prediction_df,
        smiles_column,
        molecule_id_column
):
    output_df = pd.DataFrame()

    if molecule_id_column is not None:
        output_df[
            molecule_id_column
        ] = input_df[
            molecule_id_column
        ].values

    output_df[
        smiles_column
    ] = input_df[
        smiles_column
    ].values

    for wavelength in wavelengths:
        experimental_column = str(
            int(
                wavelength
            )
        )

        output_df[
            f"experimental_{int(wavelength)}"
        ] = (
            input_df[
                experimental_column
            ].values
            if experimental_column in input_df.columns
            else np.nan
        )

    output_df = pd.concat(
        [
            output_df.reset_index(
                drop=True
            ),
            prediction_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    spectrum_columns = [
        f"MPNN_{int(wavelength)}"
        for wavelength in wavelengths
    ]

    spectrum_array = output_df[
        spectrum_columns
    ].to_numpy(
        dtype=np.float32
    )

    lambda_max_indices = np.argmax(
        spectrum_array,
        axis=1
    )

    output_df[
        "MPNN_lambda_max"
    ] = [
        float(
            wavelengths[
                index
            ]
        )
        for index in lambda_max_indices
    ]

    output_df[
        "MPNN_status"
    ] = "success"

    output_df[
        "MPNN_error"
    ] = None

    return output_df


def main():
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input CSV was not found: {input_file}"
        )

    input_df = pd.read_csv(
        input_file,
        keep_default_na=False
    )

    smiles_column = find_column(
        dataframe=input_df,
        candidates=smiles_column_candidates
    )

    molecule_id_column = find_column(
        dataframe=input_df,
        candidates=molecule_id_column_candidates,
        required=False
    )

    validate_input_dataframe(
        dataframe=input_df,
        smiles_column=smiles_column
    )

    print()
    print(
        "Batch MPNN prediction"
    )
    print(
        "---------------------"
    )

    print(
        "Input file:",
        input_file
    )

    print(
        "Molecules:",
        len(
            input_df
        )
    )

    print(
        "Checkpoint folder:",
        mpnn_checkpoint_folder
    )

    build_batch_input(
        dataframe=input_df,
        smiles_column=smiles_column
    )

    start_time = time.time()

    result = run_batch_prediction()

    elapsed_seconds = (
        time.time()
        - start_time
    )

    prediction_df = read_raw_predictions(
        expected_rows=len(
            input_df
        )
    )

    output_df = build_output_dataframe(
        input_df=input_df,
        prediction_df=prediction_df,
        smiles_column=smiles_column,
        molecule_id_column=molecule_id_column
    )

    output_df[
        "MPNN_total_batch_seconds"
    ] = float(
        elapsed_seconds
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        "Prediction finished"
    )

    print(
        "Rows saved:",
        len(
            output_df
        )
    )

    print(
        "Spectrum columns:",
        len(
            wavelengths
        )
    )

    print(
        "Elapsed seconds:",
        round(
            elapsed_seconds,
            2
        )
    )

    print(
        "Saved:",
        output_file
    )

    if result.stdout:
        print()
        print(
            "PNNL output summary:"
        )

        for line in result.stdout.splitlines():
            if (
                line.startswith(
                    "Loading training args"
                )
                or line.startswith(
                    "Loading data"
                )
                or line.startswith(
                    "Test size"
                )
                or line.startswith(
                    "Predicting with"
                )
                or line.startswith(
                    "Saving predictions"
                )
            ):
                print(
                    line
                )


if __name__ == "__main__":
    main()