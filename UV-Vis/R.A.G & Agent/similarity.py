from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

bits = 2048
fingerprint_rad = 2

rag_folder = Path(__file__).resolve().parent
heatmap_file = rag_folder / "tanimoto_heatmap.png"

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=fingerprint_rad,
    fpSize=bits
)

def smiles_to_fingerprint(smiles):
    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return morgan_generator.GetFingerprint(
        mol
    )

def fingerprint_from_string(fingerprint):
    if fingerprint is None or pd.isna(fingerprint):
        return None

    fingerprint = str(fingerprint).strip()

    if (
        len(fingerprint) == bits
        and set(fingerprint).issubset({"0", "1"})
    ):
        return DataStructs.CreateFromBitString(
            fingerprint
        )

    try:
        fingerprint_list = ast.literal_eval(
            fingerprint
        )

        fingerprint_array = np.asarray(
            fingerprint_list,
            dtype=np.int8
        ).flatten()

        if len(fingerprint_array) != bits:
            return None

        if not np.all(
            np.isin(
                fingerprint_array,
                [0, 1]
            )
        ):
            return None

        fingerprint_string = "".join(
            str(int(bit))
            for bit in fingerprint_array
        )

        return DataStructs.CreateFromBitString(
            fingerprint_string
        )

    except (
        ValueError,
        SyntaxError,
        TypeError
    ):
        return None

def calculate_tanimoto(
        fingerprint_1,
        fingerprint_2
):
    if fingerprint_1 is None or fingerprint_2 is None:
        return None

    return float(
        DataStructs.TanimotoSimilarity(
            fingerprint_1,
            fingerprint_2
        )
    )

def calculate_dice(
    fingerprint_1,
    fingerprint_2
):
    if fingerprint_1 is None or fingerprint_2 is None:
        return None

    return float(
        DataStructs.DiceSimilarity(
            fingerprint_1,
            fingerprint_2
        )
    )

def calculate_cosine(
    fingerprint_1,
    fingerprint_2
):
    if fingerprint_1 is None or fingerprint_2 is None:
        return None

    return float(
        DataStructs.CosineSimilarity(
            fingerprint_1,
            fingerprint_2
        )
    )

def build_tanimoto_matrix(fingerprints):
    number_fingerprints = len(
        fingerprints
    )

    similarity_matrix = np.zeros(
        (
            number_fingerprints,
            number_fingerprints
        ),
        dtype=np.float32
    )

    for row in range(number_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row],
            fingerprints
        )

        similarity_matrix[row, :] = np.asarray(
            similarities,
            dtype=np.float32
        )

    return similarity_matrix

def plot_tanimoto_heatmap(
        similarity_matrix,
        labels=None,
        title="Tanimoto Heatmap",
        output_file=heatmap_file,
        show=True
):
    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=np.float32
    )

    if similarity_matrix.ndim != 2:
        raise ValueError(
            "Matrix must be two dimensional"
        )

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            "The heatmap must be square"
        )

    molecule_count = similarity_matrix.shape[0]

    if molecule_count == 0:
        raise ValueError(
            "The similarity matrix is empty"
        )

    if labels is not None and len(labels) != molecule_count:
        raise ValueError(
            "The number of labels must match the plot size"
        )

    figure_size = 12

    figure, axis = plt.subplots(
        figsize=(
            figure_size,
            figure_size
        )
    )

    heatmap = axis.imshow(
        similarity_matrix,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
        origin="lower"
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Molecule"
    )

    axis.set_ylabel(
        "Molecule"
    )

    tick_count = min(
        11,
        molecule_count
    )

    tick_positions = np.linspace(
        0,
        molecule_count - 1,
        tick_count,
        dtype=int
    )

    if labels is None:
        tick_labels = [
            str(position + 1)
            for position in tick_positions
        ]
    else:
        tick_labels = [
            labels[position]
            for position in tick_positions
        ]

    axis.set_xticks(
        tick_positions
    )

    axis.set_yticks(
        tick_positions
    )

    axis.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right"
    )

    axis.set_yticklabels(
        tick_labels
    )

    colorbar = figure.colorbar(
        heatmap,
        ax=axis
    )

    colorbar.set_label(
        "Tanimoto Similarity"
    )

    if molecule_count <= 20:
        for row in range(molecule_count):
            for column in range(molecule_count):
                axis.text(
                    column,
                    row,
                    f"{similarity_matrix[row, column]:.2f}",
                    ha="center",
                    va="center"
                )

    figure.tight_layout()

    if output_file is not None:
        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight"
        )

        print(
            "Heatmap saved: ",
            output_file
        )

    if show:
        plt.show()

    plt.close(
        figure
    )

def create_heapmap_from_smiles(
    smiles_list,
    labels=None,
    output_file=heatmap_file,
    show=True
):
    valid_fingerprints = []
    valid_labels = []

    for index, smiles in enumerate(
        smiles_list
    ):
        fingerprint = smiles_to_fingerprint(
            smiles
        )

        if fingerprint is None:
            print(
                f"Skipping invalid SMILES: {smiles}"
            )

            continue

        valid_fingerprints.append(
            fingerprint
        )

        if labels is None:
            valid_labels.append(
                str(smiles)
            )

        else:
            valid_labels.append(
                labels[index]
            )

    if not valid_fingerprints:
        raise ValueError(
            "No valid molecular fingerprints were created. "
        )

    similarity_matrix = build_tanimoto_matrix(
        valid_fingerprints
    )

    plot_tanimoto_heatmap(
        similarity_matrix=similarity_matrix,
        labels=valid_labels,
        output_file=output_file,
        show=show
    )

    return similarity_matrix

def main():
    metadata_df = pd.read_csv(
        rag_folder / "indexed_metadata.csv",
        dtype={
            "morgan_fingerprint": "string"
        },
        keep_default_na=False
    )

    fingerprints = []
    labels = []

    for _, row in metadata_df.iterrows():
        fingerprint = fingerprint_from_string(
            row["morgan_fingerprint"]
        )

        if fingerprint is None:
            continue

        fingerprints.append(
            fingerprint
        )

        if "molecule_id" in metadata_df.columns:
            labels.append(
                str(row["molecule_id"])
            )
        else:
            labels.append(
                str(len(labels) + 1)
            )

    if not fingerprints:
        raise ValueError(
            "No valid molecular fingerprints were created. "
        )

    similarity_matrix = build_tanimoto_matrix(
        fingerprints
    )

    plot_tanimoto_heatmap(
        similarity_matrix=similarity_matrix,
        labels=labels,
        output_file=heatmap_file,
        show=False
    )

    print()
    print(
        "Matrix shape:",
        similarity_matrix.shape
    )

    print(
        "Symmetric:",
        np.allclose(
            similarity_matrix,
            similarity_matrix.T
        )
    )

    print(
        "Diagonal all ones:",
        np.allclose(
            np.diag(similarity_matrix),
            1.0
        )
    )

if __name__ == "__main__":
    main()




# llm.py
# model_features.py
# counter_prop.py
# router.py
# model_wrappers.py
# predict.py
# agent.py