import json
from pathlib import Path
import faiss 
import numpy as np
import pandas as pd

rag_folder = Path(__file__).resolve().parent 
folder = rag_folder.parent 

input_file = folder / "UV_metadata.csv"

index_file = rag_folder / "fingerprint_index.faiss"
metadata_file = rag_folder / "indexed_metadata.csv"
index_info_file = rag_folder / "index_info.json"

fingerprint_col = "morgan_fingerprint"
fingerprint_bits = 2048

def fingerprint_to_array(fingerprint):
    if pd.isna(fingerprint):
        return None
    
    fingerprint = str(fingerprint).strip()

    if len(fingerprint) != fingerprint_bits:
        return None
    
    if set(fingerprint) - {"0", "1"}:
        return None
    
    return np.fromiter(
        (int(bit) for bit in fingerprint),
        dtype=np.float32,
        count=fingerprint_bits
    )

def load_dataset():
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )
    
    df = pd.read_csv(
        input_file,
        dtype={
            fingerprint_col: "string"
        }
    )

    if fingerprint_col not in df.columns:
        raise ValueError(
            f"Column '{fingerprint_col}' was not found"
            f"Available: {list(df.columns)}"
        )
    
    if "smiles" not in df.columns:
        raise ValueError(
            "Column 'smiles' was not found in the dataset"
        )
    
    return df


def prepare_fingerprints(df):
    vectors = []
    valid_rows = []
    invalid_count = 0

    total = len(df)

    for index, row in df.iterrows():
        fingerprint = fingerprint_to_array(
            row[fingerprint_col]
        )

        if fingerprint is None:
            invalid_count += 1
            continue 

        vectors.append(fingerprint)

        row_data = row.to_dict()
        row_data["orginal_row_index"] = int(index)
        valid_rows.append(row_data)
    
    if not vectors:
        raise ValueError(
            "No valid morgan fingerprints are found"
        )
    
    fingerprint_vec = np.asarray(
        vectors,
        dtype=np.float32
    )

    metadata_df = pd.DataFrame(valid_rows)
    metadata_df.reset_index(
        drop=True,
        inplace=True
    )

    metadata_df.insert(
        0,
        "faiss_id",
        np.arange(
            len(metadata_df),
            dtype=np.int64
        )
    )

    return fingerprint_vec, metadata_df, invalid_count


def build_index(fingerprint_vec):
    norm_vec = fingerprint_vec.copy()

    faiss.normalize_L2(
        norm_vec
    )

    dim = norm_vec.shape[1]

    index = faiss.IndexFlatIP(
        dim
    )

    index.add(
        norm_vec
    )
    return index

def save_files(
    index,
    metadata_df,
    invalid_count
):
    faiss.write_index(
        index,
        str(index_file)
    )

    metadata_df.to_csv(
        metadata_file,
        index=False
    )

    index_information = {
        "input_file": str(input_file),
        "index_file": str(index_file),
        "metadata_file": str(metadata_file),
        "fingerprint_type": "Morgan",
        "fingerprint_radius": 2,
        "similarity_index": "IndexFlatIP",
        "vectors_normalized": True,
        "number_of_indexed_molecules": int(index.ntotal),
        "number_of_invalid_fingerprints": int(invalid_count)
    }

    with open(
        index_info_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            index_information,
            file,
            indent=4
        )

def main():
    print()
    print("Loading:", input_file)
    df = load_dataset()
    print("Dataset rows: ", len(df))
    print()

    fingerprint_vec, metadata_df, invalid_count = (
        prepare_fingerprints(df)
    )

    print()
    print(
        "Fingerprint matrix shape: ",
        fingerprint_vec.shape
    )

    index = build_index(
        fingerprint_vec
    )

    save_files(
        index,
        metadata_df,
        invalid_count
    )

    print()
    print("Index built successfully")
    print("Indexed molecules:", index.ntotal)
    print("Invalid fingerprints:", invalid_count)
    print("Vector dimensions:", index.d)
    print()
    print("Finished")

if __name__ == "__main__":
    main()


