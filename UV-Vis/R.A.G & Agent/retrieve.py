from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from rdkit import Chem
# from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
# import ast

from similarity import (
    fingerprint_from_string,
    calculate_tanimoto as calculate_fingerprint_tanimoto
)

rag_folder = Path(__file__).resolve().parent 

index_file = rag_folder / "fingerprint_index.faiss"
metadata_file = rag_folder / "indexed_metadata.csv"

fingerprint_col = "morgan_fingerprint"
fingerprint_bits = 2048
fingerprint_radius = 2
top_k = 10
multiplier = 5

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=fingerprint_radius,
    fpSize=fingerprint_bits
)

def load_files():
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found\n"
        )
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found\n"
        )
    
    index = faiss.read_index(
        str(index_file)
    )

    metadata_df = pd.read_csv(
        metadata_file,
        dtype={
            fingerprint_col: "string"
        },
        keep_default_na=False
    )

    if fingerprint_col not in metadata_df.columns:
        raise ValueError(
            f"column '{fingerprint_col}' was not found. "
            f"available columns: {list(metadata_df.columns)}"
        )
    
    if len(metadata_df) != index.ntotal:
        raise ValueError(
            f"FAISS index and indexed metadata.csv do not match.\n"
            f"FAISS vectors: {index.ntotal}\n"
            f"Metadata rows: {len(metadata_df)}\n"
            "Run the build index file again"
        )
    

    print(
        "Example stored fingerprint:",
        metadata_df[fingerprint_col].iloc[0]
    )

    print(
        "Stored fingerprint length:",
        len(
            str(
                metadata_df[
                    fingerprint_col
                ].iloc[0]
            )
        )
    )

    return index, metadata_df


def preprocess_smiles(smiles):
    if smiles is None:
        raise ValueError(
            "SMILES string is required"
        )
    
    smiles = str(smiles).strip()

    if not smiles:
        raise ValueError(
            "The SMILES string is empty. "
        )
    
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES - {smiles}"
        )
    
    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    morgan_fingerprint = (
        morgan_generator.GetFingerprint(mol)
    )

    fingerprint_string = (
        morgan_fingerprint.ToBitString()
    )

    fingerprint_array = np.fromiter(
        (
            int(bit)
            for bit in fingerprint_string
        ),
        dtype=np.float32,
        count=fingerprint_bits
    )

    fingerprint_array = fingerprint_array.reshape(
        1,
        fingerprint_bits
    )

    faiss.normalize_L2(
        fingerprint_array
    )

    return(canonical_smiles, morgan_fingerprint, fingerprint_array)


# def fingerprint_from_string(fingerprint):
#     if fingerprint is None or pd.isna(fingerprint):
#         return None
    
#     if (
#         len(fingerprint) == fingerprint_bits
#         and set(fingerprint).issubset({"0", "1"})
#     ):
#         return DataStructs.CreateFromBitString(
#             fingerprint
#         )
    
#     try:
#         fingerprint_list = ast.literal_eval(
#             fingerprint
#         )

#         fingerprint_array = np.asarray(
#             fingerprint_list,
#             dtype=np.int8
#         ).flatten()

#         if len(fingerprint_array) != fingerprint_bits:
#             return None 

#         if not np.all(
#             np.isin(
#                 fingerprint_array,
#                 [0, 1]
#             )
#         ):
            
#             return None

#         fingerprint_string = "".join(
#             str(int(bit))
#             for bit in fingerprint_array
#         )

#         return DataStructs.CreateFromBitString(
#             fingerprint_string
#         )
    
#     except (
#         ValueError,
#         SyntaxError,
#         TypeError
#     ):
        
#         return None 
    
    # fingerprint = str(fingerprint).strip()
    # if len(fingerprint) != fingerprint_bits:
    #     return None
    
    # if set(fingerprint) != fingerprint_bits:
    #     return None 
    
    # if set(fingerprint) - {"0", "1"}:
    #     return None
    
    # return DataStructs.CreateFromBitString(
    #     fingerprint 
    # )

# calculate the tanimoto (heatmap is on the smiliarity.py)

# calculate the tanimoto (heatmap is on the smiliarity.py)

def calculate_tanimoto(
        query_fingerprint,
        database_fingerprint
):  
    database_fingerprint = fingerprint_from_string(
        database_fingerprint 
    )

    if database_fingerprint is None:
        return None 

    # EDIT - use the shared Tanimoto function from similarity.py
    return calculate_fingerprint_tanimoto(
        query_fingerprint,
        database_fingerprint
    )


# retrieval

def retrieve(smiles, k=top_k):
    if not isinstance(k, int):
        raise TypeError(
            "k must be an integer. "
        )
    
    if k <= 0:
        raise ValueError(
            "k must be > 0"
        )
    
    index, metadata_df = load_files()

    (
        canonical_smiles, 
        query_fingerprint, 
        query_vector
    ) = preprocess_smiles(smiles)

    candidate_count = min(
        max(
            k * multiplier,
            k
        ),
        index.ntotal
    )

    faiss_scores, faiss_ids = index.search(
        query_vector,
        candidate_count
    )

    # EDIT - print the FAISS candidate arrays only once
    print(
        "Similar Molecule IDs: ",
        faiss_ids[0]
    )

    # EDIT - print the FAISS candidate arrays only once
    print(
        "Similar Molecule Scores: ",
        faiss_scores[0]
    )

    results = []

    for faiss_score, faiss_id in zip(
        faiss_scores[0],
        faiss_ids[0]
    ):
        if faiss_id < 0:
            continue 

        row = metadata_df.iloc[
            int(faiss_id)
        ].copy()

        tanimoto_similarity = calculate_tanimoto(
            query_fingerprint,
            row[fingerprint_col]
        )

        if tanimoto_similarity is None:
            continue 

        print(
            faiss_id,
            tanimoto_similarity
        )

        result = row.to_dict()

        result["faiss_id"] = int(
            faiss_id
        )

        result["faiss_score"] = float(
            faiss_score
        )

        result["tanimoto_similarity"] = (
            tanimoto_similarity
        )

        results.append(
            result
        )

    if not results:
        return canonical_smiles, pd.DataFrame()
    
    results_df = pd.DataFrame(
        results
    )

    results_df.sort_values(
        by=[
            "tanimoto_similarity",
            "faiss_score"
        ],
        ascending=[
            False,
            False
        ],
        inplace=True
    )

    results_df = results_df.head(
        k
    ).copy()

    results_df.reset_index(
        drop=True,
        inplace=True
    )

    results_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    return canonical_smiles, results_df

def select_display_columns(results_df):
    preferred_columns = [
        "rank",
        "molecule_id",
        # "iupac_name",
        "smiles",
        # "canonical_smiles",
        # "dataset_source",
        # "molecular_formula",
        # "molecular_weight",
        # "log_p",
        "tanimoto_similarity",
        # "faiss_score"
    ]

    display_columns = [
        column 
        for column in preferred_columns
        if column in results_df.columns
    ]

    return results_df[
        display_columns
    ]


def main():
    print()
    print("Molecular R.A.G retreival")
    print()

    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    try:
        canonical_smiles, results_df = retrieve(
            smiles,
            k=top_k
        )

        print()
        print(
            "Canonical SMILES: ", canonical_smiles
        )
        print()

        if results_df.empty:
            print(
                "There are no similar molecules found. "
            )
            return 

        display_df = select_display_columns(
            results_df
        )

        print(
            display_df.to_string(
                index=False
            )
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError
    ) as error:
        print()
        print("Retrieval error: ", error)

if __name__ == "__main__":
    main()