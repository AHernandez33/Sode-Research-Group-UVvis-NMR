from pathlib import Path
import json
import numpy as np
import pandas as pd 

from retrieve import retrieve 

rag_folder = Path(__file__).resolve().parent
knowledge_file = rag_folder / "latest_knowledge_base.json"

def clean_value(value):
    if value is None:
        return None 
    
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and np.isnan(value):
        return None
    
    if pd.isna(value):
        return None
    
    return value

def get_first(
        row,
        possible_cols,
        default=None
):
    for column in possible_cols:
        if column in row.index:
            value = clean_value(row[column])

            if value is not None and value != "":
                return value 
            
    return default 


def create_retrved_mol(
    row,
    default_rank,
):
    molecule = {
        "rank": get_first(
            row,
            [
                "rank",
                "retrieval_rank"
            ],
            default_rank
        ),

        "molecule_id": get_first(
            row,
            [
                "molecule_id",
                "compound_id",
                "id"
            ]
        ),

        "name": get_first(
            row,
            [
                "name",
                "compound_name", 
                "molecule_name"
            ]
        ),

        "smiles": get_first(
            row,
            [
                "smiles",
                "canonical_smiles",
                "SMILES"
            ]
        ),

        "tanimoto_similarity": get_first(
            row,
            [
                "tanimoto_similarity",
                "tanimoto_score",
                "similarity",
                "similarity_score"
            ]
        ),

        "faiss_score": get_first(
            row,
            [
                "faiss-score",
                "faiss_similarity",
                "index_score"
            ]
        ),

        "logp": get_first(
            row,
            [
                "logp",
                "LogP",
                "mol_logp"
            ]
        ),

        "molecular_weight": get_first(
            row, 
            [
                "molecular_weight",
                "molecular_weight_g_mol",
                "mol_weight",
                "MolWt"
            ]
        ),

        "molecular_formula": get_first(
            row,
            [
                "molecular_formula",
                "formula"
            ]
        ),

        "wavelength": get_first(
            row,
            [
                "wavelength",
                "wavelengths",
                "x_wavelength",
                "x"
            ]
        ),

        "absorbance": get_first(
            row,
            [
                "absorbance",
                "absorbance_values",
                "y_absorbance",
                "y"
            ]
        ),

        "source": get_first(
            row,
            [
                "source",
                "dataset_source"
            ]
        )
    }

    return molecule 

def calc_rvl_summary(
        retrieved_molecules
):
    similarities = []

    for molecule in retrieved_molecules:
        similarity = molecule.get(
            "tanimoto_similarity"
        )

        if similarity is None:
            continue

        try:
            similarities.append(
                float(similarity)
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    if similarities:
        highest_similarity = max(
            similarities
        )

        lowest_similarity = min(
            similarities
        )

        avg_similarity = float(
            np.mean(similarities)
        )

    else:
        highest_similarity = None
        lowest_similarity = None
        avg_similarity = None

    return {
        "number_retrieved": len(
            retrieved_molecules
        ),

        "highest_similarity": highest_similarity,

        "lowest_similarity": lowest_similarity,

        "average_similarity": avg_similarity
    }

def format_number(
        value,
        decimal_places=6
):
    
    if value is None:
        return "Unknown"
    
    try:
        return f"{float(value):.{decimal_places}f}"
    
    except(
        TypeError,
        ValueError
    ): 
        return str(value)
    
def build_llm_context(
    query_smiles,
    canonical_smiles,
    retrieval_summary,
    retrieved_mols  
):
    context_lines = [
        "UV-Vis molecular retrieval knowledge base",
        "",
        "Query Molecule: ",
        f"- Input the SMILES: {query_smiles}",
        f"- Canonical SMILES: {canonical_smiles}",
        "",
        "Retrieval Summary: ",
        (
            "- Number of retrieved molecules: "
            f"{retrieval_summary['number_retrieved']}"
        ),
        (
            "- Highest Tanimoto similarity: "
            f"{format_number(retrieval_summary['highest_similarity'])}"
        ),
        (
            "- Average Tanimoto similarity: "
            f"{format_number(retrieval_summary['average_similarity'])}"
        ),
        "",
        "Retrieved molecules: "

    ]

    if not retrieved_mols:
        context_lines.append(
            "- No molecules were retrieved. "
        )

        return "\n".join(
            context_lines
        )
    
    for molecule in retrieved_mols:
        context_lines.extend(
            [
                "",
                f"Neighbor rank {molecule['rank']}:",
                (
                    "- Molecule ID: "
                    f"{molecule.get('molecule_id') or 'Unknown'}"
                ),
                (
                    "- Name: "
                    f"{molecule.get('name') or 'Unknown'}"
                ),
                (
                    "- SMILES: "
                    f"{molecule.get('smiles') or 'Unknown'}"
                ),
                (
                    "- Tanimoto Similarity: "
                    f"{format_number(molecule.get('tanimoto_similarity'))}"
                ),
                (
                    "- LogP: "
                    f"{format_number(molecule.get('logp'), 3)}"
                ),
                (
                    "- Molecular Weight: "
                    f"{format_number(molecule.get('molecular_weight'), 3)}"
                ),
                (
                    "- Molecular Formula: "
                    f"{molecule.get('molecular_formula') or 'Unknown'}"
                ),
                (
                    "- Dataset Source: "
                    f"{molecule.get('source') or 'Unknown'}"
                )
            ]
        )

        wavelength = molecule.get(
            "wavelength"
        )

        absorbance = molecule.get(
            "absorbance"
        )

        if wavelength is not None:
            context_lines.append(
                f"- Wavelength data: {wavelength}"
            )

        if absorbance is not None:
            context_lines.append(
                f"- Absorbance data: {absorbance}"
            )

    context_lines.extend(
        [
            "",
            "Instructions for later reasoning: ",
            (
                "- Take the retrieving molecules as supporting evidence, "
                "not as exact predictions for the molecule that is being queried."
            ),
            (
                "- Give more weight and consideration to the neighbor molecules with higher Tanimoto similarity"
            ),
            (
                "- Consider differences in the molecular structure,"
                "conjugation, aromacity, functional groups, and LogP."
            ),
            (
                "- Do NOT claim a UV-Vis peak unless it is supported by "
                "retrieved data or a prediction model."
            )
        ]
    )

    return "\n".join(
        context_lines
    )


def make_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }
    
    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]
    
    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]
    
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    if isinstance(value, np.generic):
        return value.item()
    
    if isinstance(value, pd.Series):
        return make_json_safe(
            value.to_dict()
        )
    
    if isinstance(value, pd.DataFrame):
        return make_json_safe(
            value.to_dict(
                orient="records"
            )
        )
    
    if isinstance(value, float) and np.isnan(value):
        return None 
    
    return value

def save_knowledge_base(
        knowledge_base,
        output_file=knowledge_file
):
    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_knowledge_base = make_json_safe(
        knowledge_base
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            safe_knowledge_base,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "Knowledge base saved: ",
        output_file
    )


def build_knowledge_base(
    smiles,
    top_k=10,
    save=False,
    output_file=knowledge_file
):
    if smiles is None:
        raise ValueError(
            "SMILES cannot be None"
        )
    
    smiles = str(
        smiles
    ).strip()

    if not smiles:
        raise ValueError(
            "SMILES cannot be empty"
        )
    
    canonical_smiles, results_df = retrieve(
        smiles,
        k=top_k
    )

    if results_df is None:
        results_df = pd.DataFrame()

    if not isinstance(
        results_df,
        pd.DataFrame
    ):
        results_df = pd.DataFrame(
            results_df
        )

    retrieved_mols = []

    for index, row in results_df.iterrows():
        molecule = create_retrved_mol(
            row=row,
            default_rank=index + 1
        )

        retrieved_mols.append(
            molecule
        )

    retrieval_summary = calc_rvl_summary(
        retrieved_mols
    )

    llm_context = build_llm_context(
        query_smiles=smiles,
        canonical_smiles=canonical_smiles,
        retrieval_summary=retrieval_summary,
        retrieved_mols=retrieved_mols
    )

    knowledge_base = {
        "query": {
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles
        },

        "retrieval_summary": retrieval_summary,

        "retrieved_molecules": retrieved_mols,

        "llm_context": llm_context
    }

    if save:
        save_knowledge_base(
            knowledge_base=knowledge_base,
            output_file=output_file
        )

    return knowledge_base

def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    knowledge_base = build_knowledge_base(
        smiles=smiles,
        top_k=10,
        save=True
    )

    print()
    print(
        knowledge_base["llm_context"]
    )


if __name__ == "__main__":
    main()