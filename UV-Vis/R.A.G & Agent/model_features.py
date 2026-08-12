from pathlib import Path
import json 
import math


project_folder = Path(
    __file__
).resolve().parent

model_features_output_file = (
    project_folder
    / "latest_model_features.json"
)

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

def clean_value(
        value,
        default=None
):
    if value is None:
        return default 

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default 

    return value 


def clean_list(value):
    if value is None:
        return []

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        if value.strip() == "":
            return []

        return[value]

    return [value]

def clean_bool(
    value,
    default=False
):
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    if isinstance(value, str):
        cleaned_value = value.strip().lower()

        if cleaned_value in [
            "true",
            "yes",
            "1"
        ]:
            return True 

        if cleaned_value in [
            "false",
            "no",
            "0"
        ]: 
            return False

    return default


def get_first(
        dictionary,
        keys,
        default=None
):
    if not isinstance(dictionary, dict):
        return default

    for key in keys:
        if key in dictionary:
            value = dictionary[key]

            if value is not None:
                return value

    return default

def normalize_confidence(confidence):
    if confidence is None:
        return "unknown"

    confidence = str(
        confidence
    ).strip().lower()

    if confidence in [
        "extreme",
        "extremely high",
        "very strong",
        "certain",
        "guaranteed"
    ]:
        return "very high"

    if confidence in [
        "high",
        "strong",
        "good",
        "very high"
    ]:
        return "high"

    if confidence in [
        "moderate",
        "medium",
        "somewhat certain",
        "average",
    ]: 
        return "medium"

    if confidence in [
        "low",
        "poor",
        "weak",
        "very low"
    ]:
        return "low"

    if confidence in [
        "extremely low",
        "uncertain",
        "very poor",
        "very weak"
    ]:
        return "very low"

    return "unknown"


# aromatic atoms 

def count_aromatic_atoms(
    mol
):
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
    )


def count_aromatic_bonds(
    mol
):
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetIsAromatic()
    )

def count_double_bonds(
    mol
):
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE 
    )

def count_triple_bonds(
    mol

):
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE

    )

def count_conjugated_bonds(
    mol
):
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetIsConjugated()
    )

def count_heteroatoms(
    mol
):
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in [
            1,
            6
        ]
    )

def get_element_counts(
    mol
):
    element_counts = {}

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()

        element_counts[symbol] = (
            element_counts.get(
                symbol,
                0
            ) + 1
        )

    return element_counts


def calc_mole_features(
    smiles
):
    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:
        raise ValueError(
            f"Invalid SMILES string: {smiles}"
        )

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    num_atoms = mol.GetNumAtoms()

    num_heavy_atoms = mol.GetNumHeavyAtoms()

    num_bonds = mol.GetNumBonds()

    num_rings = rdMolDescriptors.CalcNumRings(
        mol
    )

    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(
        mol
    )

    num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(
        mol
    )

    num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(
        mol
    )

    num_aromatic_atoms = count_aromatic_atoms(
        mol
    )


    num_aromatic_bonds = count_aromatic_bonds(
        mol
    )

    num_double_bonds = count_double_bonds(
        mol
    )

    num_triple_bonds = count_triple_bonds(
        mol
    )

    num_conjugated_bonds = count_conjugated_bonds(
        mol
    )

    num_heteroatoms = count_heteroatoms(
        mol
    )

    aromatic_fraction = 0.0

    if num_atoms > 0:
        aromatic_fraction = (
            num_aromatic_atoms / num_atoms
        )

    conjugated_fraction = 0.0

    if num_bonds > 0:
        conjugated_fraction = (
            num_conjugated_bonds / num_bonds
        )

    molecular_features = {
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(
            mol
        ),

       "molecular_weight": float(
           Descriptors.MolWt(
               mol
           )
       ),

       "exact_molecular_weight": float(
           Descriptors.ExactMolWt(
               mol
           )
       ),

       "logp": float(
           Descriptors.MolLogP(
               mol
           )
        ),

        "tpsa": float(
            rdMolDescriptors.CalcTPSA(
                mol
            )
        ),

        "num_atoms": int(
            num_atoms
        ),

        "num_heavy_atoms": int(
            num_heavy_atoms  
        ),

        "num_bonds": int(
            num_bonds
        ),

        "num_rings": int(
            num_rings
        ),

        "num_aromatic_rings": int(
            num_aromatic_rings  
        ),

        "num_aliphatic_rings": int(
            num_aliphatic_rings
        ),

        "num_saturated_rings": int(
            num_saturated_rings
        ),

        "num_aromatic_atoms": int(
            num_aromatic_atoms
        ),

        "num_aromatic_bonds": int(
            num_aromatic_bonds
        ),

        "num_double_bonds": int(
            num_double_bonds
        ),

        "num_triple_bonds": int(
            num_triple_bonds  
        ),

        "num_conjugated_bonds": int(
            num_conjugated_bonds
        ),

        "num_heteroatoms": int(
            num_heteroatoms  
        ),

        "num_hydrogen_bond_acceptors": int(
            Lipinski.NumHAcceptors(
                mol
            )
        ),

        "num_rotatable_bonds": int(
            Lipinski.NumRotatableBonds(
                mol
            )
        ),

        "formal_charge": int(
            Chem.GetFormalCharge(
                mol 
            )
        ),

        "aromatic_fraction": float(
            aromatic_fraction 
        ),

        "conjugated_fraction": float(
            conjugated_fraction
        ),

        "has_aromatic_atoms" : (
            num_aromatic_atoms > 0
        ),

        "has_aromatic_rings": (
            num_aromatic_rings > 0
        ),

        "has_conjugation": (
            num_conjugated_bonds > 0
        ),

        "has_double_bonds": (
            num_double_bonds > 0
        ),

        "has_triple_bonds": (
            num_triple_bonds > 0 
        ),

        "element_counts" : get_element_counts(
            mol
        )

    }

    return molecular_features


# extract the retrieval features 

def extract_retrieval_features(
    knowledge_base
):
    if not isinstance(
        knowledge_base,
        dict 
    ):
        return {
            "highest_similarity": 0.0,
            "average_similarity": 0.0,
            "lowest_similarity": 0.0,
            "num_retrieved_molecules": 0,
            "retrieved_molecule_ids": [],
            "retrieved_smiles": [],
            "retrieval_quality": "unknown"
        }

    retrieval_summary = get_first(
        knowledge_base,
        [
            "retrieval_summary",
            "summary"
        ],
        {}
    )

    retrieved_molecules = get_first(
        knowledge_base,
        [
            "retrieved_molecules",
            "neighbors",
            "results"
        ],
        []
    )

    if not isinstance(
        retrieved_molecules,
        list 
    ):
        retrieved_molecules = []

    similarities = []

    molecule_ids = []

    retrieved_smiles = []

    for molecule in retrieved_molecules:
        if not isinstance(
            molecule,
            dict 
        ):
            continue 

        similarity = get_first(
            molecule,
            [
                "tanimoto_similarity",
                "similarity",
                "exact_similarity",
                "score"
            ],
            None 
        )

        if similarity is not None:
            try:
                similarities.append(
                    float(
                        similarity 
                    )
                )

            except (
                TypeError,
                ValueError 
            ):
                pass

        molecule_id = get_first(
            molecule,
            [
                "molecule_id",
                "compound_id",
                "id" 
            ],
            None 
        )

        if molecule_id is not None:
            molecule_ids.append(
                molecule_id 
            )

        molecule_smiles = get_first(
            molecule,
            [
                "smiles",
                "canonical_smiles"
            ],
            None 
        )

        if molecule_smiles is not None:
            retrieved_smiles.append(
                molecule_smiles 
            )

    highest_similarity = get_first(
        retrieval_summary,
        [
            "highest_similarity",
            "highest_tanimoto_similarity",
            "max_similarity"
        ],
        None 
    )

    average_similarity = get_first(
        retrieval_summary,
        [
            "average_similarity",
            "average_tanimoto_similarity",
            "mean_similarity" 
        ],
        None 
    )

    lowest_similarity = get_first(
        retrieval_summary,
        [
            "lowest_similarity",
            "lowest_tanimoto_similarity",
            "min_similarity"
        ],
        None 
    )

    if highest_similarity is None:
        if similarities:
            highest_similarity = max(
                similarities  
            )

        else:
            highest_similarity = 0.0

    if average_similarity is None:
        if similarities:
            average_similarity = (
                sum(
                    similarities
                ) 
                /
                len(
                    similarities
                )
            )

        else:
            average_similarity = 0.0

    if lowest_similarity is None:
        if similarities:
            lowest_similarity = min(
                similarities
            )

        else:
            lowest_similarity = 0.0

    highest_similarity = float(
        highest_similarity
    )

    average_similarity = float(
        average_similarity
    )

    lowest_similarity = float(
        lowest_similarity
    )

    retrieval_quality = classify_retrieval_quality(
        highest_similarity=highest_similarity,
        average_similarity=average_similarity,
        num_retrieved_molecules=len(
            retrieved_molecules 
        )
    )

    retrieval_features = {
        "highest_similarity": highest_similarity,
        "average_similarity": average_similarity,
        "lowest_similarity": lowest_similarity,
        "num_retrieved_molecules": len(
            retrieved_molecules
        ),
        "retrieved_molecule_ids": molecule_ids,
        "retrieved_smiles": retrieved_smiles,
        "retrieval_quality": retrieval_quality
    }

    return retrieval_features 

# classify retrieval quality 

def classify_retrieval_quality(
        highest_similarity,
        average_similarity,
        num_retrieved_molecules
):
    if num_retrieved_molecules == 0:
        return "none"

    # very high would be greater than 0.85
    # high will be between 0.65-0.8499
    # medium will be 0.45-0.6499
    # low will be 0.20-0.4499
    # very low will be <= 0.1999

    # for average similarity 
    # very high will be higher than 0.66
    # high will be 0.50-0.6599
    # med will be 0.30-0.4999
    # low will be 0.10-0.2999
    # very low will be less than 0.10

    if (
        highest_similarity >= 0.85
        and average_similarity >= 0.66
    ):
        return "very high"

    if (
        highest_similarity >= 0.65
        and average_similarity >= 0.50
    ):
        return "high"

    if (
        highest_similarity >= 0.45
        and average_similarity >= 0.30

    ):
        return "medium"

    if (
        highest_similarity >= 0.20
        and average_similarity >= 0.10
    ):
        return "low"

    return "very low"

def extract_llm_features(
    llm_output
):
    if not isinstance(llm_output, dict):
        llm_output = {}

    retrieval_confidence = normalize_confidence(
        get_first(
            llm_output,
            [
                "retrieval_confidence",
                "confidence"
            ],
            "unknown"
        )
    )

    functional_groups = clean_list(
        get_first(
            llm_output,
            [
                "functional_groups",
                "functional_group"
            ],
            []
        )
    )

    chromophores = clean_list(
        get_first(
            llm_output,
            [
                "chromophores",
                "chromophore"
            ],
            []
        )
    )

    auxochromes = clean_list(
        get_first(
            llm_output,
            [
                "auxochromes",
                "auxochrome"
            ],
            []
        )
    )

    warnings = clean_list(
        get_first(
            llm_output,
            [
                "warnings",
                "warning"
            ],
            []
        )
    )

    recommended_models = clean_list(
        get_first(
            llm_output,
            [
                "recommended_models",
                "model_recommendation"
            ],
            []
        )
    )


    descriptor_importance = clean_list(
        get_first(
            llm_output,
            [
                "descriptor_importance",
                "important_descriptors"
            ],
            []
        )
    )

    
    llm_features = {
        "retrieval_confidence": retrieval_confidence,
        "functional_groups": functional_groups,
        "chromophores": chromophores,
        "auxochromes": auxochromes,
        "llm_aromatic": clean_bool(
            get_first(
                llm_output,
                [
                    "aromatic",
                    "is_aromatic"
                ],
                False
            )
        ),
        "llm_conjugated": clean_bool(
            get_first(
                llm_output,
                [
                    "conjugated",
                    "is_conjugated"
                ],
                False
            )
        ),
        "reasoning": clean_value(
            get_first(
                llm_output,
                [
                    "reasoning",
                    "structural_analysis"
                ],
                ""
            ),
            ""
        ),
        "retrieval_analysis": clean_value(
            get_first(
                llm_output,
                [
                    "retrieval_analysis"
                ],
                ""
            ),
            ""
        ),
        "routing_notes": clean_value(
            get_first(
                llm_output,
                [
                    "routing_notes"
                ],
                ""
            ),
            ""
        ),
        "warnings": warnings,
        "recommended_models": recommended_models,
        "descriptor_importance": descriptor_importance,
        "uncertainty_notes": clean_list(
            get_first(
                llm_output,
                [
                    "uncertainty_notes"
                ],
                []
            )
        )
    }

    return llm_features

def estimate_uv_vis_complexity(
        molecular_features,
        llm_features
):
    complexity_score = 0

    if molecular_features[
        "num_aromatic_rings"
    ] > 0:
        complexity_score += 2

    if molecular_features[
        "num_conjugated_bonds"
    ] >= 2:
        complexity_score += 2

    if molecular_features[
        "num_conjugated_bonds"
    ] >= 5:
        complexity_score += 1

    if molecular_features[
        "num_heteroatoms"
    ] >= 2:
        complexity_score += 1

    if molecular_features[
        "num_rings"
    ] >= 2:
        complexity_score += 1

    if molecular_features[
        "molecular_weight"
    ] >= 200:
        complexity_score += 1

    if len(llm_features["chromophores"]) > 0:
        complexity_score += 2

    if len(llm_features["auxochromes"]) > 0:
        complexity_score += 1

    if complexity_score <= 2:
        complexity_level = "very low"

    elif complexity_score <= 3:
        complexity_level = "low"

    elif complexity_score <= 5:
        complexity_level = "medium"

    elif complexity_score <= 7:
        complexity_level = "high"

    else:
        complexity_level = "very high"

    return {
        "complexity_score": complexity_score,
        "complexity_level": complexity_level
    }

# estimate data support 

def estimate_data_support(retrieval_features, llm_features):
    highest_similarity = retrieval_features[
        "highest_similarity"
    ]

    average_similarity = retrieval_features[
        "average_similarity"
    ]

    retrieval_confidence = llm_features[
        "retrieval_confidence"
    ]

    support_score = 0


    if highest_similarity >= 0.80:
        support_score += 5

    elif highest_similarity >= 0.60:
        support_score += 4

    elif highest_similarity >= 0.40:
        support_score += 3

    elif highest_similarity >= 0.20:
        support_score += 2

    elif highest_similarity >= 0.0:
        support_score += 1

    if average_similarity >= 0.66:
        support_score += 4

    elif average_similarity >= 0.45:
        support_score += 3

    elif average_similarity >= 0.30:
        support_score += 2

    elif average_similarity >= 0.15:
        support_score += 1

    if retrieval_confidence == "very high":
        support_score += 4

    elif retrieval_confidence == "high":
        support_score += 3

    elif retrieval_confidence == "medium":
        support_score += 2

    elif retrieval_confidence == "low":
        support_score += 1

    if support_score >= 10:
        support_level = "very high"

    elif support_score >= 7:
        support_level = "high"

    elif support_score >= 5:
        support_level = "medium"

    elif support_score >= 3:
        support_level = "low"

    elif support_score >= 1:
        support_level = "very low"

    else:
        support_level = "none"

    return {
        "data_support_score": support_score,
        "data_support_level": support_level
    }

def build_routing_flags(molecular_features, retrieval_features, llm_features, complexity_features, support_features):
    routing_flags = {
        "small_molecule": (
            molecular_features[
                "num_heavy_atoms"
            ] <= 35
        ),
        "large_molecule": (
            molecular_features[
                "num_heavy_atoms"
            ] >= 85
        ),

        "aromatic_system": (
            molecular_features[
                "has_aromatic_atoms"
            ]
            or
            llm_features[
                "llm_aromatic"
            ]
        ),

        "conjugated_system": (
            molecular_features[
                "has_conjugation"
            ]
            or
            llm_features[
                "llm_conjugated"
            ]
        ),

        "multiple_rings": (
            molecular_features[
                "num_rings"
            ] >= 2
        ),

        "flexible_molecule": (
            molecular_features[
                "num_rotatable_bonds"
            ] >= 5
        ),

        "high_molecular_weight": (
            molecular_features[
                "molecular_weight"
            ] >= 600
        ),

        "contains_uncommon_elements": any(
            element not in [
                "C",
                "H",
                "N",
                "O",
                "F",
                "P",
                "S",
                "Cl",
                "Br",
                "I"
            ]

            for element in molecular_features[
                "element_counts"
            ]
        ),

        "low_retrieval_support": (
            support_features[
                "data_support_level"
            ] in [
                "none",
                "low"
            ]
        ),

        "high_retrieval_support": (
            support_features[
                "data_support_level"
            ] == "high"
        ),

        "requires_uncertainty_warning": (
            retrieval_features[
                "highest_similarity"
            ] < 0.33
        )
    }

    return routing_flags


def build_model_features(
        smiles,
        knowledge_base=None,
        llm_output=None 
):
    if knowledge_base is None:
        knowledge_base = {}

    if llm_output is None:
        llm_output = {}

    molecular_features = calc_mole_features(
        smiles 
    )

    retrieval_features = extract_retrieval_features(
        knowledge_base
    )

    llm_features = extract_llm_features(
        llm_output 
    )

    complexity_features = estimate_uv_vis_complexity(
        molecular_features=molecular_features,
        llm_features=llm_features
    )

    support_features = estimate_data_support(
        retrieval_features=retrieval_features,
        llm_features=llm_features
    )

    routing_flags = build_routing_flags(
        molecular_features=molecular_features,
        retrieval_features=retrieval_features,
        llm_features=llm_features,
        complexity_features=complexity_features,
        support_features=support_features
    )

    model_features = {
        "smiles": molecular_features[
            "canonical_smiles"
        ],

        "molecular_features": molecular_features,
        "retrieval_features": retrieval_features,
        "llm_features": llm_features,
        "complexity_features": complexity_features,
        "support_features": support_features,
        "routing_flags": routing_flags 
    }

    return model_features


def save_model_features(
        model_features,
        file=None
):
    if file is None:
        file = model_features_output_file

    file = Path(
        file
    )

    if not file.is_absolute():
        file = (
            project_folder
            / file
        )

    file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file,
        "w",
        encoding="utf-8"
    ) as output_file:
        json.dump(
            model_features,
            output_file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "Model features file saved:"
    )

    print(
        file
    )

    return file

def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    example_knowledge_base = {
        "retrieval_summary": {
            "highest_similarity": 0.2727272727,
            "average_similarity": 0.135,
            "lowest_similarity": 0.09
        },
        "retrieved_molecules": [
            {
                "molecule_id": "MOL01055",
                "smiles": "CC[Se][Se]CC",
                "tanimoto_similarity": 0.2727272727
            },
            {
                "molecule_id": "MOL01068",
                "smiles": "CCOCC",
                "tanimoto_similarity": 0.2666666667
            }
        ]
    }

    example_llm_output = {
        "query_smiles": smiles,
        "retrieval_confidence": "low",
        "reasoning": (
            "The query molecule is small and contains "
            "an alcohol functional group."
        ),
        "functional_groups": [
            "alcohol"
        ],
        "chromophores": [],
        "auxochromes": [
            "hydroxyl group"
        ],
        "aromatic": False,
        "conjugated": False,
        "retrieval_analysis": (
            "The retrieved molecules have low structural similarity."
        ),
        "routing_notes": (
            "The prediction should rely more strongly on "
            "learned molecular representations."
        ),
        "warnings": [
            "Low similarity to retrieved molecules."
        ]
    }

    model_features = build_model_features(
        smiles=smiles,
        knowledge_base=example_knowledge_base,
        llm_output=example_llm_output
    )

    save_model_features(
        model_features
    )

    print()

    print(
        json.dumps(
            model_features,
            indent=4
        )
    )


if __name__ == "__main__":
    main()