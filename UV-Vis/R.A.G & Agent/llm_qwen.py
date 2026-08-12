# Gemma 3:4b
import json
import ollama

from knowledge_base import build_knowledge_base

model_name = "gemma3:4b"

prompt = """
You are an expert computational chemist specializing in UV-Vis spectroscopy.

You are NOT predicting the UV-Vis spectrum. Your job is ONLY to
analyze the retrieved molecular evidence.

You MUST:
- Analyze the query molecule.
- Analyze the retrieved neighboring molecules.
- Determine whether retrieval confidence is:
    - high
    - medium
    - low

- Identify:
    - functional groups
    - aromaticity
    - conjugation
    - likely chromophores
    - possible auxochromes

- Explain whether the retrieved molecules are chemically relevant.

- Suggest what information a prediction model should pay attention to.

Return ONLY valid JSON.

Use this exact schema:

{
    "query_smiles": "",
    "retrieval_confidence": "",
    "reasoning": "",
    "functional_groups": [],
    "chromophores": [],
    "auxochromes": [],
    "aromatic": false,
    "conjugated": false,
    "retrieval_analysis": "",
    "routing_notes": "",
    "warnings": []
}

Do NOT include markdown.

Do NOT include explanations outside the JSON.
"""


def ask_llm(
    knowledge_base
):
    response = ollama.chat(
        model=model_name,
        format="json",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": knowledge_base[
                    "llm_context"
                ]
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ]

    return json.loads(
        answer
    )


def analyze_smiles(
        smiles,
        top_k=10
):
    knowledge_base = build_knowledge_base(
        smiles=smiles,
        top_k=top_k
    )

    llm_output = ask_llm(
        knowledge_base
    )

    return {
        "knowledge_base": knowledge_base,
        "llm_analysis": llm_output
    }


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    results = analyze_smiles(
        smiles
    )

    print()

    print(
        json.dumps(
            results[
                "llm_analysis"
            ],
            indent=4
        )
    )


if __name__ == "__main__":
    main()


# # LLama 3 2:3b

# import json
# from pathlib import Path

# import ollama

# from knowledge_base import build_knowledge_base


# MODEL_NAME = "llama3.2:3b"

# PROJECT_FOLDER = Path(
#     __file__
# ).resolve().parent

# OUTPUT_FILE = (
#     PROJECT_FOLDER
#     / "latest_llama_analysis.json"
# )


# SYSTEM_PROMPT = """
# You are an expert computational chemist specializing in UV-Vis spectroscopy.

# You are NOT predicting the UV-Vis spectrum. Your job is ONLY to analyze
# the retrieved molecular evidence.

# You MUST:
# - Analyze the query molecule.
# - Analyze the retrieved neighboring molecules.
# - Determine whether retrieval confidence is:
#   - high
#   - medium
#   - low
# - Identify:
#   - functional groups
#   - aromaticity
#   - conjugation
#   - likely chromophores
#   - possible auxochromes
# - Explain whether the retrieved molecules are chemically relevant.
# - Suggest what information a prediction model should pay attention to.
# - Report limitations or warnings when the retrieved evidence is weak,
#   inconsistent, incomplete, or chemically dissimilar.

# Return ONLY valid JSON.

# Use this exact schema:
# {
#     "query_smiles": "",
#     "retrieval_confidence": "",
#     "reasoning": "",
#     "functional_groups": [],
#     "chromophores": [],
#     "auxochromes": [],
#     "aromatic": false,
#     "conjugated": false,
#     "retrieval_analysis": "",
#     "routing_notes": "",
#     "warnings": [],
#     "recommended_models": [],
#     "descriptor_importance": [],
#     "uncertainty_notes": []
# }

# Do not include markdown.
# Do not include explanations outside the JSON.
# """


# def validate_llm_output(
#         output
# ):
#     if not isinstance(
#         output,
#         dict
#     ):
#         raise TypeError(
#             "The LLM response must be a JSON object."
#         )

#     required_fields = {
#         "query_smiles": "",
#         "retrieval_confidence": "unknown",
#         "reasoning": "",
#         "functional_groups": [],
#         "chromophores": [],
#         "auxochromes": [],
#         "aromatic": False,
#         "conjugated": False,
#         "retrieval_analysis": "",
#         "routing_notes": "",
#         "warnings": [],
#         "recommended_models": [],
#         "descriptor_importance": [],
#         "uncertainty_notes": []
#     }

#     validated = dict(
#         output
#     )

#     for key, default_value in required_fields.items():
#         if key not in validated:
#             validated[
#                 key
#             ] = default_value

#     confidence = str(
#         validated.get(
#             "retrieval_confidence",
#             "unknown"
#         )
#     ).strip().lower()

#     if confidence not in {
#         "high",
#         "medium",
#         "low"
#     }:
#         confidence = "unknown"

#     validated[
#         "retrieval_confidence"
#     ] = confidence

#     validated[
#         "aromatic"
#     ] = bool(
#         validated.get(
#             "aromatic",
#             False
#         )
#     )

#     validated[
#         "conjugated"
#     ] = bool(
#         validated.get(
#             "conjugated",
#             False
#         )
#     )

#     list_fields = [
#         "functional_groups",
#         "chromophores",
#         "auxochromes",
#         "warnings",
#         "recommended_models",
#         "descriptor_importance",
#         "uncertainty_notes"
#     ]

#     for key in list_fields:
#         value = validated.get(
#             key,
#             []
#         )

#         if value is None:
#             value = []

#         elif not isinstance(
#             value,
#             list
#         ):
#             value = [
#                 value
#             ]

#         validated[
#             key
#         ] = value

#     return validated


# def ask_llm(
#         knowledge_base
# ):
#     if not isinstance(
#         knowledge_base,
#         dict
#     ):
#         raise TypeError(
#             "knowledge_base must be a dictionary."
#         )

#     llm_context = knowledge_base.get(
#         "llm_context"
#     )

#     if not llm_context:
#         raise KeyError(
#             "knowledge_base does not contain a valid llm_context."
#         )

#     response = ollama.chat(
#         model=MODEL_NAME,
#         format="json",
#         messages=[
#             {
#                 "role": "system",
#                 "content": SYSTEM_PROMPT
#             },
#             {
#                 "role": "user",
#                 "content": str(
#                     llm_context
#                 )
#             }
#         ],
#         options={
#             "temperature": 0.1,
#             "seed": 42
#         }
#     )

#     answer = response[
#         "message"
#     ][
#         "content"
#     ]

#     parsed_answer = json.loads(
#         answer
#     )

#     return validate_llm_output(
#         parsed_answer
#     )


# def analyze_smiles(
#         smiles,
#         top_k=10,
#         save=True
# ):
#     smiles = str(
#         smiles
#     ).strip()

#     if not smiles:
#         raise ValueError(
#             "SMILES cannot be empty."
#         )

#     knowledge_base = build_knowledge_base(
#         smiles=smiles,
#         top_k=top_k,
#         save=True
#     )

#     llm_output = ask_llm(
#         knowledge_base
#     )

#     results = {
#         "model_name": MODEL_NAME,
#         "input_smiles": smiles,
#         "knowledge_base": knowledge_base,
#         "llm_analysis": llm_output
#     }

#     if save:
#         with open(
#             OUTPUT_FILE,
#             "w",
#             encoding="utf-8"
#         ) as file:
#             json.dump(
#                 results,
#                 file,
#                 indent=4,
#                 ensure_ascii=False
#             )

#         print(
#             "LLM analysis saved:",
#             OUTPUT_FILE
#         )

#     return results


# def main():
#     smiles = input(
#         "Enter a SMILES string: "
#     ).strip()

#     results = analyze_smiles(
#         smiles=smiles,
#         top_k=10,
#         save=True
#     )

#     print()
#     print(
#         json.dumps(
#             results[
#                 "llm_analysis"
#             ],
#             indent=4,
#             ensure_ascii=False
#         )
#     )


# if __name__ == "__main__":
#     main()


# qwen3 1.7
# import json 
# import ollama 

# from knowledge_base import build_knowledge_base

# model_name = "qwen3:1.7b"

# prompt = """
# You are an expert computational chemist specializing in UV-Vis spectroscopy.

# You are NOT predicting the UV-Vis spectrum. Your job is ONLY to
# analyze the retrieved molecular evidence. 

# You MUST: 
# - Analyze the query molecule.
# - Analyze the retrieved neighboring molecules.
# - Determine whether retrieval confidence is:
#     - high
#     - medium
#     - low

# - Identify:
#     - functional groups
#     - aromaticity
#     - conjugation
#     - likely chromophores
#     - possible auxochromes

# - Explain whether the retrieved molecules are chemically revelant.

# - Suggest what information a prediction model should pay attention to.

# Return ONLY valid JSON.

# Use this exact schema:
# {
#     "query_smiles": "",
#     "retrieval_confidence": "",
#     "reasoning": "",
#     "functional_groups": [],
#     "chromophores": [],
#     "auxochromes": []
#     "aromatic": false,
#     "conjugated": false,
#     retrieval_analysis": "",
#     "routing_notes": "",
#     "warnings": []
# }

# Do NOT include markdown

# Do not include explanations outside the JSON


# """


# def ask_llm(
#     knowledge_base
# ):
#     response = ollama.chat(
#         model=model_name,
#         format="json",
#         messages=[
#             {
#                 "role": "system",
#                 "content": prompt
#             },
#             {
#                 "role": "user",
#                 "content": knowledge_base[
#                     "llm_context"
#                 ]
#             }
#         ]
#     )

#     answer = response[
#         "message"
#     ][
#         "content"
#     ]

#     return json.loads(
#         answer
#     )

# def analyze_smiles(
#         smiles, 
#         top_k=10
# ):
#     knowledge_base = build_knowledge_base(
#         smiles=smiles,
#         top_k=top_k
#     )

#     llm_output = ask_llm(
#         knowledge_base
#     )

#     return{
#         "knowledge_base": knowledge_base,
#         "llm_analysis": llm_output
#     }


# def main():
#     smiles = input(
#         "Enter a SMILES string: "
#     ).strip()

#     results = analyze_smiles(
#         smiles
#     )

#     print()

#     print(
#         json.dumps(
#             results[
#                 "llm_analysis"
#             ],
#             indent=4
#         )
#     )


# if __name__ == "__main__":
#     main()