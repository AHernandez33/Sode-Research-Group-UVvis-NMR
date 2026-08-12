import json
import time 
import pandas as pd
import os

from dotenv import load_dotenv
# from openai import OpenAI
from google import genai 

file1 = "UV_metadata.csv"
file = "UV_metadata_enhanced_gemini.csv"

num = 10
req_delay = 5
load_dotenv()
client = genai.Client(
    api_key=os.getenv("api_key")
) 

# if not os.getenv("api_key"):
#     raise ValueError(
#         "API key environment variable was not found"
#     )

metadata_schema = {
    "type": "object",
    "properties": {
        "solvent": {
            "type": ["string", "null"]
        },
        "temperature": {
            "type": ["number", "null"]
        },
        "pH": {
            "type": ["number", "null"]
        },
        "concentration": {
            "type": ["number", "null"]
        },
       "conc_units": {
           "type": ["string", "null"]
        },
        "path_length": {
            "type": ["number", "null"]
            # this is the path length typically from a UV-vis spectrometer
            # This is in cm
            # also temperature is in C 

        },
        "source_database": {
            "type": ["string", "null"]
        },
        "source_url": {
            "type": ["string", "null"]

        },

        "doi": {
            "type": ["string", "null"]
        },
        "reference": {
            "type": ["string", "null"]
        },
        "metadata_status": {
            "type": "string",
            "enum": [
                "verified_exact_record",
                "verified_source_reference",
                "likely_match",
                "compound_match",
                "conflicts",
                "not_found"
            ]
        },
        "metadata_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "evidence": {
            "type": ["string", "null"]

        },
        "notes": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "solvent",
        "temperature",
        "pH",
        "concentration",
        "conc_units",
        "path_length",
        "source_database",
        "source_url",
        "doi",
        "reference",
        "metadata_status",
        "metadata_confidence",
        "evidence",
        "notes",
    ],
    "additionalProperties": False
}

def clean_value(value):
    if pd.isna(value):
        return None
    
    return value 

def search_metadata(row):
    molecule_data = {
        "molecule_id": clean_value(row.get("molecule_id")),
        "smiles": clean_value(row.get("smiles")),
        "canonical_smiles": clean_value(
            row.get("canonical_smiles")
        ),
        "cid": clean_value(row.get("cid")),
        "iupac_name": clean_value(row.get("iupac_name")),
        "synonyms": clean_value(row.get("synonyms")),
        "inchikey": clean_value(row.get("inchikey")),
        "cas": clean_value(row.get("cas")),
        "molecular_formula": clean_value(row.get("molecular_formula"))
    }

    prompt = f"""

You are a scientific metadata extraction agent. 
Use only information supported by trusted/legitimate/credible revelant sources. 
NEVER fabricate experimental conditions. 
    
Find the metadata for experimental UV-Vis spectra for this compound:
{json.dumps(molecule_data, indent=2)}

Search credible & trusted sources such as:

- Original database record that is linked to the UV-Vis spectrum.json
- NIST Chemistry WebBook.
- Spectral records from PubChem databases
- Peer-reviewed journal articles (example: ACS, NIH, etc.)
- Original publication cited by a spectral database.json

This is what's going to be extracted:
- Solvent 
- Temperature (be sure to convert it in Celsius if it is not converted yet)
- pH 
- concentration and the unit in concentration 
- The optical path length in centimeters
- The source database
- The source URL link 
- The DOI
- Complete reference 


Important rules: 

- Do NOT guess or infer missing experimental conditions. 
- Return null when a value is not EXPLICITLY stated. 
- A paper about the same compound does not prove that its experimental conditions produced the spectrum in my dataset.
- Use verified_source_reference when conditions come from the original publication linked to th at exact spectrum.  
- Use likely_match only when the match is strong but not exact. 
- Use compound_match when only geenral literature about the compound is found. 
- Use not_found when there is no reliable UV-Vis conditions are found. 
- The metadata_confidence must be between 0 and 1.  
- Include evidence supporting and justifying any non-null condition(s). 
"""
    
    # max_retries = 5
    # for attempt in range(max_retries):
    #     try:
    interaction = client.interactions.create(
        model = "gemini-3.1-flash-lite",
        input=prompt,
        tools=[
            {
                "type": "google_search"
            },
            {
                "type": "url_context"
            }

        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": metadata_schema
        
        },
        generation_config={
            "thinking_level": "low"
        }
    )
    # response = client.responses.create(
    #     model="gpt-5-mini",
    #     tools=[ 
    #         {
    #            "type": "web_search" 
    #         }
    #     ],
    #     input=[
    #         {
    #             "role": "system",
    #             "content": (
    #                 # "You are a scientific metadata extraction agent. "
    #                 # "Use only information supported by trusted/legitimate/credible revelant sources. "
    #                 # "NEVER fabricate experimental conditions. "
    #             )
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     text={
    #         "format": {
    #             "type": "json_schema",
    #             "name": "uv_vis_metadata",
    #             "strict": True,
    #             "schema": metadata_schema
    #         }
    #     }
    # )
    return json.loads(interaction.output_text)

df = pd.read_csv(file1)
req_columns = [
    "smiles",
    "canonical_smiles",
    "cid",
    "iupac_name",
    "inchikey"
]

missing_columns = [
    column
    for column in req_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        f"Available columns: {list(df.columns)}"
    )

rows = []

limit = min(num, len(df))
for index in range(limit):
    row = df.iloc[index]
    print()
    print(
        f"Searching {index + 1}/{limit}: "
        f"{row.get('iupac_name')}"
    )

    try:
        result = search_metadata(row)
        combined = {
            "molecule_id": clean_value(
                row.get("molecule_id")
            ),
            "smiles": clean_value(row.get("smiles")),
            "canonical_smiles": clean_value(
                row.get("canonical_smiles")
            ),
            "cid": clean_value(row.get("cid")),
            "iupac_name": clean_value(
                row.get("iupac_name")
            ),
            "inchikey": clean_value(
                row.get("inchikey")
            ),
            "cas": clean_value(row.get("cas")),
            **result
        }

        rows.append(combined)
        print(
            "Status: ",
            result["metadata_status"]
        )

        print(
            "Confidence: ",
            result["metadata_confidence"]
        )

    except Exception as error:
        print("LLM error on metadata: ", error)

        rows.append({
            "molecule_id": clean_value(
                row.get("molecule_id")
            ),
            "smiles": clean_value(row.get("smiles")),
            "canonical_smiles": clean_value(row.get("canonical_smiles")),
            "cid": clean_value(row.get("cid")),
            "iupac_name": clean_value(
                row.get("iupac_name")
            ),
            "inchikey": clean_value(
                row.get("inchikey")
            ),
            "cas": clean_value(row.get("cas")),
            "solvent": None,
            "temperature": None, 
            "pH": None,
            "concentration": None,
            "conc_units": None,
            "path_length": None,
            "source_database": None, 
            "source_url": None,
            "doi": None, 
            "reference": None,
            "metadata_status": "not_found",
            "metadata_confidence": 0.0,
            "evidence": None,
            "notes": str(error)
        })

    pd.DataFrame(rows).to_csv(
        file,
        index=False
    )

    time.sleep(req_delay)

metadata_df = pd.DataFrame(rows)

print()
print("Saved: ", file)