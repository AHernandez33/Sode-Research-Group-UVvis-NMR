import time
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import inchi


from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import re



input_file = "UV_vis_merged_w_source.csv"
output_file = "UV_metadata_updated.csv"

smiles_column ="smiles"
req_delay = 0.25
req_timeout = 20

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


session = requests.Session()
session.headers.update({
    "User-Agent": "UV-vis metadata"
})

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session.mount("https://", HTTPAdapter(max_retries=retry))

def canonical(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


def get_rdkit_metadata(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "canonical_smiles": None,
            "inchikey": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "heavy_atoms": None,
            "aromatic_rings": None,
            "h_bond_donors": None,
            "h_bond_acceptors": None,
            "rotatable_bonds": None,
            "formal_charge": None,
            "morgan_fingerprint": None,
            "maccs_fingerprint": None,
            "log_p": None
        }

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    try:
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        inchikey = None

    morgan_fingerprint = (
        morgan_generator.GetFingerprint(mol).ToBitString()
    )

    maccs_fingerprint = (
        MACCSkeys.GenMACCSKeys(mol).ToBitString()
    )

    log_p = round(Descriptors.MolLogP(mol), 4)

    return {
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "morgan_fingerprint": morgan_fingerprint,
        "maccs_fingerprint": maccs_fingerprint,
        "log_p": log_p,
    }


def get_pubchem_properties(smiles):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/smiles/{requests.utils.quote(smiles, safe='')}/"
        "property/CID,IUPACName/JSON"
    )

    try:
        response = session.get(
            url,
            timeout=req_timeout
        )

        if response.status_code == 404:
            return None, None

        response.raise_for_status()

        properties = (
            response.json()
            .get("PropertyTable", {})
            .get("Properties", [])
        )

        if not properties:
            return None, None

        first = properties[0]

        return first.get("CID"), first.get("IUPACName")

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError
    ) as error:
        print("PubChem property error:", error)
        return None, None


def get_pubchem_synonyms(cid):
    if cid is None:
        return [], None

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/cid/{cid}/synonyms/JSON"
    )

    try:
        response = session.get(
            url,
            timeout=req_timeout
        )

        if response.status_code == 404:
            return [], None

        response.raise_for_status()

        information = (
            response.json()
            .get("InformationList", {})
            .get("Information", [])
        )

        if not information:
            return [], None

        synonyms = information[0].get("Synonym", [])

        cas_pattern = re.compile(
            r"^(?:\d{2,7})-\d{2}-\d$"
        )

        cas_numbers = [
            value
            for value in synonyms
            if cas_pattern.fullmatch(str(value).strip())
        ]

        cas = cas_numbers[0] if cas_numbers else None

        return synonyms, cas

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError
    ) as error:
        print("PubChem synonym error:", error)
        return [], None


df = pd.read_csv(input_file)

if smiles_column not in df.columns:
    raise ValueError(
        f"Column '{smiles_column}' was not found. "
        f"Available columns: {list(df.columns)}"
    )


rows = []
cache = {}

total = len(df)

for index, row in df.iterrows():
    original_smiles = row[smiles_column]
    canonical_smiles = canonical(original_smiles)

    if canonical_smiles is None:
        rows.append({
            "molecule_id": f"MOL{index + 1:05d}",
            "smiles": original_smiles,
            "canonical_smiles": None,
            "cid": None,
            "iupac_name": None,
            "synonyms": None,
            "inchikey": None,
            "cas": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "heavy_atoms": None,
            "aromatic_rings": None,
            "h_bond_donors": None,
            "h_bond_acceptors": None,
            "rotatable_bonds": None,
            "formal_charge": None,
            "morgan_fingerprint": None, 
            "maccs_fingerprint": None,
            "log_p": None,
            "lookup_status": "invalid_smiles"
        })

        print(
            f"{index + 1}/{total} | "
            f"Invalid SMILES: {original_smiles}"
        )

        continue

    if canonical_smiles in cache:
        metadata = cache[canonical_smiles].copy()

        metadata["molecule_id"] = f"MOL{index + 1:05d}"
        metadata["smiles"] = original_smiles
        metadata["lookup_status"] = "duplicate_cached"

        rows.append(metadata)

        print(
            f"{index + 1}/{total} | "
            f"Duplicate | {canonical_smiles}"
        )

        continue

    rdkit_metadata = get_rdkit_metadata(canonical_smiles)

    cid, iupac_name = get_pubchem_properties(
        canonical_smiles
    )

    time.sleep(req_delay)

    synonyms, cas = get_pubchem_synonyms(cid)

    time.sleep(req_delay)

    limited_synonyms = "; ".join(synonyms[:20])

    if cid is None:
        lookup_status = "rdkit_only"
    else:
        lookup_status = "pubchem_found"

    metadata = {
        "molecule_id": f"MOL{index + 1:05d}",
        "smiles": original_smiles,
        "canonical_smiles": rdkit_metadata[
            "canonical_smiles"
        ],
        "cid": cid,
        "iupac_name": iupac_name,
        "synonyms": limited_synonyms or None,
        "inchikey": rdkit_metadata["inchikey"],
        "cas": cas,
        "molecular_formula": rdkit_metadata[
            "molecular_formula"
        ],
        "molecular_weight": rdkit_metadata[
            "molecular_weight"
        ],
        "heavy_atoms": rdkit_metadata["heavy_atoms"],
        "aromatic_rings": rdkit_metadata[
            "aromatic_rings"
        ],
        "h_bond_donors": rdkit_metadata[
            "h_bond_donors"
        ],
        "h_bond_acceptors": rdkit_metadata[
            "h_bond_acceptors"
        ],
        "rotatable_bonds": rdkit_metadata[
            "rotatable_bonds"
        ],
        "formal_charge": rdkit_metadata[
            "formal_charge"
        ],
        "morgan_fingerprint": rdkit_metadata[
            "morgan_fingerprint"
        ],
        "maccs_fingerprint": rdkit_metadata[
            "maccs_fingerprint"
        ],
        "log_p": rdkit_metadata[ 
            "log_p"
        ],
        "lookup_status": lookup_status
    }

    cache[canonical_smiles] = metadata.copy()
    rows.append(metadata)

    print(
        f"{index + 1}/{total} | "
        f"CID: {cid} | "
        f"{iupac_name} | "
        f"{canonical_smiles}"
    )


metadata_df = pd.DataFrame(rows)

metadata_df.to_csv(
    output_file,
    index=False
)

print()
print(metadata_df.head())
print()
print("Rows:", len(metadata_df))
print(
    "PubChem matches:",
    metadata_df["cid"].notna().sum()
)
print(
    "Missing PubChem matches:",
    metadata_df["cid"].isna().sum()
)
print(
    "Invalid SMILES:",
    (
        metadata_df["lookup_status"]
        == "invalid_smiles"
    ).sum()
)
print("Saved:", output_file)