# 932 molecules - 268 more to make it 1.2k
# make everything into a CSV 


# 932 molecules - 268 more to make it 1.2k
# make everything into a CSV 


import re
import json
import time 
import pandas as pd
import requests
from bs4 import BeautifulSoup
from jcamp import jcamp_read
from rdkit import Chem 
import io
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


url = "https://webbook.nist.gov/cgi/cbook.cgi"
file = "uv_vis_400_molecules.csv"
num = 400

wl_min = 220
wl_max = 400
req_delay = 0.5
req_timeout = 20

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)

def get_response(url, params=None):
    try:
        response = session.get(
            url,
            params=params,
            timeout=req_timeout 
        )

        response.raise_for_status()
        time.sleep(req_delay)

        return response
    
    except requests.RequestException as error:
        print("Request error: ", error)
        return None 
    

# def get_ids():
#     nist_ids = set()

#     characters = "abcdefghijklmnopqrstuvwxyz"
#     for character in characters:
#         print("Searching: ", character)

#         params = {
#             "Name": character + "*",
#             "Units": "SI",
#             "Mask": "400"
#         }

#         response = get_response(url, params=params)
#         if response is None:
#             continue

#         ids = re.findall(r"ID=(C\d+)", response.text) 

#         for nist_id in ids:
#             nist_ids.add(nist_id)

#     return list(nist_ids)

import re
import json
import time 
import pandas as pd
import requests
from bs4 import BeautifulSoup
from jcamp import jcamp_read
from rdkit import Chem 
import io
import os
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


url = "https://webbook.nist.gov/cgi/cbook.cgi"
file = "uv_vis_400_molecules.csv"
num = 400

wl_min = 220
wl_max = 400
req_delay = 0.5
req_timeout = 20

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)

def get_response(url, params=None):
    try:
        response = session.get(
            url,
            params=params,
            timeout=req_timeout 
        )

        response.raise_for_status()
        time.sleep(req_delay)

        return response
    
    except requests.RequestException as error:
        print("Request error: ", error)
        return None 
    

# def get_ids():
#     nist_ids = set()

#     characters = "abcdefghijklmnopqrstuvwxyz"
#     for character in characters:
#         print("Searching: ", character)

#         params = {
#             "Name": character + "*",
#             "Units": "SI",
#             "Mask": "400"
#         }

#         response = get_response(url, params=params)
#         if response is None:
#             continue

#         ids = re.findall(r"ID=(C\d+)", response.text) 

#         for nist_id in ids:
#             nist_ids.add(nist_id)

#     return list(nist_ids)

def get_ids():
    nist_ids = set()

    elements = [
        "",
        "O",
        "O2",
        "O3",
        "N",
        "N2",
        "N3",
        "NO",
        "NO2",
        "N2O",
        "S",
        "SO",
        "F",
        "Cl",
        "Br",
        "I"
    ]

    for carbon in range(0, 800, 5):
        hydrogen = carbon + 5

        print(
            "Searching: ",
            carbon,
            "-",
            hydrogen,
            "| IDS: ",
            len(nist_ids)
        )

        params = {
            "Value": f"{carbon},{hydrogen}",
            "VType": "MW",
            "Units": "SI",
            "Mask": "400"
        }

        response = get_response(url, params=params)

        if response is None:
            continue

        ids = re.findall(
            r"(?:ID|InChI)=(C\d+)",
            response.text,
            re.IGNORECASE
        )

        for nist_id in ids:
            nist_ids.add(nist_id)

        if len(nist_ids) >= 2000:
            break

    return list(nist_ids)

def get_smiles(nist_id):
    params = {
        "Str2File": nist_id
    }


    # soup = BeautifulSoup(page_html, "html.parser")

    # mol_url = None 
    # for link in soup.find_all("a", href=True):
    #     href = link["href"]

    #     if "Str2d" in href:
    #         if href.startswith("/"):
    #             mol_url = "https://webbook.nist.gov" + href
    #         else:
    #             mol_url = href
    #         break
    # if mol_url is None:
    #     return None 
    
    response = get_response(url, params=params)

    if response is None:
        return None
    
    if "<html" in response.text.lower():
        return None
    
    molecule = Chem.MolFromMolBlock(
        response.text,
        sanitize=True,
        removeHs=True
    )

    if molecule is None:
        return None
    
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True
    )


# def get_jcamp_links(page_html):
#     soup = BeautifulSoup(page_html, "html.parser")
#     links = []

#     for link in soup.find_all("a", href=True):
#         href = link["href"]

#         if "JCAMP" in link.get_text().upper() or "jdx" in href.lower():
#             if href.startswith("/"):
#                 href = "https://webbook.nist.gov" + href

#             if href not in links:
#                 links.append(href)

#     return links


def read_jcamp(nist_id):
    params = {
        "JCAMP": nist_id,
        "Index": "0",
        "Type": "UVVis"
    }
    response = get_response(url, params=params)

    if response is None:
        return None
    
    if not response.text.lstrip().startswith("##"):
        print("No data: ", nist_id)
        return None 
    
    try:
        # spectrum = jcamp_read(response.text)
        spectrum = jcamp_read(io.StringIO(response.text))
        x_val = spectrum.get("x")
        y_val = spectrum.get("y")

        if x_val is None or y_val is None:
            return None
        
        wavelengths = []
        absorbance = []

        for x_val, y_val in zip(x_val, y_val):
            try:
                x_val = float(x_val)
                y_val = float(y_val)

            except (TypeError, ValueError):
                continue

            if np.isfinite(x_val) and np.isfinite(y_val):
                wavelengths.append(x_val)
                absorbance.append(y_val)

        if len(wavelengths) < 2:
            return None
        
        wavelengths = np.asarray(wavelengths, dtype=float)
        absorbance = np.asarray(absorbance, dtype=float)

        order = np.argsort(wavelengths)
        wavelengths = wavelengths[order]
        absorbance = absorbance[order]

        wavelengths, unique_indices = np.unique(
            wavelengths,
            return_index=True
        )
        absorbance = absorbance[unique_indices]

        if wavelengths[0] > wl_min or wavelengths[-1] < wl_max:
            return None

        x_val = np.linspace(
            wl_min,
            wl_max,
            wl_max - wl_min + 1
        )

        y_val = np.interp(
            x_val,
            wavelengths,
            absorbance
        )

        wavelengths = x_val.tolist()
        absorbance = y_val.tolist()

        if len(wavelengths) != 181 or len(absorbance) != 181:
            return None 
        
        return wavelengths, absorbance
    
    except Exception as error:
        print("Error: ", error)
        return None
    

# Collect the spectra
def collect_spectra():
    rows = []
    seen_smiles = set()

    if os.path.exists(file):
        try:
            dataframe = pd.read_csv(file)

            if all(
                column in dataframe.columns
                for column in [
                    "smiles",
                    "wavelength",
                    "absorbance"
                ]
            ):
                for _, row in dataframe.iterrows():
                    try:
                        wavelengths = json.loads(row["wavelength"])
                        absorbance = json.loads(row["absorbance"])

                        if (
                            len(wavelengths) == 181
                            and len(absorbance) == 181
                        ):
                            rows.append({
                                "smiles": row["smiles"],
                                "wavelength": row["wavelength"],
                                "absorbance": row["absorbance"]
                            })

                            seen_smiles.add(str(row["smiles"]))

                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError
                    ):
                        continue

                print(
                    "Existing molecules: ",
                    len(rows)
                )

        except Exception as error:
            print("Could not read existing CSV: ", error)
            rows = []
            seen_smiles = set()

    nist_ids = get_ids()
    print("IDS: ", len(nist_ids))

    for nist_id in nist_ids:
        if len(rows) >= num:
            break 

        # print(
        #     "Checking: ",
        #     nist_id,
        #     "| Collected: ",
        #     len(rows),
        #     "/",
        #     num
        # )

        print(
            "Checking: ",
            nist_id,
            "| Collected: ",
            len(rows),
            "/",
            num
        )

        spectrum = read_jcamp(nist_id)

        if spectrum is None:
            print("No spectrum: ", nist_id)
            continue 

        smiles = get_smiles(nist_id)
        if smiles is None:
            print("No SMILES: ", nist_id)
            continue

        if smiles in seen_smiles:
            print("Dup. SMILES: ", smiles)
            continue 

        # params = {
        #     "ID": nist_id,
        #     "Units": "SI",
        #     "Mask": "400"
        # }

        wavelengths, absorbance = spectrum

        # response = get_response(url, params=params)
        # if response is None:
        #     continue 

        # if "UV/Visible spectrum" not in response.text:
        #     continue 

        # smiles = get_smiles(response.text)
        # if smiles is None:
        #     continue 

        # if smiles in seen_smiles:
        #     continue 

        # jcamp_links = get_jcamp_links(response.text)
        # if len(jcamp_links) == 0:
        #     continue 

        # spectrum_found = False

        # for jcamp_url in jcamp_links:
        #     spectrum = read_jcamp(jcamp_url)

        #     if spectrum is None:
        #         continue 

        #     wavelengths, absorbance = spectrum 

        rows.append({
            "smiles": smiles, 
            "wavelength": json.dumps(wavelengths),
            "absorbance": json.dumps(absorbance)
        })

        seen_smiles.add(smiles)

        dataframe = pd.DataFrame(
            rows,
            columns=[
                "smiles",
                "wavelength",
                "absorbance"
            ]
        )

        dataframe.to_csv(file, index=False)

        print(
            "Saved: ",
            len(rows),
            "|",
            smiles,
            "| Points: ",
            len(wavelengths)
        )

        #     seen_smiles.add(smiles)
        #     spectrum_found = True

        #     pd.DataFrame(rows).to_csv(file, index=False)
        #     break
        # if not spectrum_found:
        #     continue 

    dataframe = pd.DataFrame(
        rows, 
        columns=[
            "smiles",
            "wavelength",
            "absorbance"
        ]
    )

    dataframe.to_csv(file, index=False)
    print()
    print("Finished")
    print("Molecules collected: ", len(dataframe))
    print("Saved to: ", file)

collect_spectra()

# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import inchi
# import os 
# import re 
# import time 
# import tempfile

# import numpy as np
# import requests

# from bs4 import BeautifulSoup
# from jcamp import jcamp_readfile



# # web scraping 
# # later on


# file_1 = "UV_w_SMILES.csv"
# file_2 = "new_UV_w_SMILES.csv"
# file = "UV_w_SMILES_1200.csv"



# num = 268


# EXISTING_HAS_HEADER = False
# NEW_FILE_HAS_HEADER = True 
# NEW_SMILES_COLUMN = "smiles"


# # nist web scraping begins here 
# nist_file = "nist_molecules.csv"
# url = "https://webbok.nist.giv/cgi/cbook.cgi"

# wavelength_min = 220.0
# wavelength_max = 400.0

# request_delay = 2.5
# request_timeout = 25

# max_candidates = 1000
# min_coverage = 0.75 # require at least this fraction of the target wavelength range to be present in the spectra 



# def standardize_molecule(smiles):
#     if pd.isna(smiles):
#         return None, None 
    
#     smiles = str(smiles).strip()
#     if not smiles: 
#         return None, None

#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None, None

#         canonical = Chem.MolToSmiles(
#             mol,
#             canonical=True,
#             isomericSmiles=True
#         )

#         full_inchikey = inchi.MolToInchiKey(mol)

#         if not full_inchikey:
#             return canonical, None 

#         connectivity_key = full_inchikey.split("-")[0]
#         return canonical, connectivity_key
#     except Exception:
#         return None, None 
    

# # existing dataset 

# if EXISTING_HAS_HEADER:
#     existing_df = pd.read_csv(file_1)
#     existing_smiles_column = "smiles"
# else:
#     existing_df = pd.read_csv(file_1, header=None)
#     existing_smiles_column = 0

# print(f"Existing rows before clean:  {len(existing_df)}")


# existing_standardized = existing_df[existing_smiles_column].apply(
#     standardize_molecule
# )

# existing_df["_canonical_smiles"] = existing_standardized.apply(
#     lambda result: result[0]
# )

# existing_df["_connectivity_key"] = existing_standardized.apply(
#     lambda result: result[1]
# )


# # this removes the invalid existing smiles (duplicates)

# invalid = existing_df["_connectivity_key"].isna().sum()

# if invalid:
#     print(f"Warning: {invalid} existing rows have invalid SMILES")

# valid = existing_df[existing_df["_connectivity_key"].notna()].copy()

# existing_dups = valid.duplicated(
#     subset = "_connectivity_key",
#     keep="first"
# ).sum()

# print(f"Duplications in existing dataset: {existing_dups}")

# # remove duplicates already present inside the first dataset 
# valid = valid.drop_duplicates(
#     subset="_connectivity_key",
#     keep="first"
# ).copy()

# existing_keys = set(
#     valid["_connectivity_key"].dropna()
# )

# print(
#     f"Uq. molecular structures in existing dataset: "
#     f"{len(existing_keys)}"
# )

# # existing_keys = set(
# #     existing_df["_connectivity_key"].dropna()
# # )

# # print(f"Uq. molecular steuctures in existing dataset: {len(existing_keys)}")



# # NIST helper functions 
# def cas_to_nist_id(cas_number):
#     if pd.isna(cas_number):
#         return None
    
#     cas_number = str(cas_number).strip()

#     if not re.fullmarch(r"\d{2,7}-\d[2}-\d", cas_number):
#         return None 
    
#     return "C" + cas_number.replace("-", "")

# def make_nist_session():
#     session = requests.Session()
#     session.headers.update({
#         "User-Agent":(
#             "UV vis collection dataset script"
#         )
#     })

#     return session 

# def get_nist_page(session, nist_id):
#     parameters = {
#         "ID": nist_id,
#         "Mask": "400"
#     }

#     response = session.get(
#         url,
#         params=parameters,
#         timeout=request_timeout
#     )

#     response.raise_for_status()
#     return response.text

# def find_nist_jcamp_url(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for link in soup.find_all("a", href=True):
#         href = link["href"]

#         href_upper = href.upper()
#         link_text = link.get_text(" ", strip=True).upper()

#         if ( 
#             "JCAMP" in href_upper
#             and (
#                 "UVVIS" in href_upper
#                 or "UV/VIS" in link_text
#                 or "JCAMP-DX" in link_text
#             )
#         ):
#             if href.startswith("http"):
#                 return href
#             if href.startswith("/"):
#                 return "https://webbook.nist.gov" + href
            
#             return "https://webbok.nist.gov/cgi/" + href 
        
#     return None 

# def extract_nist_inchi(html):
#     soup = BeautifulSoup(html, "html.parser")
#     page_text = soup.get_text("\n", strip=True)
#     match = re.search(
#         r"IUPAC Standard InChI:\s*(InChI=[^\s]+)",
#         page_text,
#         flags=re.IGNORECASE
#     )

#     if match:
#         return match.group(1).strip()
    
#     return None 


# # CONTINUE WITH extract_nist_name(html)
# # -------------------------------------

# def extract_nist_name(html):
#     soup = BeautifulSoup(html, "html.parser")
#     title = soup.find("h1")
#     if title:
#         return title.get_text(" ", strip=True)

#     return None

# ## missing code here 

# if NEW_FILE_HAS_HEADER:
#     new_df = pd.read_csv(file_2)

#     if NEW_SMILES_COLUMN not in new_df.columns:
#         raise ValueError(
#             f"Column '{NEW_SMILES_COLUMN}' was not found in {file_2}.\n"
#             f"Available: {list(new_df.columns)}"
#         )
    
#     new_smiles_column = NEW_SMILES_COLUMN
# else:
#     new_df = pd.read_csv(file_2, header=None)
#     new_smiles_column = 0


# print(f"\nNew rows: {len(new_df)}")


# # standardize new molecules 

# new_standardized = new_df[new_smiles_column].apply(
#     standardize_molecule
# )

# new_df["_canonical_smiles"] = new_standardized.apply(
#     lambda result: result[0]
# )

# new_df["_connectivity_key"] = new_standardized.apply(
#     lambda result: result[1]
# )


# # remove the invalid molecules 

# invalid_new_mask = new_df["_connectivity_key"].isna()
# invalid_new_count = invalid_new_mask.sum()

# new_df = new_df[~invalid_new_mask].copy()
# print(f"Invalid new SMILES removed: {invalid_new_count}")

# # remove dups against exsiting dataset 

# dups_existing_mask = new_df["_connectivity_key"].isin(existing_keys)
# dups_existing_count = dups_existing_mask.sum()
# new_df = new_df[~dups_existing_mask].copy()
# print(
#     "New molecules present in existing d.s.: "
#     f"{dups_existing_count}"
# )


# # remove the duplicates within new dataset 
# b4_internal_dedups = len(new_df)
# new_df = new_df.drop_duplicates(
#     subset="_connectivity_key",
#     keep="first"
# ).copy()

# internal_dups = b4_internal_dedups - len(new_df)

# # 268 molecule selection from the database (NIST)


# if len(new_df) < num:
#     print(
#         f"\nWarning: only {len(new_df)} unique molecules are available. "
#         f"You still need {num - len(new_df)} more. "
#     )

#     selected_new_df = new_df.head(num).copy()

# else:
#     selected_new_df = new_df.head(num).copy()

# print(f"Molecules selected to add: {len(selected_new_df)}")

# selected_new_df[new_smiles_column] = selected_new_df["_canonical_smiles"]
# # replace with canonical smiles 


# # check the column count 
# existing_output = existing_df.drop(
#     columns=["_canonical_smiles", "_connectivity_key"]
# )

# new_output = selected_new_df.drop(
#     columns=["_canonical_smiles", "_connectivity_key"]

# )

# if not EXISTING_HAS_HEADER and not NEW_FILE_HAS_HEADER:
#     if existing_output.shape[1] != new_output.shape[1]:
#         raise ValueError(
#             "\nThe files have different numbers of columns:\n"
#             f"Existing dataset: {existing_output.shape[1]} columns\n"
#             f"New dataset: {new_output.shape[1]} columns\n\n"
#             "Both files must contain one SMILES column plus the same number of UV vis intensity columns. "
#         )
    
# # merge datasets 

# if EXISTING_HAS_HEADER and NEW_FILE_HAS_HEADER:
#     missing_columns = [
#         column  
#         for column in existing_output.columns
#         if column not in new_output.columns 
#     ]

#     if missing_columns:
#         raise ValueError(
#             "The new dataset is mssing these req. columns:\n"
#             f"{missing_columns}"
#         )
    
#     new_output = new_output[existing_output.columns]

# combined_df = pd.concat(
#     [existing_output, new_output],
#     ignore_index=True
# )


# # final check 

# final_smiles_column = (
#     "smiles"
#     if EXISTING_HAS_HEADER
#     else 0
# )

# final_keys = combined_df[final_smiles_column].apply(
#     lambda value: standardize_molecule(value)[1]
# )
# final_dup_count = final_keys.dropna().duplicated().sum()
# print(f"\nFinal duplicate structures detected: {final_dup_count}")


# # save the results 

# combined_df.to_csv(
#     file,
#     index=False, 
#     header=EXISTING_HAS_HEADER
# )

# # save accepted NIST molecules 
# selected_new_df.to_csv(
#     "268_more_molecules.csv",
#     index=False
# )

# print(f"Saved final dataset to: {file}")
