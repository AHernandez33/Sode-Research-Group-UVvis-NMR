import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

csv = "nmrshift_db2_1H_1500_expdata.csv"
folder = "XYZ FILES 2"
os.makedirs(folder, exist_ok=True)

failed = [] 

# def embed_with_fallbacks(mol, compound_id):
#     params = AllChem.ETKDGv3()
#     params.randomSeed = 1
#     # params.maxAttempts = 1000

#     status = AllChem.EmbedMolecule(mol, params=params, maxAttempts=1000)

#     if status == 0:
#         return mol

#     params = AllChem.ETKDGv3()
#     params.randomSeed = 1
#     # params.maxAttempts = 1000
#     params.useRandomCoords = True

#     status = AllChem.EmbedMolecule(mol, params=params, maxAttempts=1000)
#     if status == 0:
#         return mol

#     params = AllChem.ETKDGv3()
#     params.randomSeed = 1
#     # params.maxAttempts = 2000
#     params.useRandomCoords = True
#     params.enforceChirality = False

#     status = AllChem.EmbedMolecule(mol, params=params, maxAttempts=1000)
#     if status == 0:
#         return mol

#     print("Embedding failed after fallbacks: ", compound_id)
#     return None

def embed_with_fallbacks(mol, compound_id):
    params = AllChem.ETKDGv3()
    params.randomSeed = 1

    status = AllChem.EmbedMolecule(mol, params)

    if status == 0:
        return mol

    status = AllChem.EmbedMolecule(
        mol,
        maxAttempts=1000,
        randomSeed=1,
        useRandomCoords=True
    )

    if status == 0:
        return mol

    status = AllChem.EmbedMolecule(
        mol,
        maxAttempts=2000,
        randomSeed=1,
        useRandomCoords=True,
        enforceChirality=False
    )

    if status == 0:
        return mol

    print("Embedding failed after fallbacks: ", compound_id)
    return None

df = pd.read_csv(csv)



for i, row in df.iterrows():
    compound_id = str(row["compound_id"])
    smiles = str(row["smiles"])
    name = str(row["name"])
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Bad SMILES: ", compound_id)
            continue
        mol = Chem.AddHs(mol)

        # status = AllChem.EmbedMolecule(
        #     mol,
        #     randomSeed=1
        # )

        # if status != 0:
        #     print("Embedding Failed: ", compound_id)
        #     continue 

        # AllChem.MMFFOptimizeMolecule(mol)

        mol = embed_with_fallbacks(mol, compound_id)
        if mol is None:
            failed.append([compound_id, smiles, "embedding failed"])
            continue

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
            except:
                print("Optimization failed: ", compound_id)
                failed.append([compound_id, smiles, "optimization failed"])
                continue 

        conf = mol.GetConformer()
        compound_name = name
        for bad in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "[", "]"]:
            compound_name = compound_name.replace(bad, "_")

        compound_name = compound_name[:80]

        xyz_path = os.path.join(
            folder,
            f"{compound_id}_{compound_name}.xyz"
        )

        with open(xyz_path, "w", encoding="utf-8") as f:
            f.write(f"{mol.GetNumAtoms()}\n")
            f.write(f"{compound_id} | {name}\n")
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(
                    atom.GetIdx()
                )
                f.write(
                    f"{atom.GetSymbol()} "
                    f"{pos.x:.6f} "
                    f"{pos.y:.6f} "
                    f"{pos.z:.6f}\n"
                )

        if (i + 1) % 1 == 0:
            print("Saved: ", i + 1, "|", compound_id, "|", name)
    except Exception as e:
        print("Failed: ", compound_id, e)
        failed.append([compound_id, smiles, str(e)])

# pd.DataFrame(
#     failed, 
#     columns=["compound_id", "smiles", "reason"]
# ).to_csv(
#     "embedding_failed."
# )


print("Finished")