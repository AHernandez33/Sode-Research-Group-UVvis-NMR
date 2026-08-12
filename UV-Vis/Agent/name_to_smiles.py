import pubchempy as pcp

def name_to_smiles():
    name = input("Enter a chemical name: ").strip()
    try:
        compounds = pcp.get_compounds(name, "name")

        if not compounds:
            print("Invalid compound")
            return
        smiles = compounds[0].canonical_smiles

        print(f"SMILES: {smiles}")

    except Exception as error:
        print("Error: ", error)

if __name__ == "__main__":
    name_to_smiles()