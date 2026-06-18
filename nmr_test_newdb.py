from rdkit import Chem

supplier = Chem.SDMolSupplier(
    r"C:\Users\herna\Sode Labs\NMR\nmrshiftdb2withsignals.sd"
)

for i, mol in enumerate(supplier):
    if mol is None:
        continue

    print("Molecule", i)

    for prop in mol.GetPropNames():
        print(prop)

    break