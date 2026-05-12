# NMR

from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from pyscf.prop import nmr
from pyscf.geomopt.geometric_solver import optimize
import numpy as np
import matplotlib.pyplot as plt

# SMILES string extraction
def get_h_shieldings_with_metadata(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    geometric_lines = []
    atom_symbols = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_symbols.append(atom.GetSymbol())
        geometric_lines.append(
            f"{atom.GetSymbol()} {pos.x:6f} {pos.y:6f} {pos.z:6f}"
        )

    geometric_string = "\n".join(geometric_lines)

    pyscf_mol = gto.Mole()
    pyscf_mol.atom = geometric_string
    pyscf_mol.basis = "def2-svp"
    pyscf_mol.charge = 0
    pyscf_mol.spin = 0
    pyscf_mol.unit = "Angstrom"
    pyscf_mol.build()

    mf = dft.RKS(pyscf_mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9

    print(f"\nOptimizing geometry for {smiles}...")
    pyscf_mol = optimize(mf)

    print(f"Running final SCF for {smiles}...")
    mf = dft.RKS(pyscf_mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9
    mf.kernel()

    nmr_props = nmr.rks.NMR(mf).kernel()

    h_data = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        if atom.GetSymbol() != "H":
            continue

        neighbor = atom.GetNeighbors()[0]
        parent_idx = neighbor.GetIdx()
        parent_symbol = neighbor.GetSymbol()

        attached_h = [
            nbr.GetIdx()
            for nbr in neighbor.GetNeighbors()
            if nbr.GetSymbol() == "H"
        ]

        is_methyl_h = parent_symbol == "C" and len(attached_h) == 3
        is_oh = parent_symbol == "O"

        tensor = nmr_props[idx]
        shielding = np.trace(tensor) / 3.0

        h_data.append({
            "h_idx": idx,
            "parent_idx": parent_idx,
            "shielding": shielding,
            "is_methyl_h": is_methyl_h,
            "is_oh": is_oh,
        })

    return h_data


def convert_to_ppm(h_data, sigma_ref):
    ppm_data = []

    for item in h_data:
        ppm_data.append({
            **item,
            "shift": sigma_ref - item["shielding"],
        })

    return ppm_data


def average_methyl_groups(ppm_data):
    methyl_groups = {}
    non_methyl_peaks = []

    for item in ppm_data:
        if item["is_methyl_h"]:
            methyl_groups.setdefault(item["parent_idx"], []).append(item)
        else:
            non_methyl_peaks.append({
                "shift": item["shift"],
                "intensity": 1,
                "is_oh": item["is_oh"],
                "label": "H",
            })

    averaged_peaks = []

    for parent_idx, hydrogens in methyl_groups.items():
        avg_shift = np.mean([h["shift"] for h in hydrogens])

        averaged_peaks.append({
            "shift": avg_shift,
            "intensity": len(hydrogens),
            "is_oh": False,
            "label": "CH3",
        })

    return averaged_peaks + non_methyl_peaks


def group_close_shifts(peaks, tolerance=0.02):
    groups = []

    for peak in sorted(peaks, key=lambda item: item["shift"]):
        shift = peak["shift"]

        if peak["is_oh"]:
            groups.append([peak])
            continue

        if not groups:
            groups.append([peak])
            continue

        last_group = groups[-1]
        last_avg = np.average(
            [p["shift"] for p in last_group],
            weights=[p["intensity"] for p in last_group]
        )

        last_has_oh = any(p["is_oh"] for p in last_group)

        if last_has_oh or abs(last_avg - shift) > tolerance:
            groups.append([peak])
        else:
            last_group.append(peak)

    grouped_shifts = []
    intensities = []
    widths = []

    for group in groups:
        total_intensity = sum(p["intensity"] for p in group)

        avg_shift = np.average(
            [p["shift"] for p in group],
            weights=[p["intensity"] for p in group]
        )

        is_oh = any(p["is_oh"] for p in group)

        grouped_shifts.append(avg_shift)
        intensities.append(total_intensity)
        widths.append(0.20 if is_oh else 0.025)

    return grouped_shifts, intensities, widths


def lorentzian(x, x0, width):
    return (width / np.pi) / ((x - x0) ** 2 + width ** 2)


def add_multiplet(
    y,
    x,
    center,
    intensity,
    width,
    pattern="singlet",
    j_hz=7.0,
    spectrometer_mhz=400.0,
):
    j_ppm = j_hz / spectrometer_mhz

    if pattern == "singlet":
        offsets = [0.0]
        weights = [1]

    elif pattern == "doublet":
        offsets = [-j_ppm / 2, j_ppm / 2]
        weights = [1, 1]

    elif pattern == "triplet":
        offsets = [-j_ppm, 0.0, j_ppm]
        weights = [1, 2, 1]

    elif pattern == "quartet":
        offsets = [-1.5 * j_ppm, -0.5 * j_ppm, 0.5 * j_ppm, 1.5 * j_ppm]
        weights = [1, 3, 3, 1]

    elif pattern == "multiplet":
        offsets = [-1.5 * j_ppm, -0.5 * j_ppm, 0.5 * j_ppm, 1.5 * j_ppm]
        weights = [1, 2, 2, 1]

    else:
        offsets = [0.0]
        weights = [1]

    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    for offset, weight in zip(offsets, weights):
        y += intensity * weight * lorentzian(x, center + offset, width)

    return y


def choose_pattern(shift, intensity, width):
    if width >= 0.15:
        return "singlet"

    if shift > 6.0:
        return "multiplet"

    return "singlet"


# 1,2,4-trimethylbenzene
smiles = "Cc1ccc(C)c(C)c1"

h_data = get_h_shieldings_with_metadata(smiles)
tms_h_data = get_h_shieldings_with_metadata("C[Si](C)(C)C")

sigma_ref = np.mean([item["shielding"] for item in tms_h_data])

ppm_data = convert_to_ppm(h_data, sigma_ref)

# This is the important fix:
# average the 3 hydrogens on each methyl carbon before peak grouping.
peaks = average_methyl_groups(ppm_data)

shifts_ppm, intensities, widths = group_close_shifts(peaks, tolerance=0.02)

print("\nComputed peaks:")
for shift, intensity, width in zip(shifts_ppm, intensities, widths):
    pattern = choose_pattern(shift, intensity, width)
    print(f"{shift:8.3f} ppm   intensity {intensity}H   width {width}   {pattern}")


x = np.linspace(-1, 13, 8000)
y = np.zeros_like(x)

for shift, intensity, width in zip(shifts_ppm, intensities, widths):
    pattern = choose_pattern(shift, intensity, width)

    y = add_multiplet(
        y,
        x,
        center=shift,
        intensity=intensity,
        width=width,
        pattern=pattern,
        j_hz=7.5,
        spectrometer_mhz=400.0,
    )

plt.figure()
plt.plot(x, y, color="black", linewidth=0.8)
plt.gca().invert_xaxis()
plt.xlabel("Chemical shift in ppm")
plt.ylabel("Intensity")
plt.title("1H NMR spectrum of 1,2,4-trimethylbenzene")
plt.show()
