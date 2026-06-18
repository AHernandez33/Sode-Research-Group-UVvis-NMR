# NMR

from rdkit import Chem
from rdkit.Chem import AllChem

from pyscf import gto, dft
from pyscf.prop import nmr
from pyscf.geomopt.geometric_solver import optimize

import numpy as np
import matplotlib.pyplot as plt


# Builds a molecule from a SMILES string, optimizes its geometry,
# runs an NMR calculation, and returns hydrogen shielding data
# along with metadata about each hydrogen atom.
def get_h_shieldings_with_metadata(smiles):
    # Convert the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add explicit hydrogen atoms to the molecule
    mol = Chem.AddHs(mol)

    # Generate an initial 3D structure and optimize it using MMFF
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Get the optimized 3D coordinates from RDKit
    conf = mol.GetConformer()
    geometric_lines = []
    atom_symbols = []

    # Convert atom coordinates into the text format required by PySCF
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_symbols.append(atom.GetSymbol())
        geometric_lines.append(
            f"{atom.GetSymbol()} {pos.x:6f} {pos.y:6f} {pos.z:6f}"
        )

    # Join all atom coordinate lines into one geometry string
    geometric_string = "\n".join(geometric_lines)

    # Create a PySCF molecule object
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = geometric_string
    pyscf_mol.basis = "def2-svp"
    pyscf_mol.charge = 0
    pyscf_mol.spin = 0
    pyscf_mol.unit = "Angstrom"
    pyscf_mol.build()

    # Set up a DFT calculation using the B3LYP functional
    mf = dft.RKS(pyscf_mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9

    # Optimize the molecular geometry using PySCF/geomeTRIC
    print(f"\nOptimizing geometry for {smiles}...")
    pyscf_mol = optimize(mf)

    # Run a final DFT calculation on the optimized geometry
    print(f"Running final SCF for {smiles}...")
    mf = dft.RKS(pyscf_mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-9
    mf.kernel()

    # Calculate NMR shielding tensors
    nmr_props = nmr.rks.NMR(mf).kernel()

    h_data = []

    # Extract shielding data only for hydrogen atoms
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        # Skip atoms that are not hydrogen
        if atom.GetSymbol() != "H":
            continue

        # Identify the atom that this hydrogen is attached to
        neighbor = atom.GetNeighbors()[0]
        parent_idx = neighbor.GetIdx()
        parent_symbol = neighbor.GetSymbol()

        # Count hydrogens attached to the same parent atom
        attached_h = [
            nbr.GetIdx()
            for nbr in neighbor.GetNeighbors()
            if nbr.GetSymbol() == "H"
        ]

        # Detect whether this hydrogen belongs to a methyl group
        is_methyl_h = parent_symbol == "C" and len(attached_h) == 3

        # Detect whether this hydrogen is attached to oxygen
        is_oh = parent_symbol == "O"

        # Convert the shielding tensor into an isotropic shielding value
        tensor = nmr_props[idx]
        shielding = np.trace(tensor) / 3.0

        # Store hydrogen metadata and shielding value
        h_data.append({
            "h_idx": idx,
            "parent_idx": parent_idx,
            "shielding": shielding,
            "is_methyl_h": is_methyl_h,
            "is_oh": is_oh,
        })

    return h_data


# Converts shielding values into chemical shifts in ppm
# using a reference shielding value.
def convert_to_ppm(h_data, sigma_ref):
    ppm_data = []

    for item in h_data:
        ppm_data.append({
            **item,
            "shift": sigma_ref - item["shielding"],
        })

    return ppm_data


# Averages the three hydrogens in each methyl group into one peak.
# Non-methyl hydrogens are kept as individual peaks.
def average_methyl_groups(ppm_data):
    methyl_groups = {}
    non_methyl_peaks = []

    # Separate methyl hydrogens from all other hydrogens
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

    # Average the shifts of hydrogens attached to each methyl carbon
    for parent_idx, hydrogens in methyl_groups.items():
        avg_shift = np.mean([h["shift"] for h in hydrogens])

        averaged_peaks.append({
            "shift": avg_shift,
            "intensity": len(hydrogens),
            "is_oh": False,
            "label": "CH3",
        })

    return averaged_peaks + non_methyl_peaks


# Groups peaks that are very close together in chemical shift.
# OH peaks are kept separate and given broader peak widths.
def group_close_shifts(peaks, tolerance=0.02):
    groups = []

    # Sort peaks from low to high chemical shift before grouping
    for peak in sorted(peaks, key=lambda item: item["shift"]):
        shift = peak["shift"]

        # Keep OH peaks separate
        if peak["is_oh"]:
            groups.append([peak])
            continue

        # Start the first group if no groups exist yet
        if not groups:
            groups.append([peak])
            continue

        last_group = groups[-1]

        # Calculate the intensity-weighted average shift of the last group
        last_avg = np.average(
            [p["shift"] for p in last_group],
            weights=[p["intensity"] for p in last_group]
        )

        # Check if the previous group contains an OH peak
        last_has_oh = any(p["is_oh"] for p in last_group)

        # Start a new group if the peak is too far away or the last group has OH
        if last_has_oh or abs(last_avg - shift) > tolerance:
            groups.append([peak])
        else:
            last_group.append(peak)

    grouped_shifts = []
    intensities = []
    widths = []

    # Convert each group into one final peak
    for group in groups:
        total_intensity = sum(p["intensity"] for p in group)

        # Calculate the intensity-weighted average shift for the group
        avg_shift = np.average(
            [p["shift"] for p in group],
            weights=[p["intensity"] for p in group]
        )

        # Determine whether the group contains an OH peak
        is_oh = any(p["is_oh"] for p in group)

        grouped_shifts.append(avg_shift)
        intensities.append(total_intensity)

        # OH peaks are plotted broader than normal hydrogen peaks
        widths.append(0.20 if is_oh else 0.025)

    return grouped_shifts, intensities, widths


# Defines the Lorentzian peak shape used to simulate NMR signals
def lorentzian(x, x0, width):
    return (width / np.pi) / ((x - x0) ** 2 + width ** 2)


# Adds a peak or multiplet pattern to the simulated spectrum
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
    # Convert coupling constant from Hz to ppm
    j_ppm = j_hz / spectrometer_mhz

    # Define peak offsets and relative intensities for each splitting pattern
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

    # Normalize the multiplet weights so total intensity stays consistent
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)

    # Add each component of the multiplet to the spectrum
    for offset, weight in zip(offsets, weights):
        y += intensity * weight * lorentzian(x, center + offset, width)

    return y


# Chooses a simple splitting pattern based on shift and peak width
def choose_pattern(shift, intensity, width):
    # Broad peaks, such as OH peaks, are treated as singlets
    if width >= 0.15:
        return "singlet"

    # Aromatic-region peaks are treated as multiplets
    if shift > 6.0:
        return "multiplet"

    return "singlet"


smiles = "c1ccc(-c2ccc3ccccc3c2)cc1"

# Calculate hydrogen shielding values for the target molecule
h_data = get_h_shieldings_with_metadata(smiles)

# Calculate hydrogen shielding values for TMS, used as the NMR reference
tms_h_data = get_h_shieldings_with_metadata("C[Si](C)(C)C")

# Compute the average TMS hydrogen shielding to use as the reference value
sigma_ref = np.mean([item["shielding"] for item in tms_h_data])

# Convert calculated shieldings into chemical shifts in ppm
ppm_data = convert_to_ppm(h_data, sigma_ref)

# This is the important fix:
# average the 3 hydrogens on each methyl carbon before peak grouping.
peaks = average_methyl_groups(ppm_data)

# Group nearby peaks and assign peak intensities and widths
shifts_ppm, intensities, widths = group_close_shifts(peaks, tolerance=0.02)

# Print the calculated peak positions, intensities, widths, and patterns
print("\nComputed peaks:")
for shift, intensity, width in zip(shifts_ppm, intensities, widths):
    pattern = choose_pattern(shift, intensity, width)
    print(f"{shift:8.3f} ppm   intensity {intensity}H   width {width}   {pattern}")


# Create the x-axis chemical shift range and initialize the spectrum intensity
x = np.linspace(-1, 13, 8000)
y = np.zeros_like(x)

# Add each calculated peak or multiplet to the simulated spectrum
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

# Plot the simulated 1H NMR spectrum
plt.figure()
plt.plot(x, y, color="black", linewidth=0.8)

# Reverse the x-axis, as is standard for NMR spectra
plt.gca().invert_xaxis()

# Add axis labels and title
plt.xlabel("Chemical shift in ppm")
plt.ylabel("Intensity")
plt.title("1H NMR spectrum of 2-phenylnaphthalene")

# Display the plot
plt.show()