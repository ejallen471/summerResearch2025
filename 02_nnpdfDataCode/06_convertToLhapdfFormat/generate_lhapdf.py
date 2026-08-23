"""
We have an early synthetic prototype for exploring the text layout of an
LHAPDF replica set. It is not used for the KDE reconstruction.

Run with the following command:

python generate_lhapdf.py

This file does the following:

1. Create invented x and Q grids, metadata and example parton functions.
2. Add independent random variations to create synthetic replica values.
3. Write prototype .info and .dat files for inspecting the intended layout.
"""

import numpy as np
import os

def write_lhapdf_info(output_dir, set_name, num_replicas, metadata):
    """
    Write the .info file for LHAPDF format
    """
    info_content = f"""SetDesc: "{metadata.get('description', 'Custom PDF set')}"
SetIndex: {metadata.get('set_index', 999999)}
Authors: {metadata.get('authors', 'Unknown')}
Reference: {metadata.get('reference', 'arXiv:xxxx.xxxxx')}
Format: lhagrid1
DataVersion: 1
NumMembers: {num_replicas + 1}
Particle: {metadata.get('particle', 2212)}
Flavors: [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
OrderQCD: {metadata.get('order_qcd', 1)}
FlavorScheme: {metadata.get('flavor_scheme', 'variable')}
NumFlavors: {metadata.get('num_flavors', 5)}
ErrorType: replicas
XMin: {metadata.get('xmin', 1.0e-9)}
XMax: {metadata.get('xmax', 1.0)}
QMin: {metadata.get('qmin', 1.0)}
QMax: {metadata.get('qmax', 1.0e5)}
MZ: {metadata.get('mz', 91.1876)}
MUp: 0.0
MDown: 0.0
MStrange: 0.0
MCharm: {metadata.get('mc', 1.51)}
MBottom: {metadata.get('mb', 4.92)}
MTop: {metadata.get('mt', 172.5)}
AlphaS_MZ: {metadata.get('alphas_mz', 0.118)}
AlphaS_OrderQCD: {metadata.get('alphas_order', 1)}
AlphaS_Type: ipol
"""

    info_file = os.path.join(output_dir, f"{set_name}.info")
    with open(info_file, 'w') as f:
        f.write(info_content)

    print(f"Written info file: {info_file}")


def write_lhapdf_dat(output_dir, set_name, replica_num, x_grid, q_grid, pdf_values):
    """
    Write a single .dat file for one PDF replica in LHAPDF grid format

    Parameters:
    -----------
    output_dir : str
        Directory to write files
    set_name : str
        Name of the PDF set
    replica_num : int
        Replica number (0 for central value)
    x_grid : array
        x values (momentum fraction)
    q_grid : array
        Q values (energy scale in GeV)
    pdf_values : dict
        Dictionary with keys as PDG flavor codes and values as 2D arrays [nx, nq]
        E.g., {21: gluon_array, 2: up_array, -2: antiup_array, ...}
    """

    dat_file = os.path.join(output_dir, f"{set_name}_{replica_num:04d}.dat")

    with open(dat_file, 'w') as f:
        # Write header
        f.write("PdfType: replica\n")
        f.write("Format: lhagrid1\n")
        f.write("---\n")

        # Define flavor ordering (standard LHAPDF ordering)
        flavors = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]

        # Write x grid
        f.write(f" {x_grid[0]:.8e} {x_grid[-1]:.8e} {len(x_grid)}\n")
        for x in x_grid:
            f.write(f" {x:.8e}")
        f.write("\n")

        # Write Q grid
        f.write(f" {q_grid[0]:.8e} {q_grid[-1]:.8e} {len(q_grid)}\n")
        for q in q_grid:
            f.write(f" {q:.8e}")
        f.write("\n")

        # Write flavor list
        f.write(f" {' '.join(map(str, flavors))}\n")

        # Write PDF values
        f.write("---\n")

        # Loop over Q values, then x values, then flavors
        for iq, q in enumerate(q_grid):
            for ix, x in enumerate(x_grid):
                for flavor in flavors:
                    if flavor in pdf_values:
                        value = pdf_values[flavor][ix, iq]
                    else:
                        value = 0.0
                    f.write(f" {value:.8e}")
                f.write("\n")

    print(f"Written replica {replica_num}: {dat_file}")


def example_pdf_function(x, q, flavor):
    """
    Example PDF function - replace with your actual PDF parametrization
    This is a very simplified model for demonstration
    """
    # Simple parametrization (not physical, just for demonstration)
    if flavor == 21:  # gluon
        return 3.0 * x**(-0.3) * (1-x)**5 * np.log(q/1.0)
    elif flavor in [1, 2]:  # u, d valence-like
        return 0.5 * x**(-0.5) * (1-x)**3
    elif flavor in [-1, -2]:  # ubar, dbar sea-like
        return 0.1 * x**(-0.2) * (1-x)**6
    elif flavor in [3, -3]:  # s, sbar
        return 0.05 * x**(-0.2) * (1-x)**7
    elif flavor in [4, -4]:  # c, cbar
        return 0.01 * x**(-0.1) * (1-x)**8 if q > 1.5 else 0.0
    elif flavor in [5, -5]:  # b, bbar
        return 0.005 * x**(-0.1) * (1-x)**9 if q > 5.0 else 0.0
    else:
        return 0.0


def generate_pdf_replicas(set_name, num_replicas, output_dir="./"):
    """
    Main function to generate LHAPDF replica set
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define grids
    # Logarithmic x grid
    x_grid = np.logspace(-9, 0, 100)  # 100 points from 10^-9 to 1

    # Logarithmic Q grid
    q_grid = np.logspace(0, 5, 50)  # 50 points from 1 GeV to 10^5 GeV

    # Metadata
    metadata = {
        'description': 'Example PDF replica set',
        'set_index': 999999,
        'authors': 'Your Name',
        'reference': 'arXiv:xxxx.xxxxx',
        'particle': 2212,  # proton
        'order_qcd': 1,  # NLO
        'flavor_scheme': 'variable',
        'num_flavors': 5,
        'xmin': x_grid.min(),
        'xmax': x_grid.max(),
        'qmin': q_grid.min(),
        'qmax': q_grid.max(),
        'alphas_mz': 0.118,
    }

    # Write .info file
    write_lhapdf_info(output_dir, set_name, num_replicas, metadata)

    # Generate replicas
    flavors = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]

    for replica in range(num_replicas + 1):  # +1 for central member (replica 0)
        pdf_values = {}

        # Generate PDF values for each flavor
        for flavor in flavors:
            pdf_array = np.zeros((len(x_grid), len(q_grid)))

            for ix, x in enumerate(x_grid):
                for iq, q in enumerate(q_grid):
                    # Central value
                    central = example_pdf_function(x, q, flavor)

                    # Add random variation for replicas (not replica 0)
                    if replica > 0:
                        variation = 1.0 + 0.1 * np.random.randn()  # 10% random variation
                    else:
                        variation = 1.0

                    pdf_array[ix, iq] = central * variation

            pdf_values[flavor] = pdf_array

        # Write .dat file for this replica
        write_lhapdf_dat(output_dir, set_name, replica, x_grid, q_grid, pdf_values)

    print(f"\nSuccessfully generated {num_replicas + 1} replica files!")
    print(f"Files written to: {output_dir}")
    print(f"\nTo use with LHAPDF, copy the directory to your LHAPDF data path")


if __name__ == "__main__":
    # Example usage
    set_name = "MyPDFSet"
    num_replicas = 100  # Number of replicas (excluding central member)
    output_dir = f"./{set_name}"

    generate_pdf_replicas(set_name, num_replicas, output_dir)
