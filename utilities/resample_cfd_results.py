#!/usr/bin/env python3
"""
Resample CFD results folder: keep every nth timestep and rewrite the XDMF.

Creates a new output directory with symlinks to the selected HDF5 files
(no copying) and writes a new XDMF that ParaView can open directly.

Usage:
    python resample_cfd_results.py <xdmf_file> <output_dir> <n> [--start S]

Example (every 10th step):
    python resample_cfd_results.py results/run.xdmf results/run_every10 10
"""

import argparse
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample CFD results to every nth timestep"
    )
    parser.add_argument("xdmf", help="Path to the source XDMF file")
    parser.add_argument("output_dir", help="Output directory for resampled results")
    parser.add_argument("n", type=int, help="Keep every nth timestep (e.g. 10)")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="S",
        help="Index into the timestep list to start from (default: 0)",
    )
    return parser.parse_args()


def extract_h5_filenames(text):
    """Return all unique HDF5 filenames referenced in an XML text snippet."""
    return set(re.findall(r'([\w=.+-]+\.h5):', text))


def main():
    args = parse_args()

    xdmf_path = Path(args.xdmf).resolve()
    src_dir = xdmf_path.parent
    out_dir = Path(args.output_dir).resolve()
    n = args.n
    start = args.start

    if not xdmf_path.exists():
        sys.exit(f"Error: XDMF file not found: {xdmf_path}")
    if n < 1:
        sys.exit("Error: n must be >= 1")

    print(f"Reading {xdmf_path} ...")
    with open(xdmf_path, "r") as f:
        content = f.read()

    # ------------------------------------------------------------------ #
    # Extract the mesh <Grid> block (topology + geometry, no timestep data)
    # ------------------------------------------------------------------ #
    mesh_match = re.search(
        r'(<Grid Name="mesh"[^>]*>.*?</Grid>)', content, re.DOTALL
    )
    if not mesh_match:
        sys.exit("Error: could not find <Grid Name=\"mesh\"> in XDMF")
    mesh_block = mesh_match.group(1)
    mesh_h5s = extract_h5_filenames(mesh_block)

    # ------------------------------------------------------------------ #
    # Extract every timestep <Grid Name="Step-..."> block
    # ------------------------------------------------------------------ #
    step_blocks = re.findall(
        r'(<Grid Name="Step-\d+"[^>]*>.*?</Grid>)', content, re.DOTALL
    )
    if not step_blocks:
        sys.exit("Error: no Step grids found in XDMF")

    total = len(step_blocks)
    selected_blocks = step_blocks[start::n]
    kept = len(selected_blocks)

    print(f"Total timesteps : {total}")
    print(f"Start index     : {start}")
    print(f"Every nth       : {n}")
    print(f"Kept timesteps  : {kept}")

    # ------------------------------------------------------------------ #
    # Collect HDF5 files needed for the selected timesteps
    # ------------------------------------------------------------------ #
    step_h5s = set()
    for block in selected_blocks:
        step_h5s |= extract_h5_filenames(block)

    all_h5s = mesh_h5s | step_h5s

    # ------------------------------------------------------------------ #
    # Create output directory and symlinks
    # ------------------------------------------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for h5file in sorted(all_h5s):
        src = src_dir / h5file
        dst = out_dir / h5file
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if src.exists():
            os.symlink(src, dst)
        else:
            missing.append(h5file)
            print(f"  WARNING: source file not found: {src}")

    print(f"Symlinked {len(all_h5s) - len(missing)} HDF5 file(s) into {out_dir}")

    # ------------------------------------------------------------------ #
    # Write new XDMF
    # ------------------------------------------------------------------ #
    # Name it after the original but with the stride appended
    out_xdmf_name = xdmf_path.stem + f"_every{n}" + xdmf_path.suffix
    out_xdmf = out_dir / out_xdmf_name

    with open(out_xdmf, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write('<Domain>\n')
        f.write('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
        f.write(f'        {mesh_block}\n')
        for block in selected_blocks:
            f.write(f'      {block}\n')
        f.write('    </Grid>\n')
        f.write('</Domain>\n')
        f.write('</Xdmf>\n')

    print(f"Wrote {out_xdmf}")
    if missing:
        print(f"\nWarning: {len(missing)} HDF5 file(s) were not found in {src_dir}.")


if __name__ == "__main__":
    main()
