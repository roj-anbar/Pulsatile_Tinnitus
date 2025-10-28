"""
To convert HDF5 timestep files into a single XDMF time‐series for visualization in ParaView.

This script:
  1. Finds all .h5 files matching a pattern and sorts them by timestep.
  2. Reads the mesh geometry (points + topology) from the first file.
  3. Writes out a single `output.xdmf` containing all point and cell data at each timestep, so it can be opened in ParaView.

Usage:
  python h5_to_xdmf.py

Author: Rojin Anbarafshan
Contact: rojin.anbar@gmail.com
Date: July 2025
"""

import glob
import meshio


# Parameters
h5_dir     = "/project/s/steinman/ranbar/Swirl/swirl_files/PTSeg028/"
h5_pattern = h5_dir + "Swirl_*.h5"     # name of input files
dt         = 13               # time steps between each two consecutive h5 files

# 1) Collect and sort .h5 files
h5_files = sorted(glob.glob(h5_pattern))

# 2) Read mesh (points + cells) from the first file
first_mesh = meshio.read(h5_files[0])
points, cells = first_mesh.points, first_mesh.cells

# 3) Write XDMF time series
with meshio.xdmf.TimeSeriesWriter(h5_dir + "Swirl_PTSeg028.xdmf", points, cells) as ts:
    for i, fname in enumerate(h5_files):
        mesh = meshio.read(fname)
        time = i * dt
        ts.write_data(
            point_data=mesh.point_data,
            cell_data= mesh.cell_data,
            time      = time
        )
