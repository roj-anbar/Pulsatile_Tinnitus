# --------------------------------------------------------------------------------------------------------------------
# Script to perform parallel, off-screen rendering of velocity-isosurfaces from time series CFD results.
#
# What it does:
# - Reads time-series XDMF/HDF5 files storing the CFD solution at each timestep.
# - Computes velocity magnitude.
# - Extracts a velocity isosurface at a chosen value.
# - Renders off-screen and saves frames as PNGs.
#
# Requirements:
# - vtk, mpi4py
#
# Usage:
# - sbatch velocity_isosurface.sh
#
# Copyright (C) 2025 University of Toronto, by Rojin Anbarafshan (rojin.anbar@gmail.com)
# ---------------------------------------------------------------------------------------------------------------------

__author__ = "Rojin Anbarafshan <rojin.anbar@gmail.com>"
__date__ = "September 2025"
__copyright__ = "Copyright (C) 2025 U of T"
__license__  = "Private"

import os
import vtk
import pyvista as pv

print(pv.__version__)

#from vtkmodules.vtkIOXdmf2Python import vtkXdmfReader
#from vtkmodules.vtkIOXdmf3 import vtkXdmf3Reader

# ---------------------------- USER PARAMETERS -------------------------------------
CFD_PATH            = "/scratch/ranbar/PT/PT_Ramp/unsteady/CFD/results_trillium/PTSeg028_base_0p64_ts10000_cy6_Q=2t/" # path to CFD results
XDMF_FILENAME       = "art_PTSeg028_base_0p64_I2_FC_VENOUS_Q557_Per915_Newt370_ts10000_cy6_uO1.xdmf"                  # name of XDMF file
VELOCITY_ARRAY_NAME = "u"             # name of the matrix storing velocity field
ISOVALUE            = 0.50            # isosurface value for |u|

# Output parameters
IMG_SIZE            = (1280, 720)     # (width, height) in pixels
BG_COLOR            = (1.0, 1.0, 1.0) # background RGB in [0,1]
SHADE_NORMALS       = False           # True = nicer shading (slower), False = faster
OUTPUT_DIR          = "frames"        # folder for PNG frames
# ---------------------------------------------------------------------------------



# Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Create XDMF reader
#def make_reader(xdmf_path):
#    r = vtk.vtkXdmf3Reader() # legacy reader due to file format
#    r.SetFileName(xdmf_path)
#    r.UpdateInformation()  # get time metadata without loading everything

#reader = make_reader(CFD_PATH + XDMF_FILENAME)
reader = pv.XdmfReader(CFD_PATH + XDMF_FILENAME)


# Get available timesteps (vtk version)
#info = reader.GetOutputInformation(0)
#timestep_key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
#timesteps = info.Get(timestep_key)


# Get available timesteps (pyvista version)
timesteps = reader.time_values
print(f"Found {len(timesteps)} timesteps. \n")

reader.set_active_time_value(timesteps[0])
mesh0 = reader.read()
print(mesh0.array_names)




