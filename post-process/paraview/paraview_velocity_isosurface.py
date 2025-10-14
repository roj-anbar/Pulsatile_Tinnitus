# -------------------------------------------------------------------------------------------------------------------------------
# Script to perform parallel, off-screen rendering of velocity-isosurfaces from time series CFD results.
#
# Funcionality:
#   - Reads time-series XDMF/HDF5 files storing the CFD solution at each timestep.
#   - Computes velocity magnitude.
#   - Extracts a velocity isosurface at a chosen value.
#   - Renders off-screen and saves frames as PNGs or animation.
#
# Requirements:
#   - paraview/6.0.0
#
# Usage:
#   - Edit the DATA_DIR, XDMF_FILE, and OUTPUT_PATH variables below to match your simulation results and desired export location.
#   - To run the script on CCDB Trillium HPC in batch mode (paraview_parallel.sh):
#       >> module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 paraview/6.0.0
#       >> srun pvbatch --force-offscreen-rendering --opengl-window-backend OSMesa paraview_velocity_isosurface.py
#
# Notes:
#   - The script assumes your data contains a vector array "u" and will
#     contour its magnitude. Adjust ContourBy if needed.
#   - Paths are hard-coded below; for different cases, only edit variables at the top of the file.
#   - Ray tracing is disabled in this script.
#
# Copyright (C) 2025 University of Toronto, by Rojin Anbarafshan (rojin.anbar@gmail.com)
# Original state file generated using paraview version 6.0.0
# -------------------------------------------------------------------------------------------------------------------------------

__author__ = "Rojin Anbarafshan <rojin.anbar@gmail.com>"
__date__ = "September 2025"
__copyright__ = "Copyright (C) 2025 U of T"
__license__  = "Private"


import os
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

# Import the simple module from the paraview
from paraview.simple import *

# Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

print('Start setting the params...\n', flush=True)

# ----------------------------------- USER PARAMETERS ------------------------------------- #
# Directory that contains your .xdmf and linked .h5 files
DATA_DIR = "/scratch/ranbar/PT/PT_Ramp/steady/CFD/results_trillium/PTSeg028_base_0p64_ts10000_cy2_Q=5"
#steady: "/scratch/ranbar/PT/PT_Ramp/steady/CFD/results_trillium/PTSeg028_base_0p64_ts10000_cy2_Q=5"
#unsteady: "/scratch/ranbar/PT/PT_Ramp/unsteady/CFD/results_trillium/PTSeg028_base_0p64_ts10000_cy6_Q=2t"

# Name of the .xdmf file you want to open
XDMF_FILE = "art_PTSeg028_base_0p64_I2_FC_VENOUS_Q557_Per915_Newt370_ts10000_cy2_uO1.xdmf"
#steady: "art_PTSeg028_base_0p64_I2_FC_VENOUS_Q557_Per915_Newt370_ts10000_cy2_uO1.xdmf"
#unsteady: "art_PTSeg028_base_0p64_I2_FC_VENOUS_Q557_Per915_Newt370_ts10000_cy6_uO1.xdmf"

# Full path assembled
XDMF_PATH = f"{DATA_DIR}/{XDMF_FILE}"

# Where to save the animation (video or image sequence)
OUTPUT_PATH = "/scratch/ranbar/PT/PT_Ramp/steady/post-process/PTSeg028_base_0p64_ts10000_cy2_Q=5"
VIDEO_FILE  = "velocity_iso.avi" #for video
IMG_FILE    = "velocity_iso_tstep.png" #for images

# Save mode flag (True: saves video / False: saves png sequence)
SAVE_VIDEO = True 

ISOVALUE = 1              # velocity magnitude (m/s)
BG_COLOR = [0.1, 0.1, 0.1]  # color of display background
FONT_SIZE = 30              # font size for text


# Check if the defined paths exist
if not os.path.isfile(XDMF_PATH):
    raise FileNotFoundError(f"XDMF not found: {XDMF_PATH}")

if not os.path.isdir(OUTPUT_PATH):
    raise FileNotFoundError(f"Output folder not found: {OUTPUT_PATH}")


# ------------------------------------ SETUP VIEWS ------------------------------------- #

print('Start creating the render view...\n', flush=True)
# Get the material library
#materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView = CreateView('RenderView')
renderView.Set(
    ViewSize=[1371, 744],
    CenterOfRotation=[20.58187484741211, -35.11595106124878, -2.8254013061523438],
    CameraPosition=[339.64243904438194, -34.20444434275868, 48.6645497344558],
    CameraFocalPoint=[20.58187484741211, -35.11595106124878, -2.8254013061523438],
    CameraViewUp=[-0.15689133930142712, 0.19191862205963262, 0.9687891154213782],
    CameraFocalDisk=1.0,
    CameraParallelScale=101.21370795192328,
    EnableRayTracing=0,
    #BackEnd='OSPRay raycaster',
    #OSPRayMaterialLibrary=materialLibrary1,
    Background=BG_COLOR,
)

SetActiveView(None)


# Create new layout object 'Layout #1'
layout = CreateLayout(name='Layout #1')
layout.AssignView(0, renderView)
layout.SetSize(1371, 744)


# Restore active view
SetActiveView(renderView)


# ------------------------------------ READ DATA ------------------------------------- #

# Create 'XDMF Reader'
print('Start reading the data...\n', flush=True)
data_reader = XDMFReader(registrationName=XDMF_FILE, FileNames=[XDMF_PATH])

data_reader.Set(
    PointArrayStatus=['p', 'u'],
    # If you want to read a subset of timesteps define it in GridStatus
    #GridStatus=['mesh', 'Step-000100', 'Step-001000', 'Step-005000', 'Step-010000', 'Step-015000', 'Step-020000', 'Step-025000'],
)

print('Finished reading the data...\n', flush=True)

# ------------------------------------ SETUP PIPELINE ------------------------------------- #

# Create 'calculator' for velocity magnitude
calc = Calculator(Input=data_reader)
calc.Set(
    ResultArrayName='u_mag',
    Function='mag(u)',
)


# Create a 'Contour' on velocity magnitude
contour_velocity = Contour(registrationName='Contour1', Input=calc)
contour_velocity.Set(
    ContourBy=['POINTS', 'u_mag'],
    Isosurfaces=[ISOVALUE],
)

# Create 'Annotate Time Filter'
annotateTimeFilter = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=contour_velocity)
annotateTimeFilter.Format = 'Time (ms): {time:.3f}'

print('Finished the processing...\n', flush=True)

# ------------------------------------ SETUP DISPLAY ------------------------------------- #

# 1. DISPLAY GEOMETRY
# Show data from the xdmf file
dataDisplay = Show(data_reader, renderView, 'UnstructuredGridRepresentation')

# Set properties of the data display
dataDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', ''],
    Opacity=0.2,
)

# Init the 'Piecewise Function' selected for 'ScaleTransferFunction'
dataDisplay.ScaleTransferFunction.Points = [-0.5512140393257141, 0.0, 0.5, 0.0, 2.1777005195617676, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
dataDisplay.OpacityTransferFunction.Points = [-0.5512140393257141, 0.0, 0.5, 0.0, 2.1777005195617676, 1.0, 0.5, 0.0]


# 2. DISPLAY CONTOUR
contourDisplay = Show(contour_velocity, renderView, 'GeometryRepresentation')

# Set properties of the contour display
contourDisplay.Set(
    Representation='Surface',
    AmbientColor=[0.0, 1.0, 1.0],
    ColorArrayName=['POINTS', ''],
    DiffuseColor=[0.0, 1.0, 1.0],
    Interpolation='Flat',
    Specular=1.0,
    Ambient=0.08,
    SelectNormalArray='Normals',
)

# Init the 'Piecewise Function' selected for 'ScaleTransferFunction'
contourDisplay.ScaleTransferFunction.Points = [0.8132432401180267, 0.0, 0.5, 0.0, 0.8133653402328491, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contourDisplay.OpacityTransferFunction.Points = [0.8132432401180267, 0.0, 0.5, 0.0, 0.8133653402328491, 1.0, 0.5, 0.0]



# 3. DISPLAY TIME
annotateTimeFilterDisplay = Show(annotateTimeFilter, renderView, 'TextSourceRepresentation')

# Set properties of the time annotation
annotateTimeFilterDisplay.Set(
    FontSize=FONT_SIZE,
    Bold=1,
    Color=[1.0, 1.0, 0.5],
)


# ------------------------------------ SETUP EXPORTS ------------------------------------- #

# Setup animation scene, tracks and keyframes

# get the time-keeper
#timeKeeper1 = GetTimeKeeper()

# get time animation track
#timeAnimationCue1 = GetTimeTrack()



# get animation scene
animationScene = GetAnimationScene()

# Discover available timesteps from the reader
tvals = list(data_reader.TimestepValues)

# initialize the animation scene
animationScene.Set(
    ViewModules=renderView,
    #Cues=timeAnimationCue1,
    AnimationTime=tvals[0],
    StartTime=tvals[0],
    EndTime=tvals[-1],
    PlayMode='Snap To TimeSteps',
)


# restore active source
SetActiveSource(annotateTimeFilter)


# Ensure first render has happened (safe on pvpython/pvbatch)
RenderAllViews()


print('Starting to save the animation...\n')

if SAVE_VIDEO:
    # Option 1: write a VIDEO file (requires ParaView built with a video writer)
    # Use extension to choose format: .mp4 (FFmpeg), .avi, .ogv, etc.
    SaveAnimation(
        f"{OUTPUT_PATH}/{VIDEO_FILE}", renderView,
        FrameRate=30,
        FrameWindow=[0, len(tvals)-1],
        FrameStride=20,
        Compression=True)

else: 
    # Option 2: write a PNG frame sequence
    SaveAnimation(
        f"{OUTPUT_PATH}/frames/{IMG_FILE}", renderView,
        FrameRate=30,
        FrameWindow=[0, len(tvals)-1], #[0, len(tvals)-1]
        FrameStride=6000)
