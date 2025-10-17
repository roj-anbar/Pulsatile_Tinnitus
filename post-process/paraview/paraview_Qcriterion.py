# -------------------------------------------------------------------------------------------------------------------------------
# Script to perform parallel, off-screen rendering of Q-criterion isosurfaces from time series CFD results.
#
# Funcionality:
#   - Reads time-series XDMF/HDF5 files storing the CFD solution at each timestep.
#   - Computes Q-criterion.
#   - Extracts an isosurface at a chosen value.
#   - Renders off-screen and saves frames as PNGs or animation.
#
# Requirements:
#   - paraview/6.0.0
#
# Execution:
#   - Edit the DATA_DIR, XDMF_FILE, and OUTPUT_PATH variables below to match your simulation results and desired export location.
#   - To run the script on CCDB Trillium HPC in batch mode (paraview_parallel.sh):
#       >> module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 paraview/6.0.0
#       >> srun pvbatch --force-offscreen-rendering --opengl-window-backend OSMesa paraview_velocity_isosurface.py
#
# Notes:
#   - The script assumes your data contains a vector array "u".
#   - Paths are hard-coded below; for different cases, only edit variables at the top of the file.
#   - Ray tracing is disabled in this script.
#
# Copyright (C) 2025 University of Toronto, by Rojin Anbarafshan (rojin.anbar@gmail.com).
# Original state file generated using paraview version 6.0.0.
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


# ----------------------------------- USER PARAMETERS ------------------------------------- #

ISOVALUE     = 0.15             # Q-criterion magnitude
SAVE_VIDEO   = True             # Save mode flag (True: saves video / False: saves png sequence)
FRAME_STRIDE = 1                # Number of frames to skip for export
FRAME_RATE   = 100              # Frame rate of the exported animation
START_FRAME  = 50000            # Frame to start saving animations
END_FRAME    = 52000            # Frame to end saving animations
BG_COLOR     = [0.1, 0.1, 0.1]  # Color of display background
FONT_SIZE    = 30               # Font size for text



# ----------------------------------- DEFINE PATHS ------------------------------------- #
# Directory that contains your .xdmf and linked .h5 files
DATA_DIR = "/scratch/ranbar/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq2"   

# Name of the .xdmf file you want to open
XDMF_FILE = "PTSeg028_base_0p64_ts10000_cy6.xdmf"                 

                              
# Path and filenames to save the animation (video or image sequence)
OUTPUT_PATH = "/scratch/ranbar/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Videos"
VIDEO_FILE  = f"Qcriterion_ISO{ISOVALUE}_FRAMES{START_FRAME}-{END_FRAME}_FS{FRAME_STRIDE}_FR{FRAME_RATE}.avi" #name of video file
IMG_FILE    = "Qcriterion_tstep.png"   #name of images

# Full path assembled
XDMF_PATH = f"{DATA_DIR}/{XDMF_FILE}"    

# Check if the defined paths exist
if not os.path.isfile(XDMF_PATH):
    raise FileNotFoundError(f"XDMF not found: {XDMF_PATH}")

if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/frames/", exist_ok=True)




# ------------------------------------ SETUP VIEWS ------------------------------------- #

# Get the material library
#materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView = CreateView('RenderView')
renderView.Set(
    ViewSize=[1425, 744],
    CenterOfRotation=[20.58187484741211, -35.11595106124878, -2.8254013061523438],
    CameraPosition=[93.00126808053199, 22.424864515154955, 43.291897969863065],
    CameraFocalPoint=[23.515646717080337, -31.46568863342259, -10.29986395308911],
    CameraViewUp=[-0.703295829476546, 0.24213336989229686, 0.6683909091433676],
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
layout.SetSize(1425, 744)


# Restore active view
SetActiveView(renderView)


# ------------------------------------ READ DATA ------------------------------------- #

# Create 'XDMF Reader'
data_reader = XDMFReader(registrationName=XDMF_FILE, FileNames=[XDMF_PATH])

data_reader.Set(
    PointArrayStatus=['p', 'u'],
    # If you want to read a subset of timesteps define it in GridStatus
    #GridStatus=['mesh', 'Step-000100', 'Step-001000', 'Step-005000', 'Step-010000', 'Step-015000', 'Step-020000', 'Step-025000'],
)

#print('Finished reading the data...\n')

# ------------------------------------ SETUP PIPELINE ------------------------------------- #

# Create a 'Gradient' filter
gradient = Gradient(registrationName='Gradient1', Input=data_reader)
gradient.Set(
    ScalarArray=['POINTS', 'u'],
    ComputeGradient=0,
    ComputeQCriterion=1,
)


# Create a 'Contour' on Q-criterion
contour_Q = Contour(registrationName='Contour1', Input=gradient)
contour_Q.Set(
    ContourBy=['POINTS', 'Q Criterion'],
    Isosurfaces=[ISOVALUE],
)

# Create 'Annotate Time Filter'
annotateTimeFilter = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=contour_Q)
annotateTimeFilter.Format = 'Time (ms): {time:.3f}'



#print('Finished the processing...\n')


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
dataDisplay.ScaleTransferFunction.Points = [-2.1082217693328857, 0.0, 0.5, 0.0, 7.2338128089904785, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
dataDisplay.OpacityTransferFunction.Points = [-2.1082217693328857, 0.0, 0.5, 0.0, 7.2338128089904785, 1.0, 0.5, 0.0]


# 2. DISPLAY CONTOUR
contourDisplay = Show(contour_Q, renderView, 'GeometryRepresentation')

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
contourDisplay.ScaleTransferFunction.Points = [2.562795639038086, 0.0, 0.5, 0.0, 2.563283920288086, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contourDisplay.OpacityTransferFunction.Points = [2.562795639038086, 0.0, 0.5, 0.0, 2.563283920288086, 1.0, 0.5, 0.0]


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
# note: the Get..() functions create a new object, if needed

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
    AnimationTime=tvals[START_FRAME], #tvals[0],
    StartTime=tvals[START_FRAME], #tvals[0],
    EndTime=tvals[END_FRAME], #tvals[-1],
    PlayMode='Snap To TimeSteps',
)


# restore active source
SetActiveSource(annotateTimeFilter)


# Ensure first render has happened (safe on pvpython/pvbatch)
RenderAllViews()


#print(f"Found {len(tvals)} timesteps. Starting export...", flush=True)

if SAVE_VIDEO:
    # Option 1: write a VIDEO file (requires ParaView built with a video writer)
    # Use extension to choose format: .mp4 (FFmpeg), .avi, .ogv, etc.
    SaveAnimation(
        f"{OUTPUT_PATH}/{VIDEO_FILE}", renderView,
        FrameRate=FRAME_RATE,
        FrameWindow=[START_FRAME, END_FRAME],
        FrameStride=FRAME_STRIDE,
        Compression=True)

else: 
    # Option 2: write a PNG frame sequence
    SaveAnimation(
        f"{OUTPUT_PATH}/frames/{IMG_FILE}", renderView,
        FrameRate=FRAME_RATE,
        FrameWindow=[START_FRAME, END_FRAME], #[0, len(tvals)-1],
        FrameStride=FRAME_STRIDE)
