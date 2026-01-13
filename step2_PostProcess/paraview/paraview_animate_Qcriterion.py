# -------------------------------------------------------------------------------------------------------------------------------
# Script to perform parallel rendering of Q-criterion isosurfaces from Qcriterion timeseries HDF5 file.
#
#  __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - Reads time-series XDMF/HDF5 file storing the Qcriterion solution at each timestep.
#   - Extracts an isosurface at a chosen value.
#   - Renders off-screen and saves frames as PNGs or animation.
#
# REQUIREMENTS:
#   - paraview/6.0.0
#
# EXECUTION:
#   - Edit the DATA_DIR, XDMF_FILE, and OUTPUT_PATH variables below to match your simulation results and desired export location.
#   - To run the script on CCDB Trillium HPC in batch mode (paraview_parallel.sh):
#       >> module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 paraview/6.0.0
#       >> srun pvbatch --force-offscreen-rendering --opengl-window-backend OSMesa paraview_animate_Qcriterion.py
#
# Copyright (C) 2025 University of Toronto, by Rojin Anbarafshan (rojin.anbar@gmail.com).
# Original state file generated using paraview version 6.0.0.
# -------------------------------------------------------------------------------------------------------------------------------

__author__ = "Rojin Anbarafshan <rojin.anbar@gmail.com>"
__date__ = "October 2025"
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

ISOVALUE     = 0.1              # Q-criterion magnitude
SAVE_VIDEO   = True             # Save mode flag (True: saves video / False: saves png sequence)
FRAME_STRIDE = 1                # Number of frames to skip for export
FRAME_RATE   = 500              # Frame rate of the exported animation
START_FRAME  = 2000            # Frame to start saving animations
END_FRAME    = 29700            # Frame to end saving animations
BG_COLOR     = [0.1, 0.1, 0.1]  # Color of display background
FONT_SIZE    = 40               # Font size for text


# ----------------------------------- DEFINE PATHS ------------------------------------- #
# Directory that contains your .xdmf and linked .h5 files
BASE_DIR  = "/scratch/ranbar/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Qcriterion"
CASE_NAME = "PTSeg028_base_0p64"
DATA_DIR = f"{BASE_DIR}/cy6_saveFreq2"   

# Name of the .xdmf file you want to open
XDMF_FILE = "PTSeg028_base_0p64_Qcriterion.xdmf"                 
# Full path assembled
XDMF_PATH = f"{DATA_DIR}/{XDMF_FILE}"    

                              
# Path and filenames to save the animation (video or image sequence)
OUTPUT_PATH = f"{DATA_DIR}/Videos"
VIDEO_FILENAME  = f"Qcriterion_iso{ISOVALUE}_frames{START_FRAME}-{END_FRAME}_fs{FRAME_STRIDE}_fr{FRAME_RATE}.avi" #name of video file
IMG_FILENAME    = "Qcriterion_tstep.png"   #name of images


# Check if the defined paths exist
if not os.path.isfile(XDMF_PATH):
    raise FileNotFoundError(f"XDMF not found: {XDMF_PATH}")

if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/frames/", exist_ok=True)


# ------------------------------------ SETUP VIEWS ------------------------------------- #


# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView = CreateView('RenderView')
renderView.Set(
    ViewSize=[1496, 742],
    CenterOfRotation=[20.58187484741211, -35.115968227386475, -2.8254127502441406],
    CameraPosition=[117.08181711407107, -0.8379612465741169, 16.160909232291704],
    CameraFocalPoint=[16.953987550435635, -21.440809896761717, 3.7326431301954193],
    CameraViewUp=[-0.16677756699648133, 0.23251519571625529, 0.9581867912405001],
    CameraFocalDisk=1.0,
    CameraParallelScale=101.21371359744069,
    OSPRayMaterialLibrary=materialLibrary1,
    Background=BG_COLOR,
)

SetActiveView(None)


# Create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView)
layout1.SetSize(1496, 742)

# Restore active view
SetActiveView(renderView)


# ------------------------------------ READ DATA ------------------------------------- #


# Create 'XDMF Reader'
data_reader = XDMFReader(registrationName=XDMF_FILE, FileNames=[XDMF_PATH])

data_reader.Set(PointArrayStatus=['QCriterion'])


# ------------------------------------ SETUP PIPELINE ------------------------------------- #

# Create a 'Contour' on Q-criterion
contour_Q = Contour(registrationName='ContourQ', Input=data_reader)
contour_Q.Set(
    ContourBy=['POINTS', 'QCriterion'],
    Isosurfaces=[ISOVALUE],
)



# Create 'Annotate Time Filter'
annotateTimeFilter = AnnotateTimeFilter(registrationName='AnnotateTimeFilter', Input=contour_Q)
annotateTimeFilter.Format = 'Time (ms): {time:.3f}'


# ------------------------------------ SETUP DISPLAY ------------------------------------- 
# 1. DISPLAY GEOMETRY
# Show data from the xdmf file
dataDisplay = Show(data_reader, renderView, 'UnstructuredGridRepresentation')

# Set properties of the data display
dataDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', ''],
    Opacity=0.1,
)

# Init the 'Piecewise Function' selected for 'ScaleTransferFunction'
dataDisplay.ScaleTransferFunction.Points = [-8.011973841348663e-05, 0.0, 0.5, 0.0, 0.00025806663325056434, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
dataDisplay.OpacityTransferFunction.Points = [-8.011973841348663e-05, 0.0, 0.5, 0.0, 0.00025806663325056434, 1.0, 0.5, 0.0]


# 2. DISPLAY CONTOUR
contourDisplay = Show(contour_Q, renderView, 'GeometryRepresentation')

# Set properties of the contour display
contourDisplay.Set(
    Representation='Surface',
    AmbientColor=[0.0, 1.0, 1.0],
    ColorArrayName=['POINTS', ''],
    DiffuseColor=[0.0, 1.0, 1.0],
    SelectNormalArray='Normals',
)

# Init the 'Piecewise Function' selected for 'ScaleTransferFunction'
contourDisplay.ScaleTransferFunction.Points = [8.897345105651766e-05, 0.0, 0.5, 0.0, 8.898835221771151e-05, 1.0, 0.5, 0.0]

# Init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contourDisplay.OpacityTransferFunction.Points = [8.897345105651766e-05, 0.0, 0.5, 0.0, 8.898835221771151e-05, 1.0, 0.5, 0.0]


# 3. DISPLAY TIME
annotateTimeFilterDisplay = Show(annotateTimeFilter, renderView, 'TextSourceRepresentation')

# Set properties of the time annotation
annotateTimeFilterDisplay.Set(
    Color=[0.0, 0.0, 0.0],
    Bold=1,
    FontSize=FONT_SIZE,
)


# ------------------------------------ SETUP EXPORTS ------------------------------------- 
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


# Initialize the animation scene
animationScene.Set(
    ViewModules=renderView,
    #Cues=timeAnimationCue1,
    AnimationTime=tvals[START_FRAME], #tvals[0],
    StartTime=tvals[START_FRAME], #tvals[0],
    EndTime=tvals[END_FRAME],
    PlayMode='Snap To TimeSteps',
)

# restore active source
SetActiveSource(annotateTimeFilter)


if SAVE_VIDEO:
    # Option 1: write a VIDEO file (requires ParaView built with a video writer)
    # Use extension to choose format: .mp4 (FFmpeg), .avi, .ogv, etc.
    SaveAnimation(
        f"{OUTPUT_PATH}/{VIDEO_FILENAME}", renderView,
        FrameRate=FRAME_RATE,
        FrameWindow=[START_FRAME, END_FRAME],
        FrameStride=FRAME_STRIDE,
        Compression=True)

else: 
    # Option 2: write a PNG frame sequence
    SaveAnimation(
        f"{OUTPUT_PATH}/frames/{IMG_FILENAME}", renderView,
        FrameRate=FRAME_RATE,
        FrameWindow=[START_FRAME, END_FRAME], #[0, len(tvals)-1],
        FrameStride=FRAME_STRIDE)
