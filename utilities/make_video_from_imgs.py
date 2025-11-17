import os
import re
import glob
import numpy as np
import cv2 #openCV library for video generation
#from PIL import Image


# ------------------------------- Set Parameters ------------------------------
BASE_DIR     = "/Users/rojin/Library/CloudStorage/OneDrive-UniversityofToronto/Education/PhD/Research/Projects/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Spectrogram_wall_pressure/cy6_saveFreq1/window5000_overlap0.9_ROIsphere"
INPUT_PATH   = BASE_DIR + "/imgs/*.png"        # Path to the image sequence
OUTPUT_PATH  = BASE_DIR + "/spectrogram.mp4"   # Path to the output video
FRAME_RATE   = 5                                 # Frames per second
RESIZE_DIM   = None                              # (width, height) or None to keep original size
CODEC        = "mp4v"                            # "mp4v" is broadly compatible; try "avc1" on macOS if needed


# ------------------------------- Filename Utilities ------------------------------
def sort_key_ROI(path: str):
    """Function to sort the files based on ROI indices: For spectrograms we want the ROIs with higher indices first."""
    filename = os.path.basename(path) # find the filename from the path
    m =  re.search(r'_ROI(\d+)_', filename) #m.group(1) would the ROI index
    
    if m:
        ROI_tag = int(m.group(1))
        # negative for descending; tie-break on full name to keep sort stable
        return -ROI_tag



# ------------------------------- Generate Videos ------------------------------
def make_video(input_path, output_path, frame_rate, resize_dim, codec):

    # Sort the images
    files = sorted(glob.glob(input_path), key=sort_key_ROI)

    if not files:
        raise FileNotFoundError(f"No images found with the specified pattern in {input_path}!")

    # Read first frame to get size
    img0 = cv2.imread(files[0], cv2.IMREAD_COLOR)
    #img0 = Image.open(files[0]).convert("RGB")

    #if resize_dim is not None:
    #    img0 = img0.resize(resize, Image.LANCZOS)
    
    h, w = img0.shape[:2]

    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))

    # Write first frame
    #frame0 = cv2.cvtColor(np_from_pillow(im0), cv2.COLOR_RGB2BGR)
    out.write(img0)

    # Write remaining frames
    for f in files[1:]:
        img = cv2.imread(f, cv2.IMREAD_COLOR)

        out.write(img)

    out.release()
    print(f"Saved video of {len(files)} frames at {frame_rate} fps and size {w}x{h}) to: \n {os.path.abspath(output_path)}  ")



if __name__ == "__main__":
    make_video(INPUT_PATH, OUTPUT_PATH, FRAME_RATE, resize_dim=RESIZE_DIM, codec=CODEC)