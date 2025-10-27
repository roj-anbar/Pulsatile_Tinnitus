"""
Generate pressure or velocity spectrograms for desired surfaces using BSL tools (from bsl.dataset library).
Important: Adjust line 34 accordingly.
Modified by Rojin Anbarafshan from original script written by Anna Haley
Date: January 2025
Contact: rojin.anbar@gmail.com


INPUTS:  1) Paths to the desired file given from comman line (specified in the job script) 

OUTPUTS: 1) Processed spectrograms saved as .npz
         2) Spectrogram images saved as .png

         
"""

### -------- INITIALIZE -------- ###
import sys
import os
from pathlib import Path

# Set the directory for Matplotlib configuration to avoid conflicts
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ranbar/.config/mpl'
os.environ['PYVISTA_USERDATA_PATH'] = '/scratch/s/steinman/ranbar/.local/share/pyvista'


import numpy as np 
import pyvista as pv
import matplotlib.pyplot as plt
from bsl.dataset import Dataset  # all the spectrogram calculations are based on functions from this library
#from scipy.spatial import cKDTree as KDTree #Efficient data structure for nearest-neighbor searches

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, meshfolder=None, file_stride=1, mesh_glob_key=None):

        self.folder = Path(folder)
        
        if mesh_glob_key is None:
            mesh_glob_key = '*h5'

        keyl = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]
        self.swirl_files = sorted(Path(folder).glob('Swirl_*.h5'), key=keyl)
        #print(self.swirl_files)  

        self.mesh_file=list(Path(meshfolder).glob('*.h5'))[0]

        self.arrays = {}
        self.spectrogram_data = {}

    def __call__(self, idx, array='u_pAvg'):
        h5_file = self.swirl_files[idx]    
        with h5py.File(h5_file, 'r') as hf:
            val = np.array(hf[array])
        return val
        
    def assemble_mesh(self):
        """ Create UnstructuredGrid from h5 mesh file. """
        
        with h5py.File(self.mesh_file, 'r') as hf:
            points = np.array(hf['Mesh']['coordinates'])
            cells = np.array(hf['Mesh']['topology'])

            celltypes = np.empty(cells.shape[0], dtype=np.uint8)
            celltypes[:] = vtk.VTK_TETRA

            cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
            cells = np.concatenate([cell_type, cells], axis = 1)
            self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
            self.surf = self.mesh.extract_surface()
    
        # self.assemble_surface()
        return self

    def assemble_matrix(self, array_key='u_p', quantity='u_p', array=None, mask=None):
        """ Create a N * ts matrix of scalar u_mag or p data.

        Args:
            array (array or None): Supplying array will overwrite array.
            ind (array or None): Indicies for a subset.

        Used for spectrograms.
        """
        self.arrays[array_key] = np.zeros((self.mesh.n_points, len(self.swirl_files)), dtype=np.float64)
        # Get indices of mask
        ind = np.where(self.mesh.point_arrays[mask])
        self.arrays[array_key] = self.arrays[array_key][ind]
        key = quantity        

        for idx in range(len(self.swirl_files)):
            if idx % 100 == 0:
                print(idx, '/', len(self.swirl_files))

            arr = self(idx, array=key)[ind]
            self.arrays[array_key][:,idx] = arr.reshape((-1,))
        
        return self
    
    def spectrogram(self, array_key, n_fft=None, period=0.915):
        """ Compute spectrogram from an array, usually u_mag.

        Args:
            array (array or None): Array containing u magnitude or QoI.
            indices (list or None): Location indices of, for example, sac.
            spec_file (path or None): Where to save spectrogram data.
            spec_img_file (path or None): Path to save spectrogram img.
        """
        array = self.arrays[array_key]
        n_samples = array.shape[1]
        sr = array.shape[1] / period 

        if n_fft is None:
            n_fft = spectral.shift_bit_length(int(n_samples / 10))
        
        spec_args = {}
        spec_args['sr'] = sr
        spec_args['n_fft'] = n_fft
        spec_args['hop_length'] = int(0.25*n_fft)
        spec_args['win_length'] = n_fft
        spec_args['detrend'] = 'linear'
        spec_args['pad_mode'] = 'cycle'

        S, bins, freqs = spectral.average_spectrogram(
            data=array, 
            **spec_args
            )
            
        # Remove last frame
        S = S[:,:-1]
        bins = bins[:-1]

        spec_data = {}
        spec_data['S'] = S
        spec_data['bins'] = bins
        spec_data['freqs'] = freqs
        spec_data['sr'] = sr 
        spec_data['n_fft'] = n_fft

        self.spectrogram_data[array_key] = spec_data

        return self



### -------- Define Parameters -------- ###
# Choose quantity to sonify or build the spectrogram based on (choose between 'velocity' or 'pressure')
spec_quantity = 'pressure'

# Define ROI Properties
point_ID = 519235 # ID of the point of interest (center of sphere) -> obtain from mesh in paraview
radius = 2      # radius of sphere (same units as the mesh [mm])


# Define fluid properties (used for converting oasis pressure to actual pressure)
density = 1 #1050 #[kg/m3]




### -------- COMPUTE SPECTROGRAMS -------- ###

# Add the if to ensure that this script runs only when executed directly
if __name__ == "__main__":

    ### ----- I/O PATHS ------ ###
    # Determine paths for input/output files from the command-line argument
    
    # To read inputs
    folder = Path(sys.argv[1]) #results folder eg. results/art_
    case_name = sys.argv[2] #eg. PTSeg028_low
    
    # To store outputs
    spec_data_out_folder = Path(sys.argv[3]) #eg. spec_data
    spec_img_out_folder  = Path(sys.argv[4]) #eg spec_imgs
    
    # Sample data by selecting only every nth file
    stride = int(sys.argv[5])


    # Reading the data
    dd = Dataset(folder, file_stride=stride, mesh_glob_key='*.h5')#, case_name = case_name)
    # Extract and organize mesh data from the dataset (gets mesh info from results/art_/PT_Seg028_low.h5)
    dd = dd.assemble_mesh() 

    # Creates the output filename (npz format: numpy zipped file)
    spec_out_file = spec_data_out_folder / f"{case_name}_{spec_quantity}_ID={point_ID}_r={radius}.npz" #eg. specs/PTSeg028_low_pressure_ID=9301_r=0.1.npz
    

    # Create the spherical ROI (using pyvista)
    center = dd.mesh.points[point_ID] # return coordinates of the desired point_ID
    ROI_sphere = pv.Sphere(radius = radius, center = center) # creates a 2d sphere around desired point (units of the radius same as units of the mesh)
    
    # Save the sphere to a .vtp file (for visualization in paraview later)
    ROI_sphere.save(f'{spec_data_out_folder}/ROI_sphere_ID={point_ID}_r={radius}.vtp') 

    # Selects mesh points inside the surface, with a certain tolerance (using pyvista)
    mesh_sel = dd.mesh.select_enclosed_points(ROI_sphere, tolerance=0.01)
    dd.mesh = mesh_sel


    # Assembles data for the selected mesh points
    # array_key and quantity can be set to 'p' or 'umag'

    if spec_quantity == 'pressure':
        
        dd.assemble_matrix(array_key='p', quantity='p', mask='SelectedPoints') #SelectedPoints is the output of 'select_enclosed_points' function
        
        # Multiply all pressures by density (since oasis return p/rho)
        dd.arrays['p'] *= density

        # Generate Spectrograms
        dd.spectrogram(array_key='p')
        spec_data = dd.spectrogram_data['p']
        np.savez(spec_out_file, **dd.spectrogram_data['p'])

    elif spec_quantity == 'velocity':
        # the variable is 'u' but bsl tools calculate the norm of it -> 'umag'
        dd.assemble_matrix(array_key='u_mag', quantity='u_mag', mask='SelectedPoints')

        # Generate Spectrograms
        dd.spectrogram(array_key='u_mag')
        spec_data = dd.spectrogram_data['u_mag']
        np.savez(spec_out_file, **dd.spectrogram_data['u_mag']) 

    # Extract relevant data for plotting
    bins = spec_data['bins']
    freqs = spec_data['freqs']
    S = spec_data['S']

    # Clamp values below -20dB
    #S[S < -20] = -20 


    ### ----------------- PLOTTING ------------------ ###


    # Setting plot properties
    size = 10
    plt.rc('font', size=size) #controls default text size
    plt.rc('axes', titlesize=12) #fontsize of the title
    plt.rc('axes', labelsize=size) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
    plt.rc('legend', fontsize=size) #fontsize of the legend


    fig, ax = plt.subplots(1,1, figsize=(6,4))
    spectrogram = ax.pcolormesh(bins, freqs, S, shading='gouraud')
    #spectrogram.set_clim([-20, 0])
    ax.set_xlabel('Time (s)', labelpad=-5)
    ax.set_ylabel('Freq (Hz)', labelpad=-10)
    ax.set_xticks([0, 0.9])
    ax.set_xticklabels(['0.0', '0.9'])
    ax.set_yticks([0, 600,800])
    ax.set_yticklabels(['0', '600', '800'])
    ax.set_ylim([0, 800])


    title = f'{case_name}_{spec_quantity}_pointID={point_ID}_radius={radius}' 

    ax.set_title(title)
    plt.tight_layout
    plt.colorbar(spectrogram, ax=ax) # Adding the colorbar
    plt.savefig(spec_img_out_folder / (title + '.png'))#, transparent=True)




