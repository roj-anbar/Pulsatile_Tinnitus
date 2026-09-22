# -----------------------------------------------------------------------------------------------------------------------
# oasis_problem_PT.py 
# This is the problem definition for Oasis/NSfracStep.py (patient-specific arteries/veins).
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-09
#
# PURPOSE:
#   - To provide case specific problem parameters (e.g., mesh loading, boundary conditions, time hooks) for Oasis NSfracStep solver (legacy FEniCS).
#
# REQUIREMENTS:
#   - Oasis package available locally: https://github.com/mikaem/Oasis.git
#   - BSLSolver available locally: https://github.com/Biomedical-Simulation-Lab/BSLSolver.git
#
# EXECUTION:
#   - This script is used as a problem file when calling the Oasis solver (as in the oasis_solver_PT.sh).
#   - <mpirun -n $NP oasis NSfracStep problem=oasis_problem_PT>.
#   - Note: This script cannot be ran directly.
#
# INPUTS:
#   - mesh_name            : basename of mesh in ./data (expects .xml.gz and .info)
#   - period               : waveform period [ms]
#   - timesteps            : time steps per cycle
#   - cycles               : number of cycles
#   - viscosity_mu_Pas     : dynamic viscosity [Pa·s]     (default: 0.0037)
#   - density_kgm3         : fluid density [kg/m³]        (default: 1057)
#   - uOrder               : velocity polynomial order for FE
#   - save_frequency       : save every N steps
#   - checkpoint           : write restart every N steps
#   - inlet_BC_type        : type of the inlet boundary condition --> choose from: {'pulsatile', 'ramp', 'constant'} (default is 'pulsatile')
#
# OPTIONAL:
#   - restart_folder           : path to a previous results folder to restart from
#   - zero_pressure_outlets    : set True to enforce p=0 at all outlets (default: False)
#   - save_first_cycle         : set True to also save the spin-up cycle (default: False)
#   - flat_profile_at_intlet_bc: set True for plug/flat inlet profile (default: False)
#   - inflowrate_constant_mLs  : constant inflow rate [mL/s], used when inlet_BC_type='constant' (default: 5.0)
#
# OUTPUTS:
#   - Results written under ./results/{case_fullname}/
#   - Per-step velocity and pressure in HDF5 + XDMF format via BSLSolver.common.h5io
#
# NOTES:
#   - Units in comments follow Oasis code: mm, ms, mL/s (consistent with FEniCS fields).
#   - Keep behavior compatible with existing postprocessing.
#
# Adapted from Artery.py originally written by Mehdi Najafi (2018) and Anna Haley (2022). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

__authors__   = "Mehdi Najafi <mnuoft@gmail.com>. Anna Haley <ahaley@mie.utoronto.ca>. Rojin Anbarafshan <rojin.anbar@gmail.com>"
__date__      = "2018-2025"
__copyright__ = "Copyright (C) 2018-2025 UofT"
__license__   = "Private"

# ---------------------------------------- Imports and Basic Setup -----------------------------------------------------
from oasis import * 
from oasis.problems import * 
from oasis.problems.NSfracStep import *

from dolfin import *
import numpy as np
import sys, time, os, glob, pickle
from os import getpid, path, makedirs, getcwd
#from probe import * - this is never used and probably out of date AH

from BSLSolver import *
from BSLSolver.common import CustomFunction, h5io, Womersley
#from BSLSolver.common import naming
#from BSLSolver.common import WSS #only works in serial

# MPI Setup
mpi_size = MPI.size(MPI.comm_world)
mpi_rank = MPI.rank(MPI.comm_world)


# Setup timer
if mpi_rank == 0:
    from dolfin import Timer
    _global_timer = Timer()
    initial_wall_time = _global_timer.elapsed()[0]

# I/O helper from BSLSolver
h5stdio = h5io.HDF5StdIO()

# Reasonable wall-time guard (in seconds)
max_wtime_before_kill = (23.5*60*60)


# ---------------------------------------- Utilities Functions -----------------------------------------------------
def mpi_comm():
    return MPI.comm_world

def tuple2str(t, fmt='%12.10f'):
    return ','.join([fmt]*len(t))%tuple(t)

# For output formatting
def print_section_header(title, width=100):
    if mpi_rank == 0:
        print ("-"*width)
        print(title)
        sys.stdout.flush()

def print_section_footer(width=100):
    if mpi_rank == 0:
        print ("-"*width)
        sys.stdout.flush()

def info_gray(s, check=True):
    if mpi_rank == 0 and check:
        print ("\033[1;37;30m%s\033[0m"%s)

# For I/O handling
def get_cmdarg(cmdline, key, default_value = None):
    """Retrieve key from commandline kwargs with light type coercion based on default."""
    if key in cmdline.keys():
        value = cmdline[key]
        if default_value:
            if type(default_value) is int:
                return int(value)
            if type(default_value) is float:
                return float(value)
            if type(default_value) is bool:
                return bool(eval(value))
        return value
    return default_value

def get_file_paths(results_folder):
    if mpi_rank == 0:
        counter = 1
        to_check = path.join(results_folder, "data", "%s")
        while path.isdir(to_check % str(counter)):
            counter += 1

        if counter > 1:
            counter -= 1
        # if not path.exists(path.join(to_check % str(counter), "VTK")):
        #     makedirs(path.join(to_check % str(counter), "VTK"))
    else:
        counter = 0

    counter = MPI.max(MPI.comm_world, counter)

    common_path = path.join(results_folder, "data", str(counter), "VTK")
    file_u = [path.join(common_path, "u%d.h5" % i) for i in range(3)]
    file_p = path.join(common_path, "p.h5")
    file_nu = path.join(common_path, "nut.h5")
    file_u_mean = [path.join(common_path, "u%d_mean.h5" % i) for i in range(3)]
    files = {"u": file_u, "p": file_p, "u_mean": file_u_mean, "nut": file_nu}

    return files

def read_mesh_info(mesh_info_path, boundary_key):
    """
    Parse the casename.info file for <INLETS> or <OUTLETS> blocks.
    Retrive the boundary information from the info file.
    Each line has the format:
        id  waveform_tag  (cx,cy,cz)  (nx,ny,nz)  radius  area  flowrate_or_ratio

    Returns:
        boundary_ids  : list[int]      mesh boundary tag IDs
        flowrates     : list[float]    mean flowrates [mL/s] for inlets; OR split ratios for outlets
        areas         : list[float]    areas (mm^2)
        waveform_tags : list[str]      waveform file tags for inlets; OR 'None' for outlets
    """
    # Extract inflow rate and outflow split ratios:
    # Sample
    # <INLETS>
    # 3 ICA_V27:FC_MCA_10 (3.4,13.3,-28.2) (-0.2,0.1,-0.9) 1.76 9.78 5.16
    #
    # <OUTLETS>
    # 1  None  (-16.8,-1.4,17.7)  (-0.9,-0.1,-0.1)   0.86   2.33   0.33
    # 2  None    (7.9,-9.2,15.9)   (0.7,-0.4,-0.4)   1.33   5.64   0.66

    # Initialize outputs
    boundary_ids = []
    flowrates = []
    areas = []
    radii = []
    waveform_tags = []

    # Open info file
    info = open(mesh_info_path, 'r').read()

    # Looking for the given boundary key in the info file
    boundary_info_start = info.find(boundary_key)
    
    # In case there is not info for the boundary
    if boundary_info_start < 0: return [], [], [], []

    boundary_info_start += len(boundary_key)                #to go the line after boundary key
    boundary_info_end = info.find('<', boundary_info_start) #block ends when next <boundary> starts 
    
    # Reading the info block for the boundary
    if boundary_info_end < 0:
        boundary_info = info[boundary_info_start:] #read to end of mesh.info
    else:
        boundary_info = info[boundary_info_start:boundary_info_end-1] #read to end of boundary block
    

    lines = boundary_info.split('\n')
    # Reading the info values for the boundary
    for line in lines:
        tokens = line.split() #split the line from spaces
        if len(tokens) > 1: #skip useless tokens (like line breaks and empty values)
            boundary_ids.append(int(tokens[0]))
            waveform_tags.append(tokens[1])
            radii.append(eval(tokens[4]))
            areas.append(eval(tokens[5]))
            flowrates.append(tokens[6])          # store raw string, resolve later


    # Flowrate (last column) may use A/R as shorthand for this boundary's area/radius
    # Below script is to resolve this:

    # Build lookup dicts now that all boundaries are collected
    area_by_id   = dict(zip(boundary_ids, areas))
    radius_by_id = dict(zip(boundary_ids, radii))

    for i, (raw, area, radius) in enumerate(zip(flowrates, areas, radii)):
        expr = raw
        for bid in boundary_ids:            # cross-references A[id]/R[id] first
            expr = expr.replace(f'A[{bid}]', str(area_by_id[bid]))
            expr = expr.replace(f'R[{bid}]', str(radius_by_id[bid]))
        expr = expr.replace('A', str(area)).replace('R', str(radius))  # plain A/R last
        flowrates[i] = eval(expr)

    # Force outlet ratios to sum exactly to 1.0
    if boundary_key == '<OUTLETS>':
        flowrates[-1] = 1.0 - sum(flowrates[:-1])

    # print the summary
    # for i, flow_value in enumerate(flowrates):
    #     if mpi_rank == 0 and boundary_key == '<INLETS>':  print ('Inlet  id:', boundary_ids[i], ' flowrate (mL/s):', flow_value)
    #     if mpi_rank == 0 and boundary_key == '<OUTLETS>': print ('Outlet id:', boundary_ids[i], ' flowrate ratio:', flow_value)


    return boundary_ids, flowrates, areas, waveform_tags

def beta(err, p):
    if p < 0:
        if err >= 0.1:
            return 0.5
        else:
            return 1.0 - 5*err**2
    else:
        if err >= 0.1:
            return 1.5
        else:
            return 1.0  + 5*err**2


#UNUSED FUNCTIONS
"""
def step_str(i, l=10):
    # Get the zero leading string for given time step No.
    a = str(i)
    la = l - len(a)
    return '0'*la + a

def w(P):
    return 1.0 / ( 1.0 + 20.0*abs(P))

# check if the period is mentioned in the fc waveform file
def _not_used_get_period_from_fcs(fcs):
    periods = [951.0 for f in fcs]
    for i,f in enumerate(fcs):
        fcs_i_filename = f.split(':')[-1]
        if path.exists( path.join('./data', fcs_i_filename) ):
            fcs_ifname = path.join('./data', fcs_i_filename)
        else:
            fcs_ifname = path.join(path.dirname(path.abspath(__file__)), 'data', fcs_i_filename)
            if not path.exists( fcs_ifname ):
                print ('<!> Cannot find the waveform:', fcs_i_filename)
        for line in open(fcs_ifname,'r').readlines():
            if line.strip()[0] in ['#','!','/']:
                p = line.find('period_ms')
                if p > 0:
                    periods[i] = float(''.join((ch if ch in '0123456789.-e' else ' ') for ch in line[p+9:]).strip().split(' ')[0])
    return periods
"""

# ---------------------------------- Setup Parameters ---------------------------------------------------
def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
    """
    Fill NS_parameters with case-specific settings.
    Also prepares HDF5 naming template for BSLSolver IO.
    Note: Parameters are in mm and ms!!
    """
    
    mesh_name       = get_cmdarg(commandline_kwargs, "mesh_name")
    mesh_path       = path.join("./data", mesh_name + ".xml.gz")
    mesh_info_path  = path.join("./data", mesh_name + ".info")

    if mpi_rank == 0: print('\n[I/O] Reading mesh information:', mesh_info_path)

    # Check that the mesh files exist
    if mesh_name is None:
        raise RuntimeError("mesh_name is required (expects ./data/<mesh_name>.xml.gz and .info).")

    if mesh_path == None:
        print('<!> Unable to run without a mesh file.')

    # Obtain mesh information
    inlet_ids, Q_means, inlet_area, waveform_tags  = read_mesh_info(mesh_info_path, '<INLETS>')
    outlet_ids, area_ratio, outlet_area, _         = read_mesh_info(mesh_info_path, '<OUTLETS>')

    restart_folder = get_cmdarg(commandline_kwargs, 'restart_folder')

    if restart_folder:
        f = open(path.join(restart_folder, 'params.dat'), 'rb')
        NS_parameters.update(pickle.load(f))
        f.close()
        NS_parameters['restart_folder'] = restart_folder
        case_name = NS_parameters['case_name']
        case_fullname = NS_parameters['case_fullname']

    else:
        case_name       = get_cmdarg(commandline_kwargs, 'mesh_name')
        period          = get_cmdarg(commandline_kwargs, 'period', 915.0)   # [ms]
        timesteps       = get_cmdarg(commandline_kwargs, 'timesteps', 2000)
        no_of_cycles    = get_cmdarg(commandline_kwargs, 'cycles', 2)
        save_freq       = get_cmdarg(commandline_kwargs, 'save_frequency', 5)
            
        if mpi_rank == 0: print('[I/O] Found out period [ms] = %s '%str(period))


        # Build a descriptive case_fullname
        #txt = ''
        #for i, id in enumerate(id_in):
        #    txt += '_I%d_%s_Q%d'%(id,fcs[i].replace(':','_'),int(Q_means[i]*100))
        #txt += '_Per%d'%int(period)

        #case_fullname = ("art_" + mesh_name + txt + "_Newt370" + "_ts" + str(timesteps) + "_cy" + str(cycles) + "_uO" + str(uOrder))
        case_fullname = (mesh_name + "_ts" + str(timesteps) + "_cy" + str(no_of_cycles))
        results_folder = f"./results/{case_fullname}_saveFreq{save_freq}"

        #####--------- IMPORTANT: OASIS expects all parameters in [mm] and [ms]! -------------####
        
        # Calculate kinematic viscosity nu (all in SI units) -->
        mu_Pas   = get_cmdarg(commandline_kwargs, 'viscosity_mu_Pas', 0.0037)
        rho_kgm3 = get_cmdarg(commandline_kwargs, 'density_kgm3', 1057)
        nu_m2s   = mu_Pas/rho_kgm3
        nu_mm2ms = nu_m2s*1000 #convert from SI to units that oasis expects

        NS_parameters.update(
            # Case-specific params
            case_name           = case_name,
            case_fullname       = case_fullname,
            results_folder      = results_folder,
            mesh_path           = mesh_path, 
            inlet_ids           = inlet_ids,
            outlet_ids          = outlet_ids,
            area_ratio          = area_ratio,

            # Physics params
            no_of_cycles        = no_of_cycles,                                                         # total number of cycles
            period              = period,                                                               # time of a single cycle [ms]
            T                   = period * no_of_cycles,                                                # total simulation time [ms]
            dt                  = period / timesteps,                                                   # timestep size [ms]
            time_steps          = timesteps,
            nu                  = nu_mm2ms, #get_cmdarg(commandline_kwargs, 'viscosity', 0.0035),       # kinematic viscosity [mm^2/ms]
            velocity_degree     = get_cmdarg(commandline_kwargs, 'uOrder', 1),                          # FE degree of velocity

            # I/O params
            save_freq           = get_cmdarg(commandline_kwargs, 'save_frequency', 5),                  # save every N steps 
            save_first_cycle    = get_cmdarg(commandline_kwargs, 'save_first_cycle', False),            # flag to save first cycle or not
            save_exact_tsteps   = get_cmdarg(commandline_kwargs, 'save_exact_tsteps', False),           # save exact timesteps
            #save_tsteps_list   = get_cmdarg(commandline_kwargs, 'save_exact_tsteps', False), 
            checkpoint          = get_cmdarg(commandline_kwargs, 'checkpoint', 500),                    # write restart every N steps
            killtime            = get_cmdarg(commandline_kwargs, 'maxwtime', max_wtime_before_kill),
            dump_stats          = 1000,
            compute_flux        = 5,
            save_step           = get_cmdarg(commandline_kwargs, 'save_step', 100000),                  # Mehdi doesn't use the oasis output
            print_intermediate_info = 1000,                                                              # Controls the frequency of printing summary of timings to log file
            #print_WSS          = get_cmdarg(commandline_kwargs, 'print_WSS', True),
            #plot_interval      = 10e10,

            # Boundary conditions params
            inlet_BC_type             = get_cmdarg(commandline_kwargs, 'inlet_BC_type', 'pulsatile'), # choose from 'ramp', 'pulsatile', 'constant', 'custom'
            Qin_constant_mLs          = get_cmdarg(commandline_kwargs, 'inflowrate_constant_mLs', 5.0),       # constant inflow rate, used when inlet_BC_type='constant' [mL/s]
            ramp_slope                = get_cmdarg(commandline_kwargs, 'ramp_slope',  2.0),                   # slope of inflow ramp, used when inlet_BC_type='ramp'
            ramp_offset               = get_cmdarg(commandline_kwargs, 'ramp_offset', 2.0),                   # offset of inflow ramp, used when inlet_BC_type='ramp'
            not_zero_pressure_outlets = not get_cmdarg(commandline_kwargs, 'zero_pressure_outlets', False),
            include_gravity           = get_cmdarg(commandline_kwargs, 'include_gravitational_effects', False),
            flat_profile_at_intlet_bc = get_cmdarg(commandline_kwargs, 'flat_profile_at_intlet_bc', False),
            
            use_krylov_solvers  = True,
            krylov_solvers      = dict(
                monitor_convergence=False,
                error_on_nonconvergence=False, 
                nonzero_initial_guess=True, 
                maximum_iterations=200,
                relative_tolerance=1e-8,
                absolute_tolerance=1e-8)
            )

    # Write parameters to log
    if mpi_rank == 0:
        # Check if the kinematic viscosity (nu) is within correct range
        nu = NS_parameters['nu']
        if not (0.003 <= nu <= 0.004):
            raise ValueError("Error: Kinematic viscosity (nu) is out of expected range (0.003-0.004 [mm^2/ms])! CHECK THE UNTIS! \n")
        else:
            print(f"[I/O] Blood kinematic viscosity (nu) is within expected range: nu (mm^2/ms)= {nu:.4f} \n")

        # Print all NS_parameters to log
        info_gray(str(NS_parameters))

    
    # Initialize BSLSolver HDF5 pattern
    f = int(np.log10(NS_parameters['T']))+2
    g = len(str(int(NS_parameters['time_steps']*NS_parameters['no_of_cycles'])))+1
    h5stdio.init(NS_parameters['results_folder'], case_fullname+'_curcyc_%%d_t=%%0%d.4f_ts=%%0%dd_up.h5'%(f+4,g))


# --------------------------------------- Mesh ----------------------------------------------------------
def mesh(mesh_path, **NS_namespace):
    """Oasis function to create the mesh."""

    print_section_header('Loading mesh file: ' + mesh_path)

    #mesh_folder = mesh_path #path.join(path.dirname(path.abspath(__file__)), mesh_path)

    m =  Mesh(mesh_path)
    m.mpi_comm = mpi_comm

    # Mesh statistics
    num_points        = Function(FunctionSpace(m, "CG", 1)).vector().size()
    mesh_volume       = MPI.sum(MPI.comm_world, assemble(Constant(1)*dx(m)))
    cell_diameter     = [Cell(m,i).circumradius() for i in range (m.num_cells())]
    avg_cell_diameter = sum(cell_diameter) / len(cell_diameter)
    num_cells         = int( MPI.sum(MPI.comm_world, m.num_cells()) )
    #num_points       = int( MPI.sum(MPI.comm_world, m.num_vertices()) ) // shared points?
    hmin              = MPI.min(MPI.comm_world, m.hmin()) #[mm]
    hmax              = MPI.max(MPI.comm_world, m.hmax()) #[mm]
    num_facets        = int( MPI.sum(MPI.comm_world, m.num_facets()) )
    
    # Create HDF5 mesh filename
    mesh_stem        = os.path.basename(mesh_path).replace('.xml.gz', '').replace('.xml', '')
    mesh_h5_filename = mesh_stem + '.h5'
    mesh_h5_filepath = os.path.join(NS_namespace['results_folder'], mesh_h5_filename)


    # Write mesh parameters to log
    if mpi_rank == 0:
        #print ("-"*100)
        print ("Mesh Name:                  ", mesh_path)
        print ("Number of cells:            ", num_cells)
        print ("Number of points:           ", num_points)
        print ("Number of facets:           ", num_facets)
        print ("Mesh Volume:                ", mesh_volume)
        print (f"Min cell diameter [mm]:     {hmin:.4f}")
        print (f"Max cell diameter [mm]:     {hmax:.4f}")
        print (f"Average cell diameter [mm]: {avg_cell_diameter:.4f}")
        sys.stdout.flush()
        #info(m, False)

    boundary_markers = MeshFunction("size_t", m, m.geometry().dim() - 1, m.domains())

    inout_area = {}
    dS = {}
    for inlet_id in NS_namespace['inlet_ids']:
        dS[inlet_id] = ds(inlet_id, domain=m, subdomain_data=boundary_markers)
        inout_area[inlet_id] = abs( assemble(1.0*dS[inlet_id]) )
    for outlet_id in NS_namespace['outlet_ids']:
        dS[outlet_id] = ds(outlet_id, domain=m, subdomain_data=boundary_markers)
        inout_area[outlet_id] = abs( assemble(1.0*dS[outlet_id]) )
    
    NS_namespace['inout_area'] = inout_area

    normals = FacetNormal(m)

    if mpi_rank == 0:
        print('\n[I/O] Writing to: ', mesh_h5_filepath)
        sys.stdout.flush()

    # Output the Mesh file into HDF5 format
    Hdf = HDF5File(m.mpi_comm(), mesh_h5_filepath, "w")
    Hdf.write(m, '/Mesh')
    Hdf.close()

    h5stdio.SetMeshInfo(mesh_h5_filepath, mesh_h5_filename, num_cells, num_points)

    print_section_footer()

    return m, dS, boundary_markers, normals, m.geometry().dim(), inout_area

# Oasis hook: Overrides the default parameters
def post_import_problem(NS_parameters, mesh, commandline_kwargs, NS_expressions, **NS_namespace):
    """Oasis hook: Called after importing from problem."""

    # Update NS_parameters with all parameters modified through command line
    for key, val in commandline_kwargs.items():
        if isinstance(val, dict):
            NS_parameters[key].update(val)
        else:
            NS_parameters[key] = val

    # If the mesh is a callable function, then create the mesh here.
    if callable(mesh):
        mesh, dS, boundary_markers, normals, dimension, inout_area= mesh(**NS_parameters)

    assert(isinstance(mesh, Mesh))

    # Returned dictionary to be updated in the NS namespace
    dic = dict(mesh=mesh, dS=dS, subdomain_data=boundary_markers, normals=normals, dim=dimension, inout_area=inout_area)
    dic.update(NS_parameters)
    dic.update(NS_expressions)
    return dic



# --------------------------------------- Boundary Conditions --------------------------------------------

"""
# Read Inflow wave form and return the flow rate at all times
def flow_waveform(Qmean, cycles, period, time_steps, FC):
    omega = (2.0 * np.pi / period) #* cycles
    an = []
    bn = []

    #Load the Fourier Coefficients
    infile_FC = open( path.join(path.dirname(path.abspath(__file__)), 'data', FC), 'r').readlines()
    for line in infile_FC:
        abn = line.split()
        an.append(float(abn[0]))
        bn.append(float(abn[1]))

    t_values = np.linspace(0, period*cycles, num=time_steps)
    Q_values = []
    for t in t_values:
        Qn = 0 + 0j
        t1 = t / cycles
        for i in range (len(an)):
            Qn = Qn + (an[i]-bn[i]*1j)*np.exp(1j*i*omega*t1)
        Qn = abs(Qn)
        Q_values.append( Qmean * Qn )
        #print (t, Qn)
    return t_values, Q_values
"""

def ramp_inflowrate(t, slope=2, offset=0.01):
    """
    Linear ramp flowrate for the 'ramp' inlet_BC_type.
    t in ms (simulation time); returns Q_inflow in mL/s (consistent legacy units).

    slope  : ramp rate [mL/s^2] (t/1000 converts ms -> s, so Q grows by `slope` mL/s every second)
    offset : starting flowrate at t=0 [mL/s]
    """
    return slope * t / 1000 + offset


def constant_inflowrate(t, Q=5.0, ramp_duration=50.0):
    """
    Constant flowrate for the 'constant' inlet_BC_type, with a short linear ramp-up
    from a small initial value to Q at the start of the simulation to avoid initialization shocks.
    Returns Q_inflow in mL/s.

    t [ms]            : simulation time [ms]
    Q [mL/s]          : desired constant flowrate, set via the 'inflowrate_constant_mLs' commandline parameter (default 5.0)
    ramp_duration[ms] : duration of the ramp-up (default 50.0)
    """
    Q_start = 0.01
    if t >= ramp_duration:
        return Q
    return Q_start + (Q - Q_start) * (t / ramp_duration)


def poiseuille_inlet_velocity(mesh, ds_inlet, Q_inflow, **NS_namespace):
    """
    Build Poiseuille expressions aligned with the inlet's average normal.
    Q_inflow in mL/s (consistent units).
    """

    dim = mesh.geometry().dim()

    x = SpatialCoordinate(mesh) #[mm]
    area   = assemble(Constant(1.0)*ds_inlet) #[mm^2]
    center = (assemble(x[0]*ds_inlet)/area, assemble(x[1]*ds_inlet)/area, assemble(x[2]*ds_inlet)/area ) #[mm]
    

    # ----------------------- Obtain inlet parameters ---------------------- #
    
    ### 1. Compute the area-weighted average normal ###

    # Obtain raw normals
    # n_raw[i]: i-th component (i = 0,1,2) of the unit outward normal vector on each boundary facet
    n_raw  = FacetNormal(mesh)

    # Average the normals over the inlet
    n_avg  = np.array([assemble(n_raw[i]*ds_inlet) for i in range(dim)])
    
    # Calculate the length of average normal components (~ inlet area) -> used for normalization
    n_len  = np.sqrt(sum([n_avg[i]**2 for i in range(dim)]))

    # Normalize average normals -> normal: unit vector representing the average outward normal of the inlet patch
    normal = n_avg/n_len

    ### 2. Compute other parameters
    n0, n1, n2 = normal[0], normal[1], normal[2]
    c0, c1, c2 = center[0], center[1], center[2] #[mm]
    R = np.sqrt(area/np.pi) #[mm]

    # umax calculated based on input flowrate
    u_max  = 2.0 * Q_inflow / area        #[m/s] == [mm/ms] == [ml/s / mm^2]
    Reynolds = u_max*2*R/NS_parameters["nu"]

    dt = NS_parameters['dt']
    max_dt = 0.5*mesh.hmin()/u_max  # based on CFL = 0.5
    dt = NS_parameters['dt']
    if mpi_rank == 0:
        print(f"Starting simulations for dt [ms] = {dt}")
        print (f"Inlet properties: \n"
            f"R [mm]        =   {R:.4f} \n"
            f"Area [mm2]    =   {area:.4f} \n"
            f"Q [mL/s]      =   {Q_inflow:.4f} \n"
            f"umax [m/s]    =   {u_max:.4f} \n"
            f"Reynolds      =   {Reynolds:.1f} \n"
            f"centroid [mm] =   [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] \n"
            f"normal        =   [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}] \n"
            )


    # -------------------- Create Expressions for each direction ------------------- #
    # Obtain inlet poiseuille velocity (one component per axis)
    uin_expressions = [[],[],[]]

    # The poiseuille velocity profile kernel:

    kernel = (
        "-ncomp * (2.0 * Q_inflow/ area) * (1.0 - "
        "( pow((x[0]-c0) - n0 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[1]-c1) - n1 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[2]-c2) - n2 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) ) "
        " / (R*R) )"
    )

    # The ramp equation for inlet flowrate is embedded in the below Kernel: Q_in = 2*t/1000 + 0.01 -> t is in [ms]
    # kernel = (
    #     "-ncomp * (2.0 * (2*(t/1000) + 0.01)/ area) * (1.0 - "
    #     "( pow((x[0]-c0) - n0 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
    #     "  pow((x[1]-c1) - n1 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
    #     "  pow((x[2]-c2) - n2 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) ) "
    #     " / (R*R) )"
    # )

    for j in range(dim):      
        uin_expressions[j] = Expression(kernel, ncomp=normal[j], Q_inflow=Q_inflow, area=area,
                                        c0=c0, c1=c1, c2=c2,
                                        n0=n0, n1=n1, n2=n2,
                                        R=R, degree=2)
        #uin_expressions[j] = Expression(kernel, ncomp=normal[j], t=0., area=area, c0=c0, c1=c1, c2=c2, n0=n0, n1=n1, n2=n2, R=R, degree=2)

    return uin_expressions


# Create Boundary conditions
def create_bcs(u_, p_, p_1, t, NS_expressions, V, Q, area_ratio, mesh, subdomain_data, 
               dS, normals, results_folder, mesh_path, nu,
               inlet_ids, outlet_ids, velocity_degree, pressure_degree, no_of_cycles,
               T, not_zero_pressure_outlets, flat_profile_at_intlet_bc, **NS_namespace):

    print_section_header('Inspecting boundaries and making boundary conditions:')

    # Mesh function / boundaries
    boundary_markers = subdomain_data

    # Extract inflow rate and outflow split ratios
    mesh_info_path = path.join('./data', NS_namespace["mesh_name"]+'.info')
    inlet_ids, Q_means, inlet_area, waveform_tags = read_mesh_info(mesh_info_path, '<INLETS>')
    outlet_ids, area_ratio, outlet_area, _        = read_mesh_info(mesh_info_path, '<OUTLETS>')


    # 1. Inlet BCs
    inlet_ids_count = len(inlet_ids)
    inlet_BCtype = NS_parameters['inlet_BC_type']

    # Printing info to the log
    if mpi_rank == 0:
        print (f'Inlet BC type is {inlet_BCtype}')
        print(f'Inlet BCs on boundaries: {inlet_ids}')
        if inlet_BCtype == 'pulsatile':
            firststr = '    %8s    %-12s    %10s    %15s    %6s' % ('inlet_id', 'wave_form', 'period(ms)', 'flowrate(mL/s)', 'cells')
        elif inlet_BCtype == 'ramp':
            firststr = '    %8s    %14s    %12s    %6s' % ('inlet_id', 'slope(mL/s/s)', 'offset(mL/s)', 'cells')
        elif inlet_BCtype == 'constant':
            firststr = '    %8s    %14s    %6s' % ('inlet_id', 'Q_target(mL/s)', 'cells')
        else:
            firststr = '    %8s    %6s' % ('inlet_id', 'cells')
        secondstr = 'Inlets & Outlets Information\n'+'  id   %-45s  %-45s   %-12s   %-12s'%('center','normal','radius','area')
  
    inlets = []
    inout_area = {}
    bc_inlet_u = [[],[],[]]

    # Loop over inlets
    for i in range(inlet_ids_count):

        # Computing inlet area
        ds_inlet = dS[inlet_ids[i]]

        # Obtain inlet params
        waveform_filename = waveform_tags[i].split(':')[-1]
        inlet_area_i, inlet_center_i, inlet_radius_i, inlet_normal_i = Womersley.compute_boundary_geometry_acrn(mesh, dS[inlet_ids[i]], normals)
        
        # Create the inlet flow based on the flow type given by user

        # Option1: Pulsatile Womersley
        if inlet_BCtype == 'pulsatile': #if fcs_i_filename[0:3] == 'FC_':
            # waveform_filename is already captured in the inlet summary table printed at the end
            # if mpi_rank == 0: print ('- loading inflow wave form:', waveform_filename)
            inlet_velocity = Womersley.make_womersley_bcs_2(NS_namespace["period"], Q_means[i], waveform_filename, mesh, nu, inlet_area_i, inlet_center_i, inlet_radius_i, inlet_normal_i, velocity_degree, flat_profile_at_intlet_bc)
        

        # Option2: Ramp inflow (linearly increasing) --> added by Rojin A.
        elif inlet_BCtype == 'ramp':
            
            Q_inflow = ramp_inflowrate(t, NS_parameters['ramp_slope'], NS_parameters['ramp_offset'])
            inlet_velocity = poiseuille_inlet_velocity(mesh, ds_inlet, Q_inflow)

            # Surface integrand over the boundaries
            # inlet_tag = id_in[i] #inlet_tag = 2
            # ds_inlet = dS[inlet_tag]
            # Q_inflow = 3.73 #2*t/1000 + 0.01 #[ml/s]

            
        # Option3: Constant flowrate (steady)
        elif inlet_BCtype == 'constant':
            Q_inflow = constant_inflowrate(t, NS_parameters['Qin_constant_mLs'])
            inlet_velocity = poiseuille_inlet_velocity(mesh, ds_inlet, Q_inflow)

        else:
            if mpi_rank == 0:
                print ('The inlet_BC_type is not recognized. Choose from {<pulsatile>, <ramp>, <constant>}')
     
        inlets.append(inlet_velocity)
        bci = [DirichletBC(V, velocity_component, boundary_markers, inlet_ids[i]) for velocity_component in inlet_velocity]
        for j in range(3): bc_inlet_u[j].append(bci[j])

        count = len( bci[0].get_boundary_values() )
        inout_area[inlet_ids[i]] = inlet_area_i
        
        # For printing to log
        if mpi_rank == 0:
            if inlet_BCtype == 'pulsatile':
                firststr += "\n    %8d    %-12s    %10g    %15.8g    %6d" % (inlet_ids[i], waveform_filename, NS_namespace["period"], Q_means[i], count)
            elif inlet_BCtype == 'ramp':
                firststr += "\n    %8d    %14g    %12g    %6d" % (inlet_ids[i], NS_parameters['ramp_slope'], NS_parameters['ramp_offset'], count)
            elif inlet_BCtype == 'constant':
                firststr += "\n    %8d    %14g    %6d" % (inlet_ids[i], NS_parameters['Qin_constant_mLs'], count)
            else:
                firststr += "\n    %8d    %6d" % (inlet_ids[i], count)
            secondstr += "\nI %2d   %-45s  %-45s   %-12.10f   %-12.10f"%( inlet_ids[i], tuple2str(inlet_center_i), tuple2str(inlet_normal_i), inlet_radius_i, inlet_area_i)

    NS_expressions["inlet"] = inlets

    # Reset the time in boundary condition expressions
    if inlet_BCtype == 'pulsatile': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]:
            for uc in inlet: uc.set_t(t)

    # elif inlet_BCtype == 'ramp': # Added by Rojin A.
    #     for inlet in NS_expressions["inlet"]:
    #         for uc in inlet: uc.t = t


    # 2. Wall BCs
    # Create No-slip BCs for the velocity at the walls
    wall = Constant(0.0)
    bc_wall = DirichletBC(V, wall, boundary_markers, 0) # wall is always with the id zero
    
    # For DEBUG:
    #bc_wall_len = len(bc_wall.get_boundary_values())
    #if mpi_rank == 0: print( 'Wall BC on ' + str(bc_wall_len) , 'cells')


    # Add wall no-slip BC to each velocity component
    for bc_u in bc_inlet_u:
        bc_u.append(bc_wall)


    # 3. Outlet BCs
    outlet_ids_count = len(outlet_ids)
    bc_p = []

    # For printing to log
    if mpi_rank == 0:
        if not_zero_pressure_outlets:
            outletstr = f'Outlet BCs on boundaries: {outlet_ids}\n'
            outletstr += '    %8s    %14s    %6s' % ('outlet_id', 'mass_flow_ratio', 'cells')
        else:
            outletstr = f'Outlet BCs: zero-pressure on boundaries {outlet_ids}'

    # Loop over outlets
    for i, ind in enumerate(outlet_ids):
        outlet_area_i, outlet_center_i, outlet_radius_i, outlet_normal_i = Womersley.compute_boundary_geometry_acrn(mesh, dS[outlet_ids[i]], normals)
        inout_area[ind] = outlet_area_i
        if not_zero_pressure_outlets:
            if NS_parameters['restart_folder']:
                p_initial = assemble(p_*dS[ind]) / inout_area[ind]
            else:
                p_initial = area_ratio[i]
        else:
            p_initial = 0
        outflow = Expression("p", p=p_initial, degree=pressure_degree)
        bc = DirichletBC(Q, outflow, boundary_markers, ind)
        bc_p.append(bc)
        NS_expressions[ind] = outflow
        count  = len(bc.get_boundary_values())

        # Accumulate for printing to log
        if mpi_rank == 0:
            secondstr += "\nO %2d   %-45s  %-45s   %-12.10f   %-12.10f"%( ind, tuple2str(outlet_center_i), tuple2str(outlet_normal_i), outlet_radius_i, outlet_area_i)
            if not_zero_pressure_outlets:
                outletstr += '\n' + ' '*8 + '%4d    %14.12f    %8d' % (ind, p_initial, count)


    if mpi_rank == 0:
        print(firststr)
        print()
        print(outletstr)
        print()
        print(secondstr)
    print_section_footer(132)

    # Return boundary conditions in dictionary
    return dict(u0=bc_inlet_u[0], u1=bc_inlet_u[1], u2=bc_inlet_u[2], p=bc_p)


# ---------------------------------- Oasis Hooks ---------------------------------------------------

def pre_solve_hook(mesh, V, Q, newfolder, results_folder, u_, mesh_path,
                   restart_folder, tstep, velocity_degree, nu,**NS_namespace):

    if restart_folder is None:
        # Get files to store results
        files = get_file_paths(results_folder)
        NS_parameters.update(dict(files=files))
    else:
        files = NS_namespace["files"]


    return dict(hdf5_link=h5stdio,
                files=files, #inout_area=NS_parameters['inout_area'],
                final_time=NS_namespace['T'], current_cycle=0, 
                timesteps=NS_namespace['time_steps'], total_cycles=NS_namespace['no_of_cycles'],
                timestep_cpu_time=0, current_time=time.time(), cpu_time=0)


def temporal_hook(u_, p_, p, q_, V, mesh, tstep, compute_flux,
                  dump_stats, newfolder, files, inlet_ids, outlet_ids, inout_area, subdomain_data,
                  normals, save_freq, save_first_cycle, hdf5_link, NS_expressions, current_cycle,
                  total_cycles, area_ratio, t, dS, timestep_cpu_time, current_time, 
                  cpu_time, final_time, timesteps, not_zero_pressure_outlets, **NS_namespace):

    # Update the current cycles
    current_cycle = int(tstep / timesteps)
    
    # Calculate worst-case CFL across all velocity components and all MPI ranks
    # USe norm('linf') to get absolute value so negative/reverse-flow velocities are captured too
    max_u = max(u_[i].vector().norm('linf') for i in range(mesh.geometry().dim()))
    CFL = NS_parameters['dt'] * max_u / mesh.hmin()
    
    if mpi_rank == 0 and tstep % 100 == 0:
        #max_u = max(u_[0].vector().get_local().max(), u_[1].vector().get_local().max())
        #CFL = NS_parameters['dt']*max_u/mesh.hmin()
        print(f"For cycle= {current_cycle}  tstep= {tstep}  t(ms)= {t:.2f}:        CFL= {CFL:.4f}")


    #boundary_markers = subdomain_data # used in commented-out area assembly lines below

    # Update boundary condition (has to loop over all 3 expressions [expr_x, expr_y, expr_z])
    if NS_parameters['inlet_BC_type'] == 'pulsatile': # Added by Rojin A.
        for inlet in NS_expressions["inlet"]: # loops over inlets
            for uc in inlet:
                uc.set_t(t) #updating time for the kernel

    elif NS_parameters['inlet_BC_type'] == 'ramp': # Added by Rojin A.
        Q_inflow_now = ramp_inflowrate(t, NS_parameters['ramp_slope'], NS_parameters['ramp_offset'])
        for inlet in NS_expressions["inlet"]: 
            for uc in inlet:
                #uc.t = t #updating time for previous kernel
                uc.Q_inflow = Q_inflow_now
    
    elif NS_parameters['inlet_BC_type'] == 'constant':
        Q_inflow_now = constant_inflowrate(t, NS_parameters['Qin_constant_mLs'])
        for inlet in NS_expressions["inlet"]:
            for uc in inlet:
                # uc.t = t
                uc.Q_inflow = Q_inflow_now
                
    timestep_cpu_time = time.time() - current_time
    current_time = time.time()
    cpu_time += timestep_cpu_time

    # Do not proceed if the time step is less than 3
    if tstep < 3: return

    # In-Going Flux & pressure
    flux_in     = {}
    Q_ins       = {}
    pressure_in = {}
    umax_ins    = {}
    Re_ins      = {}
    for inlet_id in inlet_ids:
        #inout_area[inlet_id] = abs( assemble(1.0*ds(inlet_id, domain=mesh, subdomain_data=fd)) )
        pressure_in[inlet_id] = -assemble(p_*dS[inlet_id]) / inout_area[inlet_id]
        flux_in[inlet_id]     = assemble(dot(u_, normals)*dS[inlet_id])
        Q_ins[inlet_id]       = abs(flux_in[inlet_id])
        u_mean_in              = Q_ins[inlet_id] / inout_area[inlet_id]      # m/s
        umax_ins[inlet_id]    = 2*u_mean_in                                 # m/s
        R_in                  = np.sqrt(inout_area[inlet_id] / np.pi)       # inlet radius (mm)
        Re_ins[inlet_id]      = u_mean_in * (2*R_in) / NS_parameters["nu"]

    Q_ins_sum = sum(Q_ins.values())

    # Print to log
    if mpi_rank == 0 and tstep % 100 == 0:
        print(f'Q_ins(mL/s)= {Q_ins_sum:.4f}, umax_in(m/s)= {umax_ins[inlet_ids[0]]:.4f}, Reynolds_in= {Re_ins[inlet_ids[0]]:.1f} \n')


    # Out-Going Flux
    flux_out = {}
    Q_outs =  {}
    pressure_out = {}
    for out_id in outlet_ids:
        #inout_area[out_id] = abs( assemble(1.0*ds(out_id, domain=mesh, subdomain_data=fd)) )
        pressure_out[out_id] = assemble(p_*dS[out_id]) / inout_area[out_id]
        flux_out[out_id]     = assemble(dot(u_, normals)*dS[out_id])
        Q_outs[out_id]       = abs(flux_out[out_id])
    Q_outs_sum = sum(Q_outs.values())

    """
    # Compute flux and update outlet pressure BCs using dual-pressure method
    # (Gin & Steinman, "A Dual-Pressure Boundary Condition for use in Simulations of Bifurcating Conduits")
    # Used for models with multiple outlets where we don't know a priori how the inflow will split between branches
    if not_zero_pressure_outlets:
        Q_ideals = {}
        for i, out_id in enumerate(id_out):
            Q_ideals[i] = area_ratio[i] * Q_ins_sum
            p_old    = NS_expressions[out_id].p
            R_optimal = area_ratio[i]                       # target flow fraction
            R_actual  = Q_outs[out_id] / Q_ins_sum         # achieved flow fraction
            M_err = abs(R_optimal / R_actual)
            R_err = abs(R_optimal - R_actual)

            # sign of E controls whether pressure is increased or decreased
            if p_old < 0:
                E = 1 + R_err / R_optimal
            else:
                E = -1 * (1 + R_err / R_optimal)

            delta = (R_optimal - R_actual) / R_optimal

            # 1) Linear update for the first 100 tsteps to aid initial convergence
            if tstep < 100:
                h = 0.1
                if p_old > 1 and delta < 0:
                    NS_expressions[out_id].p = p_old
                else:
                    NS_expressions[out_id].p = p_old * (1 - delta * h)

            # 2) Full dual-pressure BC once flow is initialised
            else:
                if p_old > 2 and delta < 0:
                    NS_expressions[out_id].p = p_old
                else:
                    NS_expressions[out_id].p = p_old * beta(R_err, p_old) * M_err ** E
    """

    #Print the flow rates, fluxes, pressure
    if mpi_rank == 0:
        if NS_parameters['inlet_BC_type'] == 'pulsatile':
            flux_err = 100. * (abs(sum(flux_in.values())) - abs(sum(flux_out.values()))) / abs(sum(flux_in.values()))
            print("~" * 88)
            print(f"Flow Rate / Flux Error: {flux_err:.4f} %")
            print("~" * 88)
            print("%3s  %2s  %-16s  %-16s  %-16s  %-16s" % ('I/O', 'id', 'Flux', 'Velocity', 'Pressure', 'New Pressure'))
            for inlet_id in inlet_ids:
                print("%-3s  %2d  % 16.15f  % 16.15f  % 16.15f  %-16s" % ('In', inlet_id, flux_in[inlet_id], flux_in[inlet_id]/inout_area[inlet_id], pressure_in[inlet_id], 'N/A'))
            for i, out_id in enumerate(outlet_ids):
                print("%-3s  %2d  % 16.15f  % 16.15f  % 16.15f  % 16.15f" % ('Out', out_id, flux_out[out_id], flux_out[out_id] / inout_area[out_id], pressure_out[out_id], NS_expressions[out_id].p))
            
            print("~" * 88)

        sys.stdout.flush()

        elapsed_wall_time = _global_timer.elapsed()[0] - initial_wall_time


    # ---------------------------------- Saving ----------------------------------------        
    # If save_first_cycle is True it saves every cycle; otherwise skip cycle 0 and saves everything afterwards
    should_save = save_first_cycle or (current_cycle > 0)
    if should_save and tstep % save_freq == 0:
        h5stdio.Save(current_cycle, t, tstep, Q_ins, Q_outs, NS_parameters, 'Step-%06d' % tstep, q_)
        if mpi_rank == 0:
            h5stdio.SaveXDMF(os.path.join(NS_parameters['results_folder'], NS_parameters['case_fullname'] + '.xdmf'))


def theend_hook(stop, newfolder, results_folder, **NS_namespace):
    if mpi_rank == 0:
        if stop:
            if path.exists(path.join(newfolder,'complete')):
                os.remove(path.join(newfolder,'complete'))
            last_lines = open(path.join(newfolder,'incomplete'),'w')
        else:
            if path.exists(path.join(newfolder,'incomplete')):
                os.remove(path.join(newfolder,'incomplete'))
            last_lines = open(path.join(newfolder,'complete'),'w')

        last_lines.write('\nTry: ' + newfolder.split('/')[-1])
        last_lines.write('\nCheckpoint: ' + newfolder)
        last_lines.close()

    #print ('Process %d/%d Terminated.'%(mpi_rank,mpi_size))

