# Step-by-Step Guide: Running the Oasis CFD Solver (PT Ramp Study)

**Author:** Rojin Anbarafshan  
**Date:** 2026-04  
**For:** New users of the BSL Pulsatile Tinnitus CFD pipeline

---

## Overview

Three scripts work together to run one CFD case:

| Script | Role | Do you edit it? |
|---|---|---|
| `run_oasis_PT.sh` | **Case launcher** — defines all parameters and submits the job | **Yes, for every case** |
| `oasis_solver_PT.sh` | SLURM job wrapper — sets up the container environment | **Yes, once (paths only)** |
| `oasis_problem_PT.py` | Oasis problem definition — mesh loading, BCs, saving | **No** |

The workflow is: you run `run_oasis_PT.sh` → it calls `sbatch oasis_solver_PT.sh` → which launches `oasis NSfracStep problem=oasis_problem_PT` inside the container.

---

## Prerequisites

Before running anything, make sure you have:

1. **BSLSolver** cloned to your home directory:
   ```
   ~/BSLSolver/
   ```
2. **Apptainer container image** at:
   ```
   ~/containers/fenics-legacy/fenics-oasis.sif
   ```
3. **`pyshims/` directory** copied to your `$SCRATCH`:
   ```
   /scratch/<your_username>/pyshims/
   ```
   > This directory contains a shim that aliases `ufl-legacy` as `ufl`, required for the legacy FEniCS container. Copy it from the shared lab directory.

4. **Mesh data** for your case placed under `./data/` in your case directory:
   - `./data/<casename>.xml.gz` — the mesh file
   - `./data/<casename>.info` — boundary information file (inlets/outlets)

---

## Step 1: Set Up Your Directory Structure

Each PT case should live in its own directory. Your case directory should look like this before submitting:

```
/scratch/<your_username>/<your_case_dir>/
├── data/
│   ├── PTSeg028_base_0p64.xml.gz      ← your mesh
│   └── PTSeg028_base_0p64.info        ← boundary info
├── run_oasis_PT.sh                    ← the launcher you edit and run
├── logs/                              ← created automatically
└── hpclog/                            ← created automatically
```

> `oasis_solver_PT.sh` and `oasis_problem_PT.py` do **not** need to be copied into each case directory. They live in one shared location and are pointed to via `PATH_OASIS_SOLVER`.

---

## Step 2: Edit `oasis_solver_PT.sh` (one-time setup)

This file only needs to be edited **once** when you first set things up. Update the path variables near line 74–87 to match your file system:

```bash
# Path to this solver script's directory (contains oasis_problem_PT.py)
PATH_OASIS_SOLVER="/scratch/<your_username>/path/to/scripts/step1_CFD"

# Path to your BSLSolver clone
PATH_BSLSOLVER="/home/<your_username>/BSLSolver"

# Path to the Apptainer container image
PATH_CONTAINER="/home/<your_username>/containers/fenics-legacy/fenics-oasis.sif"

# Path to the pyshims directory (ufl-legacy shim)
PATH_UFL_SHIM="/scratch/<your_username>/pyshims"
```

> **Note:** The `$scinet_user` variable in these paths is exported from `run_oasis_PT.sh`, so you can also leave the `$scinet_user` references and just make sure `scinet_user` is set correctly in the launcher.

Also check line 143 — the container path is hardcoded there too:
```bash
apptainer exec ... ~/containers/fenics-legacy/fenics-oasis.sif \
```
This uses `~` (your home directory), so it should work automatically as long as the `.sif` file is in the right place.

---

## Step 3: Edit `run_oasis_PT.sh` (for every case)

This is the only file you need to touch for each new case. Copy it into your case directory and update the following:

### 3a. Point to the solver script
```bash
PATH_OASIS_SOLVER="/scratch/<your_username>/path/to/scripts/step1_CFD/oasis_solver_PT.sh"
```
This must point to the actual `oasis_solver_PT.sh` file (not just the directory).

### 3b. Cluster settings
```bash
scinet_user=<your_username>     # your cluster login username
group_name=def-<pi_name>        # your group allocation name
debug=off                       # 'on' = debug partition (max 1 hr), 'off' = compute (max 24 hrs)
num_cores=100                   # number of MPI tasks (cores); must fit on one node
required_time="15:59:59"        # walltime request (HH:MM:SS, max 24:00:00)
```

### 3c. Case identification
```bash
casename="PTSeg028_base_0p64"   # must match the mesh filename in ./data/ (without extension)
```
The full output folder name is built automatically as:
```
./results/<casename>_ts<timesteps_per_cycle>_cy<cycles>_saveFreq<save_frequency>/
```

### 3d. Simulation parameters
```bash
cycles=6                        # number of cardiac cycles to simulate
period=915.0                    # cardiac cycle period [ms] (default: 915 ms)
timesteps_per_cycle=10000       # timesteps per cycle (more = finer time resolution, slower)
viscosity_mu_Pas=0.0037         # blood dynamic viscosity [Pa·s]
density_kgm3=1057               # blood density [kg/m³]
uOrder=1                        # velocity finite element order (1 = linear, 2 = quadratic)
```

### 3e. Inlet boundary condition type
```bash
inlet_BC_type="ramp"            # options: 'pulsatile', 'ramp', or 'custom'
```

| Option | Description |
|---|---|
| `pulsatile` | Uses Womersley BC with a Fourier-series waveform from `./data/<FC_file>` |
| `ramp` | Linear ramp: Q(t) = 2t/1000 + 0.01 [mL/s]. No waveform file needed. |
| `custom` | Custom flow function via BSLSolver (not fully supported yet) |

> For the ramp study, use `inlet_BC_type="ramp"`.

### 3f. Output settings
```bash
save_first_cycle=True           # True = also save cycle 0; False = skip first cycle
save_frequency=1                # save solution every N timesteps (1 = every step)
checkpoint=500                  # write restart checkpoint every N steps
```

---

## Step 4: Run the Job

From inside your case directory (where `./data/` lives), submit:

```bash
./run_oasis_PT.sh
```

The script will:
1. Validate that `oasis_solver_PT.sh` exists at `PATH_OASIS_SOLVER`
2. Create `./logs/` and `./hpclog/` directories
3. Submit the SLURM job with all parameters exported as environment variables

You can monitor the job with:
```bash
squeue -u <your_username>
```

---

## Step 5: Check Outputs

After the job runs, you will find:

| Location | Contents |
|---|---|
| `./results/<casename_full>/` | HDF5 + XDMF solution files, mesh `.h5`, restart checkpoints |
| `./logs/<casename_full>_<job_id>` | Per-timestep solver output (flux, CFL, pressure) |
| `./hpclog/<jobname>_<job_id>.txt` | SLURM stdout/stderr |
| `./results/<casename_full>/complete` or `incomplete` | Completion status written by the solver |

---

## Common Issues

**"mesh_name is required" error**  
→ `casename` in `run_oasis_PT.sh` does not match the `.xml.gz` filename in `./data/`.

**"Kinematic viscosity out of expected range" error**  
→ Check `viscosity_mu_Pas` and `density_kgm3`. The solver internally computes ν = μ/ρ and expects it in [0.003, 0.004] mm²/ms. Default values (0.0037 Pa·s, 1057 kg/m³) are correct for blood.

**Container not found**  
→ Confirm `~/containers/fenics-legacy/fenics-oasis.sif` exists on the cluster.

**MPI errors inside container**  
→ `HYDRA_LAUNCHER=fork` is already set in `oasis_solver_PT.sh` to handle this; if it persists, check that the `--bind` paths in `BIND_OPTS` cover all locations your data/code live in.

**Job exits immediately with no log output**  
→ Check `./hpclog/` for the SLURM stderr. A missing `pyshims/` directory or incorrect `PATH_UFL_SHIM` will cause a silent Python import failure.
