import os
import numpy         as np
import math
from   pathlib       import Path
from   typing        import Tuple,  List
from   scipy.ndimage import distance_transform_edt
from   typing        import Union
from   numpy.typing  import NDArray
import cc3d
import matplotlib.pyplot as plt
import porespy as ps
import os
from pathlib import Path
from typing import List, Optional
import glob


def write_domain_raw(path: str, x_sample: np.ndarray, filename: str = "domain.raw") -> str:
    """
    x_sample: 3D (Nz, Ny, Nx), 0 = solid, 1 = fluid (uint8)
    """
    p = Path(path) if path else Path(".")
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / filename
    np.asarray(x_sample, dtype=np.uint8).tofile(out_path)
    print(f"   -> domain.raw written to: {out_path}")
    return str(out_path)

def find_raw_in_folder(base_dir, filename):
    raw_files = glob.glob(os.path.join(base_dir, "**", filename), recursive=True)
    print(f"Found {len(raw_files)} {filename} files in {base_dir}") 
    return raw_files
    
def generate_slurm_run_scripts_chunks(
    sample_indices: List[int],
    n_proc: int,
    output_root: str,
    samples_per_job: int,
    cpu: int, 
    gpu: int,
    gres: Optional[str] = None,                        
    partition: str = "all_gpu",                        
    dispatcher_name: str = "submit_all_sims_chain.sh",
    lbpm_version: str = "lbpm/gpu/",
    use_low_prio: bool = False,
    include_allocation: bool = False
):
    """
    Gera scripts SLURM em chunks e um dispatcher centralizado.
    As variáveis GRES e PARTITION são definidas no dispatcher para fácil alteração.
    """
    sample_indices = sorted(sample_indices)
    scripts_dir = Path(output_root).resolve()
    scripts_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        sample_indices[i : i + samples_per_job]
        for i in range(0, len(sample_indices), samples_per_job)
    ]

    chunk_script_names = []

    for chunk_id, chunk_samples in enumerate(chunks):
        start_idx = chunk_samples[0]
        end_idx   = chunk_samples[-1]
        range_id = f"{start_idx:05d}_{end_idx:05d}"
        
        chunk_script_name = f"run_sims_{range_id}.sh"
        chunk_script_path = scripts_dir / chunk_script_name
        chunk_script_names.append(chunk_script_name)

        allocation_settings = ""
        if include_allocation:
            allocation_settings = f"#SBATCH --mem-per-gpu={gpu}G\n#SBATCH --cpus-per-gpu={cpu}"

        # Header do Chunk (Sem Partition/Gres fixos)
        chunk_content = f"""#!/bin/bash

# ---------------- SLURM Job Settings ----------------
#SBATCH --oversubscribe
#SBATCH --job-name=Perm_{range_id}
{allocation_settings}
#SBATCH -t 7-0:00
#SBATCH -o perm_{range_id}_%j.out
#SBATCH -e perm_{range_id}_%j.err
#SBATCH --ntasks={n_proc}

# ---------------- Environment Setup ----------------
module load $LBPM_VERSION

echo "=== Chunk {chunk_id:03d} | Samples {start_idx:05d} to {end_idx:05d} ==="
"""
        # Execução das simulações
        first = True
        for sample_i in chunk_samples:
            folder_base = f"Sample_{sample_i:05d}"
            cd_cmd = f"cd {folder_base}" if first else f"cd ../{folder_base}"
            
            # Adicionado --oversubscribe no mpirun para evitar erro de slots
            command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
{cd_cmd}
echo "Current Simulation: " ${{PWD##*/}}
mpirun --oversubscribe -np {n_proc} lbpm_permeability_simulator simulation.db
"""
            chunk_content += command_block
            first = False

        chunk_content += '\necho "--> All simulations in this chunk finished."\n'
        chunk_script_path.write_text(chunk_content, encoding="utf-8")

    # ----- Gerar Dispatcher (run.sh) -----
    dispatcher_path = scripts_dir / dispatcher_name
    gres_val = f"{gres}:{n_proc}" if gres else ""

    dispatcher_content = f"""#!/bin/bash

#SBATCH --partition="{partition}"
# =========================================================
# GLOBAL RUN SETTINGS
# Altere aqui para atualizar todos os jobs da corrente
# =========================================================
export LBPM_VERSION="{lbpm_version}"
PARTITION="{partition}"
GRES_STR="{gres_val}"

# Configuração dinâmica de GRES
GRES_FLAG=""
if [ ! -z "$GRES_STR" ]; then
    GRES_FLAG="--gres=$GRES_STR"
fi

"""
    qos_flag = "--qos=low_prio " if use_low_prio else ""
    prev_var = ""
    
    for idx, name in enumerate(chunk_script_names):
        var_name = f"j{idx+1}"
        dep = f"--dependency=afterok:${prev_var} " if idx > 0 else ""
        
        # Injeção das variáveis centrais no comando sbatch
        sbatch_line = f'{var_name}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {qos_flag}{dep}{name})'
        
        dispatcher_content += f'{sbatch_line}\n'
        dispatcher_content += f'echo "Submitted {name} to $PARTITION (Job: ${var_name})"\n\n'
        prev_var = var_name

    dispatcher_content += 'echo "--> All chained jobs submitted."\n'
    dispatcher_path.write_text(dispatcher_content, encoding="utf-8")

    print(f"[SUCCESS] {len(chunks)} scripts criados em: {scripts_dir}")
    print(f"Comando para rodar: chmod +x {dispatcher_name} && ./{dispatcher_name}")
    
    
    
    
    
    
    
def plot_lt_distribution(vol, r1, r2, real_percentage):
    
    lt = ps.filters.local_thickness(vol)
    fluid_pixels = lt[lt > 0]

    # Criando subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: imagem ---
    ax0 = axes[0]
    im = ax0.imshow(lt[0, :, :], cmap='viridis')
    ax0.set_title('Local Thickness Slice')
    ax0.axis('off')
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # --- Subplot 2: histograma ---
    ax1 = axes[1]

    # bins baseados nos valores únicos
    unique_vals = np.unique(fluid_pixels)

    if len(unique_vals) <= 10:
        bins = np.arange(np.min(unique_vals), np.max(unique_vals) + 2) - 0.5
    else:
        bins = np.linspace(np.min(fluid_pixels), np.max(fluid_pixels), 11)

    counts, bin_edges, patches = ax1.hist(
        fluid_pixels,
        bins=bins,
        edgecolor='black',
        alpha=0.7
    )

    # centros dos bins
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ticks em todos os centros
    ax1.set_xticks(bin_centers)

    # limitar eixo x
    ax1.set_xlim(0, 20)

    # calcular cobertura (via histograma)
    mask = (bin_centers >= r1) & (bin_centers <= r2)
    covered_pixels = np.sum(counts[mask])
    total_pixels = np.sum(counts)
    covered_percentage = 100 * covered_pixels / total_pixels if total_pixels > 0 else 0

    # colorir bins
    for patch, center in zip(patches, bin_centers):
        if r1 <= center <= r2:
            patch.set_facecolor('navy')
        else:
            patch.set_facecolor('skyblue')

    # linhas verticais
    ax1.axvline(r1, color='green', linestyle='--', label=f'R1: {r1}')
    ax1.axvline(r2, color='green', linestyle='--', label=f'R2: {r2}')

    ax1.set_title(
        f'Maximum Inscribed Sphere Distribution\n'
        f'{covered_percentage:.2f}% of the voxels inscribed in radii between R1 and R2'
    )

    ax1.set_xlabel('Equivalent Radius [pixels]')
    ax1.set_ylabel('Absolute Frequency [pixel count]')

    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def add_enclusure_walls(vol):
    vol[:, :, 0]   = 0
    vol[:, :, -1]  = 0
    vol[:, 0, :]   = 0
    vol[:, -1, :]  = 0
    
    return vol

def remove_isolated_pores(vol):
    print("-->Removing isolated voxels...")
    inlets              = np.zeros_like(vol, dtype=bool)
    inlets[0, :, :]     = 1  
    outlets             = np.zeros_like(vol, dtype=bool)    
    outlets[-1, :, :]   = 1 
    filt_vol            = ps.filters.trim_nonpercolating_paths(vol, inlets=inlets, outlets=outlets)
    return filt_vol

def check_local_thickness(im, min_radius, max_radius, target_percentage=70.0): 
    print("Calculating local thickness...")
    
    inlets  = np.zeros_like(im, dtype=bool)
    inlets[0, :, :] = 1  
    outlets = np.zeros_like(im, dtype=bool)    
    outlets[-1, :, :] = 1 
    im_transp = ps.filters.trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)
    
    # Maximum inscribed sphere
    lt                  = ps.filters.local_thickness(im_transp)
    # Where fluid cells are located
    fluid_pixels        = lt[lt > 0]    
    # Where the desired local thickness is met
    in_range_mask       = (fluid_pixels >= min_radius) & (fluid_pixels <= max_radius)
    # Total occurences of matched cells
    real_percentage = (np.sum(in_range_mask) / len(fluid_pixels)) * 100.0
    # Verify if the global criteria is met
    is_target_met = real_percentage >= target_percentage
    
    print(f"Mean: {np.mean(fluid_pixels)}, Std: {np.std(fluid_pixels)}")
    print(f"Percentage: {real_percentage:.2f}% in range [{min_radius}, {max_radius}]")
    print(f"Target Met: {'YES' if is_target_met else 'NO'}")
    
    return is_target_met
"""

def is_well_resolved(data: NDArray, min_pore_mean, max_pore_mean):
    snow                = ps.filters.snow_partitioning(data)
    network             = ps.networks.regions_to_network(regions=snow.regions)
    pore_diameters      = network['pore.inscribed_diameter']
    throat_diameters    = network['throat.inscribed_diameter']
    print("Mean Pore: ", np.mean(pore_diameters)/2, "; Mean Throat: ", np.mean(throat_diameters)/2)
    return np.mean(pore_diameters)/2 >= min_pore_mean and np.mean(pore_diameters)/2 <= max_pore_mean
"""
def is_percolating(
    data: NDArray,
    axis: int,
) -> Tuple[NDArray, List[int], List[int]]:
    
    data = (data > 0).astype(np.uint8)
    
    # Data array must be binary and have 1 as pore
    connectivity = 6  # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labeled_components, num_labels = cc3d.connected_components(
        data, connectivity=connectivity, return_N=True)
    if axis == 0:
        labels_inlet = np.unique(labeled_components[0, :, :])
        labels_outlet = np.unique(labeled_components[-1, :, :])
    elif axis == 1:
        labels_inlet = np.unique(labeled_components[:, 0, :])
        labels_outlet = np.unique(labeled_components[:, -1, :])
    elif axis == 2:
        labels_inlet = np.unique(labeled_components[:, :, 0])
        labels_outlet = np.unique(labeled_components[:, :, -1])
    else:
        raise ValueError()

    labels_inlet = set(labels_inlet)
    labels_outlet = set(labels_outlet)
    connected_labels = labels_inlet.intersection(labels_outlet)
    # at least the rock phase (0) will always appear on both inlet and outlet
    
    
    if 0 in connected_labels:    connected_labels.remove(0)
    
    connected_labels = list(connected_labels)

    # get all labels which were pore
    all_labels = set(range(num_labels + 1))
    all_labels.remove(0)

    # get all labels which were pore and are disconnected
    disconnected_labels = all_labels.difference(connected_labels)
    disconnected_labels = list(disconnected_labels)

    return len(connected_labels) > 0

def next_multiple_after(x, multiple_of):
    return int(((x // multiple_of) + 1) * multiple_of)



def generate_slurm_run_script(sample_indices: List[int], n_proc: int, gres: str, output_root: str, script_name: str = "run_all_sims_sequential.sh", lbpm_module: str = "lbpm/gpu/"):
    """
    Generates a SLURM/Bash script with explicit sequential commands (no loops) 
    for every sample index provided in sample_indices. This style replicates 
    the structure of your original Run_0_2.sh script.
    
    Args:
        sample_indices (List[int]): List of the actual 0-based sample indices (e.g., [0, 1, 2, ...]).
        n_proc (int): Number of MPI tasks/cores to request per simulation (e.g., 4).
        output_root (str): The base directory containing all Sample_ folders.
        script_name (str): The name of the output script file.
    """
    
    # --- 1. SLURM Header and Environment Setup ---
    script_content = f"""#!/bin/bash

# ---------------- SLURM Job Settings ----------------
# NOTE: This script runs all simulations SEQUENTIALLY. It is best for small 
# numbers of samples or if your job queue favors long, single-task jobs.

#SBATCH --oversubscribe
#SBATCH --job-name=Perm_FullRun_Sequential       # Job name for identification
#SBATCH --partition=all_gpu                      # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --gres={gres}:{n_proc}                      # Request {n_proc} GPUs (or resources)

#SBATCH -t 7-0:00                              # Max wall time: 7 days (increased for safety)
#SBATCH -o run_outputs_%j.out                  # File to write standard output (%%j = job ID)
#SBATCH -e run_error_%j.err                    # File to write standard error (%%j = job ID)

# ---------------- Environment Setup ----------------

# Load the appropriate module (as suggested by your example script)
module load lbpm/gpu/poro_dev_78ba76

# Change into the root directory where all sample folders reside


# ---------------- Job Execution (Sequential) --------------------


cd DeePore_Samples

"""
    
    # --- 2. Append sequential command block for each sample ---
    for sample_i in sample_indices:
        # Format the folder name with zero-padding (e.g., Sample_00000)
        folder_base = f"Sample_{sample_i:05d}"
        
        
        if sample_i == sample_indices[0]:
            
            command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd {folder_base}
echo \"Current Simulation: \"${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db
# Move back two directories to the root SAMPLE_ROOT directory
"""
        else:
            command_block = f"""
echo "--- Launching simulation for {folder_base} ---"
cd ../{folder_base}
echo \"Current Simulation: \"${{PWD##*/}}
mpirun -np {n_proc} lbpm_permeability_simulator simulation.db

# Move back two directories to the root SAMPLE_ROOT directory
"""
        
        script_content += command_block
    
    script_content += "\n\necho \"--> All sample simulations launched successfully.\""

    # --- 3. Write the script ---
    # Writes the file one directory up from the execution location, assuming this Python script 
    # runs inside the root of your project directory.
    script_path = os.path.join(Path(os.getcwd()).parent, script_name)
    Path(script_path).write_text(script_content, encoding="utf-8")
    print(f"\n[SUCCESS] Generated sequential run script: {script_path}")
    print("Remember to make the script executable: chmod +x run_all_sims_sequential.sh")
    

def danny_normalization_vel(vel, void_mask, tau=1.5, Re=0.1):
    
    visc    = (tau-0.5)/3
    force   = force_calculation(void_mask, tau=tau, Re=Re)
    perm_est= (0.65*np.max(distance_transform_edt(void_mask).astype("float32")))**2 / 5
    
    return vel*visc / (force*perm_est)

def silveira_normalization_vel(vel, void_mask, tau=1.5, Re=0.1):
    visc    = (tau-0.5)/3
    k0      = 1
    Kt      = visc * tau
    Ke      = Kt / k0
    force   = force_calculation(void_mask, tau=tau, Re=Re)
    vel_norm= vel * Ke / (tau*force)
    
    return vel_norm
    

def silveira_normalization_pres(pressure, void_mask, tau=1.5, Re=0.1):
    
    delta_p     = pressure_calculation(void_mask, tau=tau, Re=Re)
    p_mean      = (2+3*delta_p)/6
    delta_p_new = 0.2
    p_mean_new  = 0.15
    pr_norm     = ((pressure -p_mean)/delta_p)*delta_p_new + p_mean_new  
    return pr_norm
            

def order_ceil(value: float) -> float: 
    """ 
    Round up to the nearest power of 10 (ceil-like behavior). 
    Ex: 6.0 -> 10.0, 0.034 -> 0.1, 15.0 -> 100.0, 10.0 -> 10.0
    """ 
    if value <= 0: return 0.0
    log_value = np.log10(value) 
    exponent = math.ceil(log_value) 
    return 10 ** exponent

def timestep_calculation(    
        matriz_binaria: np.ndarray,
        tau: Union[float, int],
        Re: float = 0.01,
        Dens: float = 1.0,
        safety_factor: float = 10.0)  -> int:
    
    L = np.max(matriz_binaria.shape)
    T = int(safety_factor*3*L**2*Dens / (Re*(tau-0.5)))    
    return order_ceil(T)
    
def pressure_calculation(
    matriz_binaria: np.ndarray,
    tau:        Union[float, int],
    Re:         float = 0.1,
    Dens:       float = 1.0,
    )->float:
    
    L               = matriz_binaria.shape[0]
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    dP              = (Re * 8.0 * (Visc ** 2) * L) / (Dens * (R ** 3))

    return dP

def force_calculation(
    matriz_binaria: np.ndarray,
    tau:            Union[float, int],
    Re:             float = 0.1,
    Dens:           float = 1.0,
) -> float:
    
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    Fx              = (Re * 8.0 * (Visc ** 2)) / (Dens * (R ** 3))
    return Fx

def write_lbpm_db(
    path: str,
    *,
    db_name:    str = "simulation.db",   # used if `path` is a directory
    bc:         int = 0,
    din:        float = 1.0,
    dout:       float = 1.0,
    fz:         float = 0.0,
    fx:         float = 0.0,
    fy:         float = 0.0,
    tau:        float = 1.5,
    timestep_max: int = 50000,
    tolerance: float = 1e-4,
    # Domain
    domain_filename:str = "domain.raw",
    read_type:      str = "8bit",
    nproc:          Tuple[int, int, int] = (1, 1, 4),
    n:              Tuple[int, int, int] = (256, 256, 128),
    N:              Tuple[int, int, int] = (256, 256, 512),
    offset:         Tuple[int, int, int] = (0, 0, 0),
    voxel_length:   float = 1.0,
    read_values:    Tuple[int, int] = (0, 1),
    write_values:   Tuple[int, int] = (0, 1),
    inlet_layers:   Tuple[int, int, int] = (0, 0, 0),
    outlet_layers:  Tuple[int, int, int] = (0, 0, 0),
    # Visualization
    write_silo:     bool = True,
    save_8bit_raw:  bool = True,
    save_phase_field: bool = True,
    save_pressure:  bool = True,
    save_velocity:  bool = True,
    # Analysis
    analysis_interval:          int = 100,
    subphase_analysis_interval: int = 100_000_000,
    n_threads:                  int = 0,
    visualization_interval:     int = 100_000_000,
    restart_interval:           int = 100_000_000,
    restart_file:               str = "Restart",
) -> str:
    def tsv3(v): return f"{v[0]}, {v[1]}, {v[2]}"
    def tsv2(v): return f"{v[0]}, {v[1]}"
    def b(v):    return "true" if v else "false"
    def ffmt(x): return f"{x:.6g}"

    text = f"""MRT {{
   tau         = {ffmt(tau)}
   din         = {din}   // inlet density (controls pressure)
   dout        = {dout}  // outlet density (controls pressure)
   F           = {ffmt(fx)}, {ffmt(fy)}, {ffmt(fz)}   // Fx, Fy, Fz
   timestepMax = {timestep_max}
   tolerance   = {ffmt(tolerance)}
}}
Domain {{
   Filename = "{domain_filename}"
   ReadType = "{read_type}"      // data type

   nproc = {tsv3(nproc)}
   n     = {tsv3(n)}
   N     = {tsv3(N)}

   offset         = {tsv3(offset)} // offset to read sub-domain
   voxel_length   = {ffmt(voxel_length)}     // voxel length (in microns)
   ReadValues     = {tsv2(read_values)}    // labels within the original image
   WriteValues    = {tsv2(write_values)}    // associated labels to be used by LBPM (0:solid, 1..N:fluids)
   BC             = {bc}       // boundary condition type (0 for periodic)
   InletLayers    = {tsv3(inlet_layers)}   // specify layers along the inlet
   OutletLayers   = {tsv3(outlet_layers)}  // specify layers along the outlet
}}
Visualization {{
   format            = "vtk"
   write_silo        = {b(write_silo)}     // SILO databases with assigned variables
   save_8bit_raw     = {b(save_8bit_raw)}  // labeled 8-bit binary files with phase assignments
   save_phase_field  = {b(save_phase_field)}  // phase field within SILO database
   save_pressure     = {b(save_pressure)}    // pressure field within SILO database
   save_velocity     = {b(save_velocity)}    // velocity field within SILO database
}}
Analysis {{
   analysis_interval             = {analysis_interval}        // logging interval for timelog.csv
   subphase_analysis_interval    = {subphase_analysis_interval}  // logging interval for subphase.csv
   N_threads                     = {n_threads}                // number of analysis threads (GPU version only)
   visualization_interval        = {visualization_interval}   // interval to write visualization files
   restart_interval              = {restart_interval}         // interval to write restart file
   restart_file                  = "{restart_file}"           // base name of restart file
}}
"""
    p = Path(path)
    # If `path` is a directory or lacks a suffix, write inside it
    if p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        p = p / db_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(text, encoding="utf-8")
    return text


def create_simulation_pressure_condition(x_sample, output_root, folder_base, n_proc=4, include_walls=False):

    x_sample = x_sample.copy() 
    
    # Transform x_sample for simulation:
    if include_walls:
        x_sample[:, :, 0]   = 0
        x_sample[:, :, -1]  = 0
        x_sample[:, 0, :]   = 0
        x_sample[:, -1, :]  = 0
    
    # Sanity Checks:
    if not is_percolating(x_sample, axis=0): print("Sample failed to percolate and got removed.")
    else:
        
        folder_rbc  = os.path.join(output_root, folder_base)
        os.makedirs(folder_rbc, exist_ok=True)
        
        Re   = 0.1
        tau  = 1.5
        Dens = 1.0
        dP = pressure_calculation(           
                x_sample,
                tau     = tau,
                Re      = Re,
                Dens    = Dens
            )
        
        timestep_max = timestep_calculation(    
                matriz_binaria  =x_sample,
                tau             =tau,
                Re              =Re,
                Dens            =Dens,
                safety_factor   =10.0
                )
        
        # --- Save 3D domain as .raw ---
        write_lbpm_db(path=folder_rbc, 
                      tau       = tau,
                      bc        = 3,
                      din       = 1.0+dP*3,
                      dout      = 1.0,
                      nproc     = (1, 1, n_proc),
                      n         = (x_sample.shape[2], x_sample.shape[1], int(x_sample.shape[0]/n_proc)),
                      N         = (x_sample.shape[2], x_sample.shape[1], x_sample.shape[0]),
                      analysis_interval         =5000, # Excel
                      visualization_interval    =timestep_max, # Silo
                      timestep_max              =timestep_max,
                      subphase_analysis_interval=timestep_max,
                      restart_interval          =timestep_max)
        
        raw_path = os.path.join(folder_rbc, "domain.raw")
        x_sample.astype(np.uint8).tofile(raw_path)
        
        
def create_simulation_force_condition(x_sample, output_root, folder_base, reflect=True, outlet_layers=0, bc=0, n_proc=4, include_walls=False):
    
    x_sample = x_sample.copy() 
    
    if x_sample.shape[0]%n_proc!=0: raise Exception(f"Domain length must be divisible by n_proc={n_proc}")
    
    # Transform x_sample for simulation:
    if include_walls:
        x_sample[:, :, 0]   = 0
        x_sample[:, :, -1]  = 0
        x_sample[:, 0, :]   = 0
        x_sample[:, -1, :]  = 0
    
    
    # Sanity Checks:
    if not is_percolating(x_sample, axis=0): print("Sample failed to percolate and got removed.")
    else:

        
        folder_rbc  = os.path.join(output_root, folder_base)
        os.makedirs(folder_rbc, exist_ok=True)
        
        # Make periodic in z directipn
        if reflect:
            flipped             = np.flip(x_sample, axis=0)
            x_sample            = np.concatenate([x_sample, flipped], axis=0)
        
        # Calculate a force that ensures the desired conditions of Reynolds, Viscosity and Density
        Re   = 0.1
        tau  = 1.5
        Dens = 1.0
        force_z = force_calculation(
            x_sample,
            tau     = tau,
            Re      = Re,
            Dens    = Dens
        )
        
        timestep_max = timestep_calculation(    
                matriz_binaria  = x_sample,
                tau             = tau,
                Re              = Re,
                Dens            = Dens,
                safety_factor   = 10.0
        )
        
        # --- Save 3D domain as .raw ---
        write_lbpm_db(path  =folder_rbc, 
                      tau   =tau,
                      bc    =bc,
                      fz    =force_z,   
                      nproc = (1, 1, n_proc),
                      n     = (x_sample.shape[2], x_sample.shape[1], int(x_sample.shape[0]/n_proc)),
                      N     = (x_sample.shape[2], x_sample.shape[1], x_sample.shape[0]),
                      outlet_layers                 = (0,0,outlet_layers),
                      timestep_max                  = timestep_max,
                      analysis_interval             = 5000, # Excel
                      visualization_interval        = timestep_max, # Silo
                      subphase_analysis_interval    = timestep_max,
                      restart_interval              = timestep_max)
        
        raw_path = os.path.join(folder_rbc, "domain.raw")
        x_sample.astype(np.uint8).tofile(raw_path)
        
            
        # Include guiding image in each folder
        plt.figure()
        plt.imshow(x_sample[0], cmap='binary', interpolation='none')
        plt.axis('off')
        plt.tight_layout(pad=0) 
        plt.savefig(folder_rbc+"/domain.svg", bbox_inches='tight')
        plt.close()
