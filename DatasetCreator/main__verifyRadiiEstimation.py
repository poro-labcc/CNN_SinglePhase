import os
import re
from typing import List, Tuple, Union
import torch
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
import h5py
from numpy.random import default_rng
import utils
import matplotlib.pyplot as plt
import porespy as ps

# -------------------------------------------------------------------
# 1) Directory helpers
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str, sample_dir_pattern: str) -> List[str]:
    pattern = re.compile(sample_dir_pattern)
    samples: List[Tuple[int, str]] = []
    if not os.path.exists(base_dir):
        return []
        
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path): continue
        m = pattern.match(name)
        if m:
            num_part = int(m.group(1))
            samples.append((num_part, name))
    samples.sort(key=lambda t: t[0])
    return [name for _, name in samples]

def read_raw_volume(raw_path: str, shape: Tuple[int, int, int], dtype: np.dtype, order: str = "C") -> np.ndarray:
    flat = np.fromfile(raw_path, dtype=dtype)
    return flat.reshape(shape, order=order)

def get_latest_vis_summary_path(sample_dir: str) -> str:
    vis_pattern = re.compile(r"^vis(\d+)$")
    vis_candidates: List[Tuple[int, str]] = []
    for name in os.listdir(sample_dir):
        full_path = os.path.join(sample_dir, name)
        if os.path.isdir(full_path):
            m = vis_pattern.match(name)
            if m: vis_candidates.append((int(m.group(1)), full_path))
    if not vis_candidates: raise RuntimeError(f"No 'visY' in: {sample_dir}")
    vis_candidates.sort(key=lambda t: t[0])
    return os.path.join(vis_candidates[-1][1], "summary.pvti")

def read_summary_pvti(summary_path: str) -> pv.DataSet:
    return pv.read(summary_path)

# -------------------------------------------------------------------
# 2) Geometry helpers
# -------------------------------------------------------------------


def force_calculation_from_R(
    R:              Union[float, int],
    tau:            Union[float, int],
    Re:             float = 0.1,
    Dens:           float = 1.0,
) -> float:
    Visc            = (tau - 0.5) / 3.0
    Fx              = (Re * 8.0 * (Visc ** 2)) / (Dens * (R ** 3))
    return Fx

def permeability_calculation(out:  np.ndarray,  
                             inp:  np.ndarray,  
                             tau:  float = 1.5,   
                             Re:   float = 0.1, 
                             dens: float = 1.0) -> np.ndarray:
    B = out.shape[0]
    k_lattice = np.zeros(B, dtype=np.float32)
    visc = (tau - 0.5) / 3.0
    for b in range(B):
        r_max = np.max(inp[b])
        force_z = force_calculation_from_R(R=r_max, tau=tau, Re=Re, Dens=dens)
        u_z = out[b, 0]
        u_mean = np.mean(u_z)
        k_lattice[b] = 1013 * (u_mean * visc) / (dens * force_z)
    return k_lattice

def find_directional_local_maxima(dist_transform):
    dt = np.asarray(dist_transform)
    max_z = np.zeros_like(dt, dtype=bool)  
    max_y = np.zeros_like(dt, dtype=bool)  
    max_x = np.zeros_like(dt, dtype=bool)  
    
    max_z[1:-1, :, :] = (dt[1:-1, :, :] > dt[:-2, :, :]) & (dt[1:-1, :, :] > dt[2:, :, :])
    max_y[:, 1:-1, :] = (dt[:, 1:-1, :] > dt[:, :-2, :]) & (dt[:, 1:-1, :] > dt[:, 2:, :])
    max_x[:, :, 1:-1] = (dt[:, :, 1:-1] > dt[:, :, :-2]) & (dt[:, :, 1:-1] > dt[:, :, 2:])
    
    max_count = max_z.astype(np.int8) + max_y.astype(np.int8) + max_x.astype(np.int8)
    local_maxima_mask = (max_count >= 2) & (dt > 0)
    return local_maxima_mask    
    
# -------------------------------------------------------------------
# 2) Main Builder with HDF5 and Augmentation
# -------------------------------------------------------------------

simulations_folders  = {
    "Cylin Grain": "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_CylinGrain_120_120_120/",
    "Cylin Pore":  "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_CylinPore_120_120_120/",
    "Sph Pore":    "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_SphPore_120_120_120/",
    "Sph Grain":   "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_SphGrain_120_120_120/",
    #"Bentheimer":  "/home/gabriel/remote/hal/dissertacao/simulations/Test_Oliveira_Bentheimer_120_120_120/Samples/",
    #"Brown":       "/home/gabriel/remote/hal/dissertacao/simulations/Test_Oliveira_Brown_120_120_120/Samples/",
    #"Parker":      "/home/gabriel/remote/hal/dissertacao/simulations/Test_Oliveira_Parker_120_120_120/Samples/",
    #"Berea":       "/home/gabriel/remote/hal/dissertacao/simulations/Test_Oliveira_Berea_120_120_120/Samples/",
}
sample_dir_pattern  = r"^Sample_(\d+)$"
raw_name            = "domain.raw"
raw_shape           = (120, 120, 120)
raw_dtype           = np.uint8

# Normalization Parameters
norm_cte = 0.2
tau      = 1.5
Re       = 0.1
dens     = 1.0

# Dictionary to store metrics grouped by folder name for plotting
results_by_folder = {}

for folder_name, folder_path in simulations_folders.items():
    results_by_folder[folder_name] = {
        'true_thicknesses':     [],
        'r_est_max_edt_vals':   [],
        'r_est_local_vals':     [],
        'k_sim':                [],  # True simulation permeability
        'k_max_edt':            [],  # Permeability calculated using R_est = 0.65*Max
        'k_local_max':          [],  # Permeability calculated using R_est = Local Max
        'sample_labels':        []
    }
    
    base_dir = folder_path if os.path.isabs(folder_path) else os.path.join(os.getcwd(), folder_path)
    sample_dirs = list_sample_dirs(base_dir, sample_dir_pattern)
    
    if not sample_dirs:
        continue
        
    print(f"\n=============================================================")
    print(f"Processing Folder: {folder_name} ({len(sample_dirs)} samples found)")
    print(f"=============================================================")

    for sample_name in sample_dirs:
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_name)
        
        # Load Original Data
        vol_orig     = read_raw_volume(raw_path, raw_shape, raw_dtype)
        summary_path = get_latest_vis_summary_path(sample_dir)
        mesh         = read_summary_pvti(summary_path)
        
        vx_orig = mesh["Velocity_x"].reshape(raw_shape, order="C")
        vy_orig = mesh["Velocity_y"].reshape(raw_shape, order="C")
        vz_orig = mesh["Velocity_z"].reshape(raw_shape, order="C")
        
        porous_mask = (vol_orig == 1) 

        # Structural calculations
        edt = distance_transform_edt(porous_mask).astype("float32")
        
        # Metric 1: 0.65 * Max EDT
        R_est_1     = np.max(edt)*0.5 # Constant (3**(1/3)) mannualy adjusted to match
        
        # Metric 2: Mean of Directional Local Maxima
        maxima_mask = find_directional_local_maxima(edt)
        maximas     = edt[maxima_mask]
        R_est_2     = np.mean(maximas)* (10/7.5) # Constant (10/7.5) mannualy adjusted
        
        # Metric 3: True Mean Local Thickness (PoreSpy)
        thickness_map = ps.filters.local_thickness(porous_mask)
        mean_local_thickness = np.mean(thickness_map[porous_mask])
        
        
        

        
        # --- Base Variables for Permeability ---
        visc    = (tau-0.5)/3
        u_mean  = np.mean(vz_orig)

        por     = np.mean(porous_mask)
        # True Simulation Force & Permeability # (mDa )
        force = utils.force_calculation(porous_mask, tau=tau, Re=Re)
        k_sim =  u_mean *  (1013 * visc / (dens * force))
        
        # Permeability using Force derived from R**2 (mDa)
        k_max_edt   = 1013 *  R_est_1**2  
        
        # Permeability using Force derived from R**2 (mDa)
        k_local_max =  1013 * R_est_2**2
        
        tort        = 1 + 0.8*(1-por)
        
        #k_max_edt   = 1013 * (0.125 * R_est_1**2 * por / tort**2)   
        
        #k_local_max = 1013 * (0.125 * R_est_2**2 * por / tort**2)

        # Append data to this folder's tracking lists
        results_by_folder[folder_name]['true_thicknesses'].append(mean_local_thickness)
        results_by_folder[folder_name]['r_est_max_edt_vals'].append(R_est_1)
        results_by_folder[folder_name]['r_est_local_vals'].append(R_est_2)
        
        results_by_folder[folder_name]['k_sim'].append(k_sim)
        results_by_folder[folder_name]['k_max_edt'].append(k_max_edt)
        results_by_folder[folder_name]['k_local_max'].append(k_local_max)
        
        
        print(f"[{folder_name} | {sample_name}]:)")
        print(f"  - Thickness: {mean_local_thickness:.4f}; R_max: {R_est_1:.4f}; R_m_max: {R_est_2:.4f}")
        print(f"  - k_sim: {k_sim:.4f}; k_max_edt: {k_max_edt:.4f}; k_local_max: {k_local_max:.4f}")
        print()

# -------------------------------------------------------------------
# 3) Plotting the R_est and Permeability Comparisons
# -------------------------------------------------------------------

# Set up a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
(ax1, ax2), (ax3, ax4) = axs

# --- Data Aggregation for Axes Limits ---
all_true, all_max_edt, all_local_max = [], [], []
all_k_sim, all_k_max, all_k_local = [], [], []

for data in results_by_folder.values():
    if not data['true_thicknesses']: continue
    all_true.extend(data['true_thicknesses'])
    all_max_edt.extend(data['r_est_max_edt_vals'])
    all_local_max.extend(data['r_est_local_vals'])
    
    all_k_sim.extend(data['k_sim'])
    all_k_max.extend(data['k_max_edt'])
    all_k_local.extend(data['k_local_max'])

if all_true:  
    # Limits for Row 1 (Thickness)
    min_t = min(min(all_true), min(all_max_edt), min(all_local_max)) * 0.9
    max_t = max(max(all_true), max(all_max_edt), max(all_local_max)) * 1.1
    line_t = [min_t, max_t]
    
    ax1.plot(line_t, line_t, 'k--', label='y = x', zorder=1)
    ax2.plot(line_t, line_t, 'k--', label='y = x', zorder=1)
    ax1.set_xlim(min_t, max_t); ax1.set_ylim(min_t, max_t)
    ax2.set_xlim(min_t, max_t); ax2.set_ylim(min_t, max_t)

    # Limits for Row 2 (Permeability)
    min_k = min(min(all_k_sim), min(all_k_max), min(all_k_local)) * 0.9
    max_k = max(max(all_k_sim), max(all_k_max), max(all_k_local)) * 1.1
    line_k = [min_k, max_k]
    
    ax3.plot(line_k, line_k, 'k--', label='y = x', zorder=1)
    ax4.plot(line_k, line_k, 'k--', label='y = x', zorder=1)
    ax3.set_xlim(min_k, max_k); ax3.set_ylim(min_k, max_k)
    ax4.set_xlim(min_k, max_k); ax4.set_ylim(min_k, max_k)

# --- Plotting the Scatter Points ---
for folder_name, data in results_by_folder.items():
    if not data['true_thicknesses']: continue
        
    true_arr = np.array(data['true_thicknesses'])
    max_edt_arr = np.array(data['r_est_max_edt_vals'])
    local_max_arr = np.array(data['r_est_local_vals'])
    
    k_sim_arr = np.array(data['k_sim'])
    k_max_arr = np.array(data['k_max_edt'])
    k_local_arr = np.array(data['k_local_max'])

    # Row 1: Thickness
    ax1.scatter(true_arr, max_edt_arr, alpha=0.7, edgecolor='k', label=folder_name, zorder=2)
    ax2.scatter(true_arr, local_max_arr, alpha=0.7, edgecolor='k', label=folder_name, zorder=2)
    
    # Row 2: Permeability
    ax3.scatter(k_sim_arr, k_max_arr, alpha=0.7, edgecolor='k', label=folder_name, zorder=2)
    ax4.scatter(k_sim_arr, k_local_arr, alpha=0.7, edgecolor='k', label=folder_name, zorder=2)

# --- Subplot Styling ---
ax1.set_title('R_est (0.65 * Max EDT) vs True Local Thickness')
ax1.set_xlabel('True Mean Local Thickness (Voxels)')
ax1.set_ylabel('R_est: 0.65 * Max EDT (Voxels)')

ax2.set_title('R_est (Local Maxima Mean) vs True Local Thickness')
ax2.set_xlabel('True Mean Local Thickness (Voxels)')
ax2.set_ylabel('R_est: Mean Local Maxima (Voxels)')

ax3.set_title('Calculated Permeability (0.65*Max) vs Sim Permeability')
ax3.set_xlabel('True Sim Permeability ($K_{sim}$)')
ax3.set_ylabel('Calculated Permeability ($K_{0.65Max}$)')

ax4.set_title('Calculated Permeability (Local Max) vs Sim Permeability')
ax4.set_xlabel('True Sim Permeability ($K_{sim}$)')
ax4.set_ylabel('Calculated Permeability ($K_{LocalMax}$)')

for ax in axs.flat:
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Comparison of Length Scales and Resulting Permeabilities', fontsize=18, y=0.98)
plt.tight_layout()

plt.show()