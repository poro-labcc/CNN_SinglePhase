import os
import re
from typing import List, Tuple
import numpy as np
import pyvista as pv
from   scipy.ndimage import distance_transform_edt
import h5py
import utils 
import matplotlib.pyplot as plt
import random
from scipy.ndimage import maximum_filter
# -------------------------------------------------------------------
# 1) Helpers for folder / file discovery
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str, sample_dir_pattern: str) -> List[str]:
    """
    List all directories named 'DeePore_Sample_XXXXX' inside base_dir,
    sorted by the numeric suffix (e.g. 00010 -> 10).

    Returns a list of folder names (not full paths).
    """
    pattern = re.compile( sample_dir_pattern)

    samples: List[Tuple[int, str]] = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue

        m = pattern.match(name)
        if m:
            num_part = int(m.group(1))
            samples.append((num_part, name))

    samples.sort(key=lambda t: t[0])
    return [name for _, name in samples]


def get_raw_path(sample_dir: str, raw_filename: str) -> str:
    """
    Return the full path to the raw file (e.g. domain.raw) inside the sample_dir.
    """
    raw_path = os.path.join(sample_dir, raw_filename)
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    return raw_path


def get_latest_vis_summary_path(sample_dir: str) -> str:
    """
    Inside sample_dir, find all subdirectories named 'visY' where Y is an integer.
    Select the highest Y and return the path to 'summary.pvti' inside it.
    """
    vis_pattern = re.compile(r"^vis(\d+)$")
    vis_candidates: List[Tuple[int, str]] = []

    for name in os.listdir(sample_dir):
        full_path = os.path.join(sample_dir, name)
        if not os.path.isdir(full_path):
            continue

        m = vis_pattern.match(name)
        if m:
            y = int(m.group(1))
            vis_candidates.append((y, full_path))

    if not vis_candidates:
        raise RuntimeError(f"No 'visY' subdirectories found in: {sample_dir}")

    # Pick highest Y
    vis_candidates.sort(key=lambda t: t[0])
    _, latest_vis_dir = vis_candidates[-1]

    summary_path = os.path.join(latest_vis_dir, "summary.pvti")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"'summary.pvti' not found in: {latest_vis_dir}")

    return summary_path

def save_pvti(output_path, vel_x, vel_y, vel_z, sign_dist, origin=(0,0,0), spacing=(1,1,1)):
    # Fix the extension if it's invalid
    if not output_path.endswith('.vti'):
        base = os.path.splitext(output_path)[0]
        output_path = f"{base}.vti"
        print(f"Adjusted output path to: {output_path}")

    # 1. Create the Grid
    # dimensions should be (nx, ny, nz)
    grid = pv.ImageData()
    grid.dimensions = vel_x.shape
    grid.spacing = spacing
    grid.origin = origin

    # 2. Add the data
    # Use 'F' order to ensure the (x, y, z) mapping matches VTK's internal memory
    grid.point_data["Velocity_x"] = vel_x.flatten(order="C")
    grid.point_data["Velocity_y"] = vel_y.flatten(order="C")
    grid.point_data["Velocity_z"] = vel_z.flatten(order="C")
    grid.point_data["SignDist"]   = sign_dist.flatten(order="C")

    # 3. Save
    grid.save(output_path)
    print("Write-back successful. Open this file in ParaView to verify.")
    
# -------------------------------------------------------------------
# 2) Reading the raw volume and the pvti
# -------------------------------------------------------------------

def read_raw_volume(
    raw_path: str,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    order: str = "C",
) -> np.ndarray:
    """
    Read a .raw file as a 3D NumPy array with the given shape and dtype.

    Parameters
    ----------
    raw_path : str
        Full path to the .raw file.
    shape : (nx, ny, nz)
        Shape of the 3D volume.
    dtype : np.dtype
        Data type stored in the raw file (e.g. np.uint8, np.float32).
    order : {'C', 'F'}
        Memory order used when reshaping.

    Returns
    -------
    np.ndarray
        3D array with the specified shape and dtype.
    """
    flat = np.fromfile(raw_path, dtype=dtype)
    expected_size = int(np.prod(shape))
    if flat.size != expected_size:
        raise ValueError(
            f"Raw file size mismatch for {raw_path}: "
            f"found {flat.size} elements, expected {expected_size} "
            f"for shape {shape}"
        )

    return flat.reshape(shape, order=order)


def read_summary_pvti(summary_path: str) -> pv.DataSet:
    """
    Read a summary.pvti file as a PyVista object.
    """
    return pv.read(summary_path)


# -------------------------------------------------------------------
# 3) Main high-level functions
# -------------------------------------------------------------------

# True if everything is okay
def sanity_check(vol, vel, solid_value=0):
    solid_mask = (vol == solid_value)
    return not np.any(vel[solid_mask] != 0)


def numpy_align(array, target_zaxis=0, target_xaxis=2):
    aligned   = np.rot90(array, axes=(target_zaxis, target_xaxis))   
    aligned   = np.flip(aligned, axis=target_zaxis)  
    return aligned




# ---- main builder using HDF5 ----

output_path         = "../../NN_Datasets/Train_Danny_SphPore_120_120_120.h5"
simulations_folder  = "/home/gabriel/remote/hal/dissertacao/Simulations/Train_Danny_SphPore_120_120_120/"
sample_dir_pattern  = r"^Sample_(\d+)$"
raw_name            = "mod_domain.raw"
raw_shape           = (120, 120, 120)  # (D, H, W)
raw_dtype           = np.uint8
N_samples           = None # None for all
max_porosity        = 1.0 
# Normalization Parameters
norm_cte            = 0.2
tau                 = 1.5
Re                  = 0.1

base_dir            = os.path.join(os.getcwd(), simulations_folder)
sample_dirs         = list_sample_dirs(base_dir, sample_dir_pattern)
if len(sample_dirs)==0: raise Exception(f"No simulations found in {base_dir} with pattern {sample_dir_pattern}")

# Selection of data to plot (Sanity check)
selected_samples = random.sample(sample_dirs, min(9,len(sample_dirs)))
selected_samples.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
sample_plot_data = []
sample_plot_data_pr = []

# Create dataset
output_dir = os.path.dirname(output_path)
if output_dir: os.makedirs(output_dir, exist_ok=True)
    
with h5py.File(output_path, "w") as f:
    D, H, W     = raw_shape

    # Máximo de pontos porosos por amostra (50% de 256^3)
    max_points  = int((D * H * W) *max_porosity)  # 8_388_608 para 256^3

    # Metadados gerais
    f.attrs["description"] = (
        "LBPM velocity + EDT only where edt>0, "
        "fixed-size per sample with max 50% porosity"
    )
    f.attrs["raw_shape"]   = raw_shape
    f.attrs["vel_dtype"]   = "float32"
    f.attrs["coorX_dtype"] = "uint8"
    f.attrs["coorY_dtype"] = "uint8"
    f.attrs["coorZ_dtype"] = "uint8"
    f.attrs["edt_dtype"]   = "float32"
    f.attrs["max_points"]  = max_points

    # Create datasets for each information, expansible in samples dimension
    # Velocity Z
    vel_z_ds = f.create_dataset(
        "vel_z",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float32",
        chunks=(1, max_points),
    )
    # Velocity Y
    vel_y_ds = f.create_dataset(
        "vel_y",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float32",
        chunks=(1, max_points),
    )
    # Velocity X
    vel_x_ds = f.create_dataset(
        "vel_x",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float32",
        chunks=(1, max_points),
    )
    press_ds = f.create_dataset(
        "press",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float32",
        chunks=(1, max_points),
    )
    # Coordinates (x,y,z)
    coorZ_ds = f.create_dataset(
        "coorZ",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    coorY_ds = f.create_dataset(
        "coorY",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    coorX_ds = f.create_dataset(
        "coorX",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="uint8",
        chunks=(1, max_points),
    )
    # Distance transform
    edt_ds = f.create_dataset(
        "edt",
        shape=(0, max_points),
        maxshape=(None, max_points),
        dtype="float32",
        chunks=(1, max_points),
    )
    # Number of porous voxels
    n_valid_ds = f.create_dataset(
        "n_valid",
        shape=(0,),
        maxshape=(None,),
        dtype="int64",
    )
    # Sample name
    string_dt = h5py.string_dtype(encoding="utf-8")
    sample_names_ds = f.create_dataset(
        "sample_names",
        shape=(0,),
        maxshape=(None,),
        dtype=string_dt,
    )
    
    sample_count = 0

    for sample_name in sample_dirs:
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_name)
        
        if not os.path.isfile(raw_path):
            print(f"[SKIP] {sample_name}: no {raw_name}")
            continue

        try:
            # Load Binary domain
            vol = read_raw_volume(raw_path, raw_shape, raw_dtype)

            # Load Simulation data
            summary_path = get_latest_vis_summary_path(sample_dir)
            mesh         = read_summary_pvti(summary_path)
            vel_x        = mesh["Velocity_x"].reshape(raw_shape, order="C")
            vel_y        = mesh["Velocity_y"].reshape(raw_shape, order="C")
            vel_z        = mesh["Velocity_z"].reshape(raw_shape, order="C")
            sign_dist    = mesh["SignDist"].reshape(raw_shape,   order="C")
        
            
            if "Pressure" in mesh.array_names:
                print("Mesh contains pressure data.")
                press = mesh["Pressure"].reshape(raw_shape, order="C")
            else:
                print("Pressure data not found.")
                press        = np.ones_like(vel_z)/3

            porous_mask     = (vol == 1)
            # Recalculate the distance transform with Scipy
            edt_full        = distance_transform_edt(porous_mask).astype("float32")
            
            # Normalization
            visc      = (tau-0.5)/3
            force     = utils.force_calculation(porous_mask, tau=tau, Re=Re)
            pore_diam = 2*0.65*np.max(edt_full)
            porosity  = np.count_nonzero(porous_mask)/(raw_shape[0]*raw_shape[1]*raw_shape[2])
            perm_est  = pore_diam**2

            # Velocity Normalization
            vx_norm   = vel_x*visc / (force*norm_cte*perm_est)
            vy_norm   = vel_y*visc / (force*norm_cte*perm_est)
            vz_norm   = vel_z*visc / (force*norm_cte*perm_est)
            # Pressure Normalization
            delta_p     = utils.pressure_calculation(porous_mask, tau=tau, Re=Re)
            p_mean      = (2+3*delta_p)/6
            delta_p_new = 0.2
            p_mean_new  = 0.15
            pr_norm     = ((press -p_mean)/delta_p)*delta_p_new + p_mean_new
            print("Pore diameter: ", pore_diam)
            print("Porosity:      ", porosity)
            print("Applied force: ", force)
            print("Mins:          ", np.min(vx_norm), np.min(vy_norm), np.min(vz_norm))
            print("Means:         ", np.mean(vx_norm),np.mean(vy_norm),np.mean(vz_norm))
            print("Devs:          ", np.std(vx_norm), np.std(vy_norm), np.std(vz_norm))
            print("Maxs:          ", np.max(vx_norm), np.max(vy_norm), np.max(vz_norm))
            print("-------------------------------------------------------------")  
            
            
            if sample_name in selected_samples:
                mag_sample = np.sqrt(vx_norm[porous_mask]**2 + vy_norm[porous_mask]**2 + vz_norm[porous_mask]**2)
                sample_plot_data.append((sample_name, mag_sample))
                sample_plot_data_pr.append((sample_name, pr_norm[porous_mask]))
            
            if not np.any(porous_mask):
                print(f"[SKIP] {sample_name}: no pore space (SignDist > 0)")
                continue

        
            # Get indexes from porous cells
            k, j, i = np.where(porous_mask)
            N_points = k.size
            
            # Check if every porous cell can be stores in 'max_points' columns
            if N_points > max_points:
                raise RuntimeError(
                    f"Sample {sample_name} has {N_points} pore points, "
                    f"exceeds max_points={max_points}"
                )

            # Flatten data from porous region
            vel_z_flat = vz_norm[porous_mask].astype(np.float32)
            vel_y_flat = vy_norm[porous_mask].astype(np.float32)
            vel_x_flat = vx_norm[porous_mask].astype(np.float32)
            press_flat = pr_norm[porous_mask].astype(np.float32)
            edt_flat   = edt_full[porous_mask].astype(np.float32)
            # Type convertion
            i_coords = i.astype(np.uint8)
            j_coords = j.astype(np.uint8)
            k_coords = k.astype(np.uint8)

            # coords (N_points, 3) com [k, j, i]
            

            # --- 3) Padding para tamanho fixo max_points ---
            vel_z_row   = np.zeros(max_points, dtype=np.float32) 
            vel_y_row   = np.zeros(max_points, dtype=np.float32)
            vel_x_row   = np.zeros(max_points, dtype=np.float32)
            press_row   = np.zeros(max_points, dtype=np.float32)
            coorZ_row   = np.zeros(max_points, dtype=np.uint8)
            coorY_row   = np.zeros(max_points, dtype=np.uint8)
            coorX_row   = np.zeros(max_points, dtype=np.uint8)
            edt_row     = np.zeros(max_points, dtype=np.float32)
            
            # Fill data
            vel_z_row[:N_points]  = vel_z_flat
            vel_y_row[:N_points]  = vel_y_flat
            vel_x_row[:N_points]  = vel_x_flat
            press_row[:N_points]  = press_flat
            coorZ_row[:N_points]  = k_coords
            coorY_row[:N_points]  = j_coords
            coorX_row[:N_points]  = i_coords
            edt_row  [:N_points]    = edt_flat

            # --- 4) Aumenta datasets em 1 amostra e escreve a linha ---
            idx = sample_count

            vel_z_ds.resize((idx + 1, max_points))
            vel_y_ds.resize((idx + 1, max_points))
            vel_x_ds.resize((idx + 1, max_points))
            press_ds.resize((idx + 1, max_points))
            
            coorZ_ds.resize((idx + 1, max_points))
            coorY_ds.resize((idx + 1, max_points))
            coorX_ds.resize((idx + 1, max_points))
            
            edt_ds.resize((idx + 1, max_points))
            n_valid_ds.resize((idx + 1,))
            sample_names_ds.resize((idx + 1,))

            vel_z_ds[idx, :]  = vel_z_row
            vel_y_ds[idx, :]  = vel_y_row
            vel_x_ds[idx, :]  = vel_x_row
            press_ds[idx, :]  = press_row
            coorZ_ds[idx, :]  = coorZ_row
            coorY_ds[idx, :]  = coorY_row
            coorX_ds[idx, :]  = coorX_row
            edt_ds  [idx, :]  = edt_row

            n_valid_ds[idx]      = N_points
            sample_names_ds[idx] = sample_name

            sample_count += 1
            print(
                f"[OK] {sample_name}: {N_points} pontos porosos "
                f"(padded to {max_points})"
            )

        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}")
        
        if N_samples is not None and sample_count>=N_samples: break
    
    print(f"Finished {output_path}. Total samples written: {sample_count}")



# Allign with numpy notation            
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']



# -------------------------------------------------------------------
# Plotting Velocity Magnitude
# -------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=150)
axes_flat = axes.flatten()

for idx, (name, mag_data) in enumerate(sample_plot_data):
    ax = axes_flat[idx]
    
    # Remove zeros (solid) to see the velocity distribution clearly
    valid_mag = mag_data[mag_data > 1e-6] 
    
    ax.hist(valid_mag, bins=100, range=(0, 0.9), color='black', edgecolor='black', alpha=1)
    
    # Add mean line for better analysis
    mean_val = np.mean(valid_mag) if len(valid_mag) > 0 else 0
    ax.axvline(mean_val, color='grey', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.3f}')
    
    ax.set_title(f"Sample: {name}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Magnitude", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.2)
    ax.set_xlim(-0.1, 0.5)
    ax.legend(fontsize=16)

# Hide empty plots if you have < 9 samples
for i in range( len(sample_plot_data), 9):
    axes_flat[i].axis('off')

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# Plotting Pressure
# -------------------------------------------------------------------
fig_pr, axes_pr = plt.subplots(3, 3, figsize=(15, 12), dpi=150)
axes_pr_flat = axes_pr.flatten()

for idx, (name, pr_data) in enumerate(sample_plot_data_pr):
    ax = axes_pr_flat[idx]
    
    # Flatten just in case you kept the 3D array instead of pr_norm[porous_mask]
    valid_pr = pr_data.flatten()
    
    # If you didn't apply the porous_mask during append, solid pressure (0) 
    # evaluates to 1 + (0 - 0.33)/0.01 = -32. We filter out values below -30.
    valid_pr = valid_pr[valid_pr > -30] 
    
    # Histogram for pressure (auto-ranging bins based on data)
    ax.hist(valid_pr, bins=100, color='black', edgecolor='black', alpha=1.0)
    
    # Add mean line
    mean_val = np.mean(valid_pr) if len(valid_pr) > 0 else 0
    ax.axvline(mean_val, color='grey', linestyle='dashed', linewidth=1.5, label=f'Média: {mean_val:.3f}')
    
    # Formatting
    ax.set_title(f"Amostra: {idx}", fontsize=18, fontweight='bold')
    ax.set_xlabel("Pressão Normalizada", fontsize=18)
    ax.set_ylabel("Ocorrências", fontsize=18)
    ax.grid(axis='y', alpha=0.2)
    ax.set_xlim(-0.1, 0.5)
    ax.legend(fontsize=18, )

# Hide empty subplots if you have fewer than 9 samples
for i in range(len(sample_plot_data_pr), 9):
    axes_pr_flat[i].axis('off')

plt.tight_layout()
plt.show()