import os
import re
from typing import List, Tuple
import torch
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
import h5py
from numpy.random import default_rng
import utils
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# 1) Helpers and Augmentation Functions
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str, sample_dir_pattern: str) -> List[str]:
    pattern = re.compile(sample_dir_pattern)
    samples: List[Tuple[int, str]] = []
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

# --- Danny Ko's Augmentation Logic ---

def shift_augmentation(my_solid, my_vel, shift_range, vel_dir):
    # Assuming shift_range contains [shift_z, shift_y, shift_x]
    # For simplicity, let's say we are shifting your true X (axis=2) and true Y (axis=1)
    
    my_aug_solid = np.ones_like(my_solid)
    my_aug_vel = np.ones_like(my_vel)
    
    # Let's extract shifts specifically for your Y (axis=1) and X (axis=2)
    shift_y = shift_range[0] 
    shift_x = shift_range[1] 

    # --- Shift True Y (axis=1) ---
    if shift_y < 0:
        my_aug_solid[:, :shift_y, :]    = my_solid[:, -1*shift_y:, :]
        my_aug_solid[:, shift_y:, :]    = np.flip(my_solid, axis=1)[:, :-1*shift_y, :]
        my_aug_vel[:, :shift_y, :]      = my_vel[:, -1*shift_y:, :]
        my_aug_vel[:, shift_y:, :]      = np.flip(-1*my_vel if vel_dir == 'y' else my_vel, axis=1)[:, :-1*shift_y, :]
    elif shift_y > 0:
        my_aug_solid[:, :shift_y, :]    = np.flip(my_solid, axis=1)[:, -1*shift_y:, :]
        my_aug_solid[:, shift_y:, :]    = my_solid[:, :-1*shift_y, :]
        my_aug_vel[:, :shift_y, :]      = np.flip(-1*my_vel if vel_dir == 'y' else my_vel, axis=1)[:, -1*shift_y:, :]
        my_aug_vel[:, shift_y:, :]      = my_vel[:, :-1*shift_y, :]
    else:
        my_aug_solid, my_aug_vel = my_solid, my_vel

    my_solid, my_vel = my_aug_solid, my_aug_vel
    my_aug_solid, my_aug_vel = np.ones_like(my_solid), np.ones_like(my_vel)

    # --- Shift True X (axis=2) ---
    if shift_x < 0:
        my_aug_solid[:, :, :shift_x]    = my_solid[:, :, -1*shift_x:]
        my_aug_solid[:, :, shift_x:]    = np.flip(my_solid, axis=2)[:, :, :-1*shift_x]
        my_aug_vel[:, :, :shift_x]      = my_vel[:, :, -1*shift_x:]
        my_aug_vel[:, :, shift_x:]      = np.flip(-1*my_vel if vel_dir == 'x' else my_vel, axis=2)[:, :, :-1*shift_x]
    elif shift_x > 0:
        my_aug_solid[:, :, :shift_x]    = np.flip(my_solid, axis=2)[:, :, -1*shift_x:]
        my_aug_solid[:, :, shift_x:]    = my_solid[:, :, :-1*shift_x]
        my_aug_vel[:, :, :shift_x]      = np.flip(-1*my_vel if vel_dir == 'x' else my_vel, axis=2)[:, :, -1*shift_x:]
        my_aug_vel[:, :, shift_x:]      = my_vel[:, :, :-1*shift_x]
    else:
        my_aug_solid, my_aug_vel = my_solid, my_vel

    return my_aug_solid, my_aug_vel

def flip_augmentation(my_solid, my_vel, vel_dir, axis):
    my_aug_solid = np.flip(my_solid, axis=axis)
    if(vel_dir[0] == 'x'):
        if(axis == 2):
          my_aug_vel = np.flip(-1*my_vel, axis=axis)
        else:
          my_aug_vel = np.flip(my_vel, axis=axis)
    elif(vel_dir[0] == 'y'):
        if(axis == 1):
          my_aug_vel = np.flip(-1*my_vel, axis=axis)
        else:
          my_aug_vel = np.flip(my_vel, axis=axis)
    else:
        my_aug_vel = np.flip(my_vel, axis=axis)
        
    return my_aug_solid, my_aug_vel

# -------------------------------------------------------------------
# 2) Main Builder with HDF5 and Augmentation
# -------------------------------------------------------------------


simulations_folder  = "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_Danny_SphPore_120_120_120/"
output_path         = "../../NN_Datasets/Valid_Danny_SphPore_DAug_DNorm.h5"
sample_dir_pattern  = r"^Sample_(\d+)$"
raw_name            = "mod_domain.raw"
raw_shape           = (120, 120, 120)
raw_dtype           = np.uint8

# Augmentation Parameters
augment             = True
augGen_seed         = 10
aug_iter            = 30
shift_range         = 2
flip_range          = 5
# Normalization Parameters
tau                 = 1.5
Re                  = 0.1

base_dir            = os.path.join(os.getcwd(), simulations_folder)
sample_dirs         = list_sample_dirs(base_dir, sample_dir_pattern)
rnd_num_gen         = default_rng(augGen_seed)

min_values  = []
max_values  = []
mean_values = []

z_values = np.array([])
y_values = np.array([])
x_values = np.array([])

output_dir = os.path.dirname(output_path)
if output_dir: os.makedirs(output_dir, exist_ok=True)

with h5py.File(output_path, "w") as f:
    D, H, W = raw_shape
    max_points = int(D * H * W) # Adjust if you want a lower fraction

    # Dataset Initialization (expandable)
    vel_x_ds = f.create_dataset("vel_x", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    vel_y_ds = f.create_dataset("vel_y", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    vel_z_ds = f.create_dataset("vel_z", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    press_ds = f.create_dataset("press", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    coorX_ds = f.create_dataset("coorX", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    coorY_ds = f.create_dataset("coorY", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    coorZ_ds = f.create_dataset("coorZ", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    edt_ds   = f.create_dataset("edt",   (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    n_valid_ds = f.create_dataset("n_valid", (0,), maxshape=(None,), dtype="int64")
    sample_names_ds = f.create_dataset("sample_names", (0,), maxshape=(None,), dtype=h5py.string_dtype())

    global_idx = 0

    for sample_name in sample_dirs:
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_name)
        
        try:
            f.attrs["raw_shape"]   = raw_shape
            f.attrs["vel_dtype"]   = "float32"
            f.attrs["coorX_dtype"] = "uint8"
            f.attrs["coorY_dtype"] = "uint8"
            f.attrs["coorZ_dtype"] = "uint8"
            f.attrs["edt_dtype"]   = "float32"
            f.attrs["max_points"]  = max_points
            
            # 1. Load Original Data
            vol_orig        = read_raw_volume(raw_path, raw_shape, raw_dtype)
            summary_path    = get_latest_vis_summary_path(sample_dir)
            mesh            = read_summary_pvti(summary_path)
            
            vx_orig = mesh["Velocity_x"].reshape(raw_shape, order="C")
            vy_orig = mesh["Velocity_y"].reshape(raw_shape, order="C")
            vz_orig = mesh["Velocity_z"].reshape(raw_shape, order="C")
            if "Pressure" in mesh.array_names:
                print("Mesh contains pressure data.")
                pr_orig = mesh["Pressure"].reshape(raw_shape, order="C")
            else:
                print("Pressure data not found.")
                pr_orig        = np.zeros_like(vz_orig)
            
            porous_mask_orig = (vol_orig == 1) 

            # Velocity Normalization
            vx_norm = utils.danny_normalization_vel(vx_orig, porous_mask_orig)
            vy_norm = utils.danny_normalization_vel(vy_orig, porous_mask_orig)
            vz_norm = utils.danny_normalization_vel(vz_orig, porous_mask_orig)
            # Pressure Normalization
            pr_norm = utils.silveira_normalization_pres(pr_orig, porous_mask_orig)
                
                
            print("Mins:  ",np.min(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Means: ",np.mean(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Devs:  ",np.std(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Maxs:  ",np.max(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("-------------------------------------------------------------")            
        
        
            # Generate random augmentation parameters
            shift_val   = D // shift_range  # D=120 for your data
            rnd_shifts  = rnd_num_gen.integers(low=-shift_val, high=shift_val, size=(aug_iter, 2), endpoint=True)
            rnd_flips   = rnd_num_gen.integers(low=0, high=10, size=(aug_iter, 2))
            
            print(f"Processing {sample_name} with {aug_iter} augmentations...")
            
            for j in range(aug_iter):
                # Apply Shift (shift_range=[shift_x, shift_y])
                curr_vol, curr_vx = shift_augmentation(vol_orig, vx_norm, rnd_shifts[j], 'x')
                _, curr_vy        = shift_augmentation(vol_orig, vy_norm, rnd_shifts[j], 'y')
                _, curr_vz        = shift_augmentation(vol_orig, vz_norm, rnd_shifts[j], 'z')
                _, curr_pr        = shift_augmentation(vol_orig, pr_norm, rnd_shifts[j], 'none')
                
                # Apply Flip Axis 0 (X direction)
                if rnd_flips[j, 0] >= flip_range:
                    curr_vol, curr_vx = flip_augmentation(curr_vol, curr_vx, 'x', 2)
                    _, curr_vy        = flip_augmentation(curr_vol, curr_vy, 'y', 2)
                    _, curr_vz        = flip_augmentation(curr_vol, curr_vz, 'z', 2)
                    _, curr_pr        = flip_augmentation(curr_vol, curr_pr, 'none', 2)
                
                # Apply Flip Axis 1 (Y direction)
                if rnd_flips[j, 1] >= flip_range:
                    curr_vol, curr_vx = flip_augmentation(curr_vol, curr_vx, 'x', 1)
                    _, curr_vy        = flip_augmentation(curr_vol, curr_vy, 'y', 1)
                    _, curr_vz        = flip_augmentation(curr_vol, curr_vz, 'z', 1)
                    _, curr_pr        = flip_augmentation(curr_vol, curr_pr, 'none', 1)
                                                                          
                
                
                
                # 4. Geometry-based calculations (EDT and Mask) on augmented volume
                porous_mask = (curr_vol == 1)
                if not np.any(porous_mask): continue

                edt_full = distance_transform_edt(porous_mask).astype("float32")
                coords_k, coords_j, coords_i,  = np.where(porous_mask)
                N_points = coords_k.size

                # 5. Flatten and Pad
                # (Re-using your padding logic)
                vx_row = np.zeros(max_points, dtype="float32")
                vy_row = np.zeros(max_points, dtype="float32")
                vz_row = np.zeros(max_points, dtype="float32")
                pr_row = np.zeros(max_points, dtype="float32")
                cX_row = np.zeros(max_points, dtype="uint8")
                cY_row = np.zeros(max_points, dtype="uint8")
                cZ_row = np.zeros(max_points, dtype="uint8")
                ed_row = np.zeros(max_points, dtype="float32")

                vx_row[:N_points] = curr_vx[porous_mask]
                vy_row[:N_points] = curr_vy[porous_mask]
                vz_row[:N_points] = curr_vz[porous_mask]
                pr_row[:N_points] = curr_pr[porous_mask]
                cX_row[:N_points] = coords_i.astype(np.uint8)
                cY_row[:N_points] = coords_j.astype(np.uint8)
                cZ_row[:N_points] = coords_k.astype(np.uint8)
                ed_row[:N_points] = edt_full[porous_mask]

                # 6. Save to HDF5
                for ds, data in zip([vel_x_ds, vel_y_ds, vel_z_ds, press_ds, coorX_ds, coorY_ds, coorZ_ds, edt_ds],
                                    [vx_row,   vy_row,   vz_row,   pr_row,   cX_row,   cY_row,   cZ_row,   ed_row]):
                    ds.resize((global_idx + 1, max_points))
                    ds[global_idx, :] = data
                
                n_valid_ds.resize((global_idx + 1,))
                sample_names_ds.resize((global_idx + 1,))
                n_valid_ds[global_idx] = N_points
                sample_names_ds[global_idx] = f"{sample_name}_aug_{j}"
                
                global_idx += 1

        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}")
    
   
    f.attrs['tau_used']     = tau
    f.attrs['re_used']      = Re

    print(f"Finished. Total augmented samples written: {global_idx}")
    


