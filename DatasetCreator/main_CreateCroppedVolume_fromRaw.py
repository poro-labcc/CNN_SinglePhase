import os
import numpy         as np
import random
import utils


def get_non_overlapping_crops(volume_data, crop_D, crop_H, crop_W):
    """
    Calculates all possible non-overlapping 3D crops (cubes/volumes).

    Args:
        volume_data (np.ndarray): The 3D volume array (D, H, W).
        crop_D, crop_H, crop_W (int): Dimensions of the 3D crop.

    Returns:
        list: A list of 3D numpy arrays, where each array is a crop.
    """
    D, H, W = volume_data.shape
    crops = []
    
    # Calculate how many crops fit along each axis
    crops_d = D // crop_D
    crops_h = H // crop_H
    crops_w = W // crop_W

    print(f"Original volume size: {D}x{H}x{W}. Crop size: {crop_D}x{crop_H}x{crop_W}.")
    print(f"Total possible non-overlapping crops: {crops_d * crops_h * crops_w}")

    # Iterate over depth (Z), height (Y), and width (X)
    for k in range(crops_d):
        z_start = k * crop_D
        z_end = z_start + crop_D
        for i in range(crops_h):
            y_start = i * crop_H
            y_end = y_start + crop_H
            for j in range(crops_w):
                x_start = j * crop_W
                x_end = x_start + crop_W
                
                # Extract the 3D crop using NumPy slicing
                crop = volume_data[z_start:z_end, y_start:y_end, x_start:x_end]
                
                if np.sum(crop>0)/(crop_D*crop_H*crop_W)<0.5:
                    crops.append(crop)
            

    return crops

def get_N_crops(volume: np.ndarray, crop_size: int, N: int) -> list[np.ndarray]:
    """
    Get N random cubic crops (size x size x size) from a 3D volume, 
    ensuring they do NOT touch the domain boundaries.

    Args:
        volume (np.ndarray): The 3D volume (D, H, W).
        crop_size (int): The cubic size of the crop (C x C x C).
        N (int): The number of random crops to generate.

    Returns:
        list[np.ndarray]: A list containing N 3D crop arrays.
    """
    
    volume = volume.copy()
    # Get the dimensions of the input volume
    D, H, W = volume.shape
    
    # 1. Check if the crop size is possible
    if crop_size >= D - 1 or crop_size >= H - 1 or crop_size >= W - 1:
        print(f"Error: Crop size ({crop_size}) is too large for the volume dimensions {volume.shape} to maintain a 1-voxel margin.")
        # If the crop size is exactly D, H, or W, the range will be [1, 0), which is invalid.
        return []
    
    
    crops = []
    
    max_z_start = D - crop_size
    max_y_start = H - crop_size
    max_x_start = W - crop_size

    print(f"Volume Shape: {volume.shape}")
    print(f"Crop Size: {crop_size}")
    print(f"Valid Z Start Range: [1, {max_z_start - 1}]")
    print(f"Valid Y Start Range: [1, {max_y_start - 1}]")
    print(f"Valid X Start Range: [1, {max_x_start - 1}]")

    for i in range(N):
        # The starting index must be at least 1 (to avoid index 0).
        # The ending index (start + crop_size) must be at most D - 1 
        # (to avoid the last index D).
        
        # np.random.randint(1, max_start) generates numbers from 1 up to max_start - 1.
        z = np.random.randint(1, max_z_start)
        y = np.random.randint(1, max_y_start)
        x = np.random.randint(1, max_x_start)
        
        # Perform the slicing
        crop = volume[z:z + crop_size, y:y + crop_size, x:x + crop_size]
        crops.append(crop)
        
        print(f"Crop {i+1}: Start (Z, Y, X) = ({z}, {y}, {x}), End = ({z+crop_size}, {y+crop_size}, {x+crop_size})")
        
    return crops

# === CONFIGURATION ===

# Original rock parameters

# Berea Buff
#base_dir  = "./Test_Oliveira_BereaBuff_120_120_120/"
#rock_name = "BB_2d25um_binary.raw"

# Berea Upper Gray
#base_dir  = "./Test_Oliveira_BereaUpperGray_120_120_120/"
#rock_name = "BUG_2d25um_binary.raw"

# Berea Sinter Gray
#base_dir  = "./Test_Oliveira_BereaSinterGray_120_120_120/"
#rock_name = "BSG_2d25um_binary.raw"

# Berea
#base_dir  = "./Test_Oliveira_Berea_120_120_120/"
#rock_name = "Berea_2d25um_binary.raw"

# Castle Gate
#base_dir  = "./Test_Oliveira_CastleGate_120_120_120/"
#rock_name = "CastleGate_2d25um_binary.raw"

# Kirby
#base_dir  = "./Test_Oliveira_Kirby_120_120_120/"
#rock_name = "Kirby_2d25um_binary.raw"

# Bentheimer
base_dir  = "./Test_Oliveira_Bentheimer_120_120_120/"
rock_name = "Bentheimer_2d25um_binary.raw"

# Leopard
#base_dir  = "./Test_Oliveira_Leopard_120_120_120/"
#rock_name = "Leopard_2d25um_binary.raw"

# Parker
#base_dir  = "./Test_Oliveira_Parker_120_120_120/"
#rock_name = "Parker_2d25um_binary.raw"

# Bandera Brown (Not able to create: geometry out of scope for this resolution)
#base_dir  = "./Test_Oliveira_Brown_120_120_120/"
#rock_name = "BanderaBrown_2d25um_binary.raw"

# Bandera Gray (Not able to create: geometry out of scope for this resolution)
#base_dir  = "./Test_Oliveira_Bandera_120_120_120/"
#rock_name = "BanderaGray_2d25um_binary.raw"

rock_shape = (1000,1000,1000)
solid_value = 1

# Crops parameters
output_root = base_dir+"Samples/"
suffle      = True
create_n    = 30  # Set for 1 day (24h) of simulations= 120 samples (24x5) or 8h=40 samples
chunk_size  = 5   # Set for 1h of simulations (5 samples, 20 min per sample)
crop_shape = (120,120,120)

# Simulation parameters
gres        = "gpu:k40m" #"gpu:k40m"#"gpu:a100"
n_proc      = 1
cpu         = 2 
gpu         = 12



# === PROCESSING ===
os.makedirs(output_root, exist_ok=True)

vol         = np.fromfile(base_dir+rock_name, dtype=np.uint8)
vol         = vol.reshape(rock_shape)    # now a 3D numpy array
vol         = vol.astype(np.uint8)    # (x, y, z) 0/1


# Make it on LBPM convention (0-> Solid, 1-> Void) 
if solid_value==1:  vol         = 1 - vol
# Make crops
crops       = get_non_overlapping_crops(vol, crop_shape[0], crop_shape[1], crop_shape[2])

# Shuffle crops
indices = list(range(0, len(crops)))
if suffle: random.shuffle(indices)
created = 0
created_indices = []
for i, sample_i in enumerate(indices):
    if created >= create_n: break
    folder_base = f"Sample_{sample_i:05d}"
    x_sample    = crops[sample_i]  
    
    # Transform x_sample for simulation:
    x_sample[:, :, 0]   = 0
    x_sample[:, :, -1]  = 0
    x_sample[:, 0, :]   = 0
    x_sample[:, -1, :]  = 0
    
    # Check if the sample can percolate:
    if not utils.is_percolating(x_sample, axis=0):
        print(f"Sample {sample_i} do not percolate and got removed.")
    elif not utils.is_well_resolved(x_sample, min_pore_mean=3, max_pore_mean=5.5):
        print(f"Sample {sample_i} has geometry out of scope and got removed.")
    else:        
        utils.create_simulation_force_condition(x_sample,  output_root, folder_base, reflect=False, bc=5, n_proc=n_proc)
        created+=1
        created_indices.append(sample_i)
        print(f"Sample {sample_i} got included.")
        

utils.generate_slurm_run_scripts_chunks(created_indices,
                                  n_proc,
                                  gres,
                                  output_root,
                                  chunk_size,
                                  cpu, 
                                  gpu,
                                  f"Run_{indices[0]}_{indices[-1]}.sh")
    
print(f"--> All {len(created_indices)} samples exported successfully!")
                   
