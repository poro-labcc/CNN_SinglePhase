import os
import numpy as np
import utils

# === CONFIGURATION ===

#base_folder = "/home/gabriel/Desktop/Dissertacao/GradSimulations/Train_Danny_SphPore_120_120_120/"
#repositories = [
#    "Sample_00000",
#    "Sample_00001",
#    "Sample_00002",
#    "Sample_00003",
#    "Sample_00004",
#] 

#base_folder = "/home/gabriel/Desktop/Dissertacao/GradSimulations/Valid_Danny_SphPore_120_120_120/"
#repositories = [
#    "Sample_00005",
#    "Sample_00006",
#] 

base_folder = "/home/gabriel/Desktop/Dissertacao/GradSimulations/Test_Danny_SphPore_120_120_120/"
repositories = [
    "Sample_00007",
    "Sample_00008",
] 

rock_name = "mod_domain.raw"

# Parameters
include_walls       = True
remove_isolated     = False
target_percentage   = 70
crop_shape          = (120, 120, 120) # Define the shape of the incoming crops

# === EXECUTION === 
folder_paths = []
created_indices = []
created = 0

for sample_i, repo_name in enumerate(repositories):
    
    # --- PATH HANDLING ---
    # Safely combine the base folder and the current sample folder
    current_sample_dir = os.path.join(base_folder, repo_name)

    # 2. Trick the utils function into writing exactly in this directory
    output_root = current_sample_dir

    # === PROCESSING ===
    vol_path = os.path.join(current_sample_dir, rock_name)
    
    if not os.path.exists(vol_path):
        print(f"File not found: {vol_path}. Skipping.")
        continue
        
    # Load the crop and reshape it to the CROP dimensions
    vol = np.fromfile(vol_path, dtype=np.uint8)
    vol = vol.reshape(crop_shape)        
    vol = vol.astype(np.uint8)           
    
    print("----------------------")
    print(f"Sample: {current_sample_dir}")
    
    # Transform sample for simulation (Condensed logic)
    x_sample = utils.add_enclusure_walls(vol) if include_walls else vol
    filt_vol = utils.remove_isolated_pores(x_sample) if remove_isolated else x_sample
    
    # Check if the sample can percolate:
    if not utils.is_percolating(filt_vol, axis=0):
        print(f"Sample {sample_i} does not percolate and got removed.")
    elif not utils.check_local_thickness(filt_vol, min_radius=5, max_radius=17, target_percentage=target_percentage):
        print(f"Sample {sample_i} has geometry out of scope and got removed.")
    else:        
        # Passes current_sample_dir and an empty string ("") 
        # so os.path.join safely keeps the output in the same folder
        folder_path = utils.create_simulation_pressure_condition(
            filt_vol, 
            output_root, 
            "",
            n_proc=1, 
            include_walls=include_walls
        )
        
        folder_paths.append(folder_path)
        created += 1
        created_indices.append(sample_i)
        print(f"Sample {sample_i} got included.")


# Generate SLURM run scripts after the loop completes
if folder_paths:
    utils.generate_slurm_run_scripts_chunks_GRADLBM(
        folder_paths        = folder_paths,
        n_proc              = 1,
        output_root         = base_folder, 
        samples_per_job     = 20,
        partition           = "close_cpu",
        nodelist            = "node[008-020]",
        cpu_per_sim         = 1, 
        mem_gb_per_sim      = 6,
        dispatcher_name     = f"Run_GRAD_0_{created}.sh",
        lbm_folder          = "/home/gabriel.silveira/GRAD_LBM/",
        ini_name            = "grad.ini",
        chain_launchers     = False,
    )