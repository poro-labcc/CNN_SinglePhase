import os
import glob
import numpy as np
import utils
from pathlib import Path

BASE_DIRECTORIES = [
    "/home/gabriel/remote/hal/dissertacao/Simulations/Train_Danny_SphPore_120_120_120/",
    "/home/gabriel/remote/hal/dissertacao/Simulations/Test_Danny_SphPore_120_120_120/",
    "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_Danny_SphPore_120_120_120/",
    ]

# --- Hardware Parameters ---
chunk_size          = 10   # Set for 1h of simulations (5 samples, 20 min per sample)
#gres                = "gpu:a100" #"gpu:k40m"#"gpu:a100"
#partition           = "all_gpu"
gres                = None #"gpu:k40m"#"gpu:a100"
partition           = "close_cpu"
n_proc              = 4
cpu                 = 12 
gpu                 = 64
use_low_prio        = False
include_allocation  = False 
lbpm_version        = "lbpm/cpu/lbpm_fork_2016010"

# --- Domains Parameters ---
RAW_FILENAME    = "mod_domain.raw"
VOL_SHAPE       = (120, 120, 120)
VOL_DTYPE       = np.uint8

# --- Simulation Parameters
Re   = 0.1
tau  = 1.5
Dens = 1.0


for BASE_DIR in BASE_DIRECTORIES:
    print("Creating runners for ", BASE_DIR)
    
    raw_files = utils.find_raw_in_folder(BASE_DIR, RAW_FILENAME)

    # Create .db based on geometry
    for file_name in raw_files:
        vol = np.fromfile(file_name, dtype=np.uint8).reshape(VOL_SHAPE).astype(np.uint8)

        dP = utils.pressure_calculation(           
                vol,
                tau     = tau,
                Re      = Re,
                Dens    = Dens
            )
        
        timestep_max = utils.timestep_calculation(    
                matriz_binaria  =vol,
                tau             =tau,
                Re              =Re,
                Dens            =Dens,
                safety_factor   =10.0
                )
        
        # --- Save 3D domain as .raw ---
        folder_path = os.path.dirname(file_name)
        
        
        utils.write_lbpm_db(
                path      = folder_path, 
                tau       = tau,
                bc        = 3,
                din       = 1.0+dP*3,
                dout      = 1.0,
                nproc     = (1, 1, n_proc),
                n         = (vol.shape[2], vol.shape[1], int(vol.shape[0]/n_proc)),
                N         = (vol.shape[2], vol.shape[1], vol.shape[0]),
                tolerance = 1e-4,
                domain_filename           = RAW_FILENAME,
                analysis_interval         = 1000, 
                visualization_interval    =timestep_max, 
                timestep_max              =timestep_max,
                subphase_analysis_interval=timestep_max,
                restart_interval          =timestep_max
                )
        
        
    # Create .sh based on number of files
    total_created = len(raw_files)
    utils.generate_slurm_run_scripts_chunks(
        sample_indices  = list(range(0, total_created + 1)),
        n_proc          = n_proc,      
        gres            = gres,       
        output_root     = BASE_DIR,   
        samples_per_job = chunk_size, 
        cpu             = cpu,         
        gpu             = gpu,
        partition       = partition,                        
        dispatcher_name = f"Run_{0}_{total_created}.sh",
        lbpm_version    = lbpm_version,
        use_low_prio        = use_low_prio,
        include_allocation  = include_allocation       
        )