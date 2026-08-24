import numpy as np
import porespy as ps
import os
import scipy.stats as sps # Import for statistical distributions
import utils


def make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=0):
    
    MIN_RADIUS          = min(MEAN_RADIUS/6,4)
    # Standard deviation estimation
    StdDev              = MEAN_RADIUS/3 
    # Define the normal distribution object (Mean=5, StdDev=3)
    radius_distribution = sps.norm(loc=MEAN_RADIUS, scale=StdDev)
    
    # Call the function using the specific signature you requested
    vol = ps.generators.polydisperse_spheres(
        shape   =SHAPE,
        porosity=1-SPHERES_FILL,
        dist    =radius_distribution,   # Pass the statistical distribution object
        r_min   =MIN_RADIUS,            # Ensure the smallest generated sphere is at least 1 voxel
        seed    =seed                   # for reproducibility
    )
    # Sphere are void
    if not solid_spheres: vol = 1-vol
    
    return vol
    



# --- Simulation Parameters ---
chunk_size      = 10   # Set for 1h of simulations (5 samples, 20 min per sample)
gres            = "gpu:a100" #"gpu:k40m"#"gpu:a100"
partition       = "all_gpu"
#gres            = None #"gpu:k40m"#"gpu:a100"
#partition       = "close_cpu"
n_proc          = 4  
cpu             = 12 
gpu             = 64 
use_low_prio        = False
include_allocation  = False 
lbpm_version        = "lbpm/gpu/lbpm_fork_965bd0d"

# --- Domains Parameters ---
DIM             = 120
SHAPE           = [DIM, DIM, DIM] # Shape must be a List for the function signature you provided
AXIS_OF_FLOW    = 0 
N_SAMPLES       = 1 # 3 for training, 2 for testing, 1 for validation (192/128/64)

include_walls   = True
remove_isolated = False

output_root = "../../Simulations/Valid_SphGrain_120_120_120"
seed = 1 # 10 for training, 2 for testing, 1 for validation 

##########################
# CREATE SPHERICAL PORES #
##########################


# Spheres fill, radii
config_pairs = [
    (0.8, 20),
    (0.8, 22),
    (0.8, 24),
    (0.8, 26),
     
    (0.7, 18),
    (0.7, 20),
    (0.7, 22),
    (0.7, 24),
    
    (0.6, 16),
    (0.6, 18),
    (0.6, 20),
    (0.6, 22),
    
    (0.5, 14),
    (0.5, 16),
    (0.5, 18),
    (0.5, 20),
]

os.makedirs(output_root, exist_ok=True)

volumes             = []
solid_spheres       = True
total_created = 0
for SPHERES_FILL, MEAN_RADIUS in config_pairs:    # Porosities large enough so that spheres touch         
        created = 0
        for n in range(N_SAMPLES*50):
            if created >= N_SAMPLES: break
            print(f"Attempt to create sample {total_created}")
            print(f"-->Filling {SPHERES_FILL*100}% with Sphere, Mean Radius {MEAN_RADIUS} ({n})")
            # Create volumes
            seed_v = int(n*1000+MEAN_RADIUS*100+SPHERES_FILL*10)+seed
            vol = make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=seed_v).astype(np.uint8)
            
            # Transform sample for simulation:
            if include_walls: vol = utils.add_enclusure_walls(vol)
                
            if remove_isolated: filt_vol = utils.remove_isolated_pores(vol)
            else: filt_vol = vol
            
            # Check porosity
            actual_porosity = np.sum(vol) / vol.size
            print(f"-->Actual Porosity: {actual_porosity*100:.2f}%")
            
            # Sanity checks
            if not utils.is_percolating(vol, axis=0):
                print(f"-->Sample do not percolate and got removed.")
            elif not utils.check_local_thickness(vol, min_radius=5, max_radius=17, target_percentage=70):
                print(f"-->Sample has geometry out of scope and got removed.")
            else:        
                print(f"-->Sample {total_created} got included.")
                folder_base = f"Sample_{total_created:05d}"
                utils.create_simulation_pressure_condition(vol,  output_root, folder_base,  n_proc=n_proc, include_walls=include_walls)
                total_created +=1
                created+=1
            print("-" * 30)

        


utils.generate_slurm_run_scripts_chunks(sample_indices  = list(range(0, total_created + 1)),
                                        n_proc          = n_proc,
                                        gres            = gres,
                                        output_root     = output_root,
                                        samples_per_job = chunk_size,
                                        cpu             = cpu, 
                                        gpu             = gpu,
                                        partition       = partition,                        
                                        dispatcher_name = f"Run_{0}_{total_created}.sh",
                                        lbpm_version    = lbpm_version,
                                        use_low_prio        = use_low_prio,
                                        include_allocation  = include_allocation     
                                        )
