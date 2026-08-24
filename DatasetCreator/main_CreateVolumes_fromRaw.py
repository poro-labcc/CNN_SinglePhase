import numpy as np
import utils 
import glob
import os

# ==============================================================================
# MAIN: CREATE .db
# ==============================================================================

include_walls   = True
remove_isolated = True
shape           = (120, 120, 120)
input_name      = "domain.raw"
output_name     = "mod_domain.raw"

#paths           = [
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00000/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00001/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00002/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00003/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00004/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00005/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00006/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00007/",
#    "../../Simulations/Train_Danny_SphPore_120_120_120/Sample_00008/",
#]


folder_path = "/home/gabriel/remote/hal/dissertacao/Simulations/Train_Danny_SphPore_120_120_120/Validation/"
paths       = glob.glob(os.path.join(folder_path, "**", input_name), recursive=True)

# Local Thickness interval check
min_radius          = 5
max_radius          = 17
target_percentage   = 70


for path in paths:
    
    path        = os.path.dirname(path)
    
    print(f"Creating geometry {path}")
    vol         = np.fromfile(path+"/"+input_name, dtype=np.uint8)
    vol         = vol.reshape(shape)
    
    # Transform sample for simulation:
    if include_walls: vol = utils.add_enclusure_walls(vol)
        
    if remove_isolated: vol = utils.remove_isolated_pores(vol)
    
    # Check porosity
    actual_porosity = np.sum(vol) / vol.size
    print(f"-->Actual Porosity: {actual_porosity*100:.2f}%")
    
    # Sanity checks
    if not utils.is_percolating(vol, axis=0):
        print("-->Sample do not percolate and got removed.")
    elif not utils.check_local_thickness(vol, min_radius=5, max_radius=17, target_percentage=70):
        print("-->Sample has geometry out of scope and got removed.")
    else:      
        print("-->Sample got included.")

    utils.write_domain_raw(path, vol, filename = output_name)
        