import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from Architectures.Unet   import Extended_DannyKo, MY_PIMODEL
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu
from Utilities import dataset_reader as dr

from scipy.ndimage import distance_transform_edt as edt

# ==============================================================================
# SETUP
# ============================================================================== 

shape       = (120, 120, 120)
device      = 'cpu'
raw_file    = "domain.raw"

# Define the dataset and the specific sample to extract
datapath    = "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_PressureWalls.h5",
sample_idx  = 0
output_base = f"./StartTest/Dataset_Sample{sample_idx}/"

# Ensure base output directory exists
os.makedirs(output_base, exist_ok=True)

# Load Models
models = {}

print("Loading Models...")
danny_model         = Extended_DannyKo()
danny_z_name        = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
danny_y_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-14PM_Job16195/model_LowerValidationLoss.pth"
danny_x_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
danny_p_name        = "./Trained_Models/NN_Trainning_24_March_2026_03-59PM_Job16921/model_LowerValidationLoss.pth"
danny_sub_comp      = SubModels_Composition(danny_model, 
                                    danny_z_name, danny_y_name, danny_x_name, danny_p_name, 
                                    device=device, bin_input=True, is_eval=True)
models["Danny_subModels"] = danny_sub_comp

javier_model         = JavierSantos_Extended()
javier_z_name        = "./Trained_Models/NN_Trainning_14_March_2026_10-52PM_Job16201/model_LowerValidationLoss.pth"
javier_y_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-17PM_Job17461/model_LowerValidationLoss.pth"
javier_x_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-15PM_Job17460/model_LowerValidationLoss.pth"
javier_p_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-18PM_Job17462/model_LowerValidationLoss.pth"
javier_sub_comp      = SubModels_Composition(javier_model, 
                                    javier_z_name, javier_y_name, javier_x_name, javier_p_name, 
                                    device=device, bin_input=False, is_eval=True)
models["Javier_subModels"] = javier_sub_comp  

danny_f_model       = Extended_DannyKo()
danny_f_name        = "./Trained_Models/NN_Trainning_10_April_2026_01-25PM/model_LowerValidationLoss.pth"
danny_f_model.load_state_dict(torch.load(danny_f_name, map_location=torch.device(device), weights_only=True))
danny_f_model.z_model.load_state_dict(torch.load(danny_z_name, map_location=torch.device(device), weights_only=True))
danny_f_model.y_model.load_state_dict(torch.load(danny_y_name, map_location=torch.device(device), weights_only=True))
danny_f_model.x_model.load_state_dict(torch.load(danny_x_name, map_location=torch.device(device), weights_only=True))
danny_f_model.p_model.load_state_dict(torch.load(danny_p_name, map_location=torch.device(device), weights_only=True))
danny_f_model.eval()
models["Danny_Final"] = danny_f_model

pinn_model          = MY_PIMODEL()
pinn_name           = "./Trained_Models/NN_Trainning_11_April_2026_01-39PM/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.z_model.load_state_dict(torch.load(danny_z_name, map_location=torch.device(device), weights_only=True))
pinn_model.y_model.load_state_dict(torch.load(danny_y_name, map_location=torch.device(device), weights_only=True))
pinn_model.x_model.load_state_dict(torch.load(danny_x_name, map_location=torch.device(device), weights_only=True))
pinn_model.p_model.load_state_dict(torch.load(danny_p_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["My_Model"] = pinn_model

# ==============================================================================
# MAIN
# ============================================================================== 

print("\nLoading data from dataset...")
dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32)

# Extract specific sample and add Batch dimension
sample_input, sample_target = dataset[sample_idx]
x_base = sample_input.unsqueeze(0).to(device)
y_true = sample_target.unsqueeze(0).to(device)

# Denormalize target for fair comparison (if your targets are normalized in the dataset)
y_true = vu.tensor_denorm(out=y_true, inp=x_base)

# Extract Geometry (Assuming > 0 is Fluid based on our previous fix)
geometry = (x_base[0, 0].cpu().numpy() > 0)

# Extract Target fields
tz = y_true[0, 0].cpu().numpy()
ty = y_true[0, 1].cpu().numpy()
tx = y_true[0, 2].cpu().numpy()
tp = y_true[0, 3].cpu().numpy()

# Calculate Target Permeability
target_perm = vu.permeability_calculation(y_true, x_base, denorm=False)

print("="*80)
print(f"--- TARGET (GROUND TRUTH) STATISTICS ---")
print(f"   -> Permeability: {target_perm:.6e}")
print(f"   -> Uz - max: {tz.max():>12.6e} | mean: {tz.mean():>12.6e} | min: {tz.min():>12.6e}")
print(f"   -> Uy - max: {ty.max():>12.6e} | mean: {ty.mean():>12.6e} | min: {ty.min():>12.6e}")
print(f"   -> Ux - max: {tx.max():>12.6e} | mean: {tx.mean():>12.6e} | min: {tx.min():>12.6e}")
print(f"   -> Pr - max: {tp.max():>12.6e} | mean: {tp.mean():>12.6e} | min: {tp.min():>12.6e}")
print("="*80, "\n")


for model_name, model in models.items():
    
    out_dir = os.path.join(output_base, model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert NN input format based on model requirements
    if model.bin_input: x = geometry.astype("float32")
    else:               x = edt(geometry).astype("float32")
    
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    
    # Make prediction
    print(f"Running prediction for: {model_name}...")
    with torch.no_grad():
        pred = model.predict(x)
        
    # Denormalize predictions
    pred = vu.tensor_denorm(out=pred, inp=x)
    
    # Extract Predicted fields
    uz = pred[0, 0].cpu().numpy()
    uy = pred[0, 1].cpu().numpy()
    ux = pred[0, 2].cpu().numpy()
    pr = pred[0, 3].cpu().numpy()
            
    # ==========================================
    # Sanity Checks
    # ==========================================
    if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape): 
        raise Exception("Prediction doesn't match specified shape.")
        
    if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
        raise ValueError(f"CRITICAL: Model {model_name} predicted NaN or Inf values!")
        
    # Solid Matching (No-Slip Condition in the Rock/Solid phase)
    solid_vel_mag = np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)
    if np.any(solid_vel_mag > 1e-6):
        print(f"   [!] WARNING: {model_name} predicted velocity inside solid! Forcing to 0.0.")
        ux[~geometry] = 0.0
        uy[~geometry] = 0.0
        uz[~geometry] = 0.0
        
    if np.max(np.abs(uz)) == 0.0 and np.max(np.abs(uy)) == 0.0 and np.max(np.abs(ux)) == 0.0:
        print(f"   [!] WARNING: {model_name} predicted a completely ZERO velocity field.")

    max_v = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
    if max_v > 0.7:
        print(f"   [!] DANGER: {model_name} predicted a max velocity of {max_v:.4f}. LBPM may be unstable (Mach limit).")
        
    # ==========================================
    # File Writing
    # ==========================================
    start_path = os.path.join(out_dir, "Start.00000")
    db_path    = os.path.join(out_dir, "start_pressure.db")
    raw_path   = os.path.join(out_dir, raw_file)

    sh.write_start_raw(filename=start_path, ux=ux, uy=uy, uz=uz, pr=pr)
    
    tau, Re, Dens = 1.5, 0.1, 1.0
    p_drop = vu.pressure_calculation(geometry, tau=tau, Re=Re, Dens=Dens)
    
    sh.write_lbpm_db(
        db_name = db_path, path = "", tau = tau, bc = 3,
        din = 1.0+3*p_drop, dout = 1.0, nproc = (1, 1, 1),
        n = shape, N = shape, analysis_interval = 1000,
        tolerance = 1e-6, out_format = "silo", Start = True  
    )
    
    # Rewrite Geometry .raw
    geometry.astype(np.uint8).tofile(raw_path)
    
    # ==========================================
    # Stats Output
    # ==========================================
    pred_perm = vu.permeability_calculation(pred, x, denorm=False)
    
    print(f"--- PREDICTED STATISTICS ({model_name}) ---")
    print(f"   -> Permeability: {pred_perm:.6e}")
    print(f"   -> Uz - max: {uz.max():>12.6e} | mean: {uz.mean():>12.6e} | min: {uz.min():>12.6e}")
    print(f"   -> Uy - max: {uy.max():>12.6e} | mean: {uy.mean():>12.6e} | min: {uy.min():>12.6e}")
    print(f"   -> Ux - max: {ux.max():>12.6e} | mean: {ux.mean():>12.6e} | min: {ux.min():>12.6e}")
    print(f"   -> Pr - max: {pr.max():>12.6e} | mean: {pr.mean():>12.6e} | min: {pr.min():>12.6e}")
    print("-"*80, "\n")