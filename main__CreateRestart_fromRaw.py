import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import matplotlib.pyplot as plt

from Architectures.Unet   import Extended_DannyKo, MY_PIMODEL
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu



paths = [
    #"./StartTest/Train_Sample/",
    "./StartTest/Bentheimer_Sample/",
    
    ]

raw_file    = "domain.raw"
shape       = (120,120,120)
rewrite_raw = True



models = {}
danny_model         = Extended_DannyKo()
danny_z_name        = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
danny_y_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-14PM_Job16195/model_LowerValidationLoss.pth"
danny_x_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
danny_p_name        = "./Trained_Models/NN_Trainning_24_March_2026_03-59PM_Job16921/model_LowerValidationLoss.pth"
danny_sub_comp      = SubModels_Composition(danny_model, 
                                    danny_z_name, 
                                    danny_y_name, 
                                    danny_x_name, 
                                    danny_p_name, 
                                    device='cpu', 
                                    bin_input=True, 
                                    is_eval=True)
models["Danny_subModels"]     = danny_sub_comp

javier_model         = JavierSantos_Extended()
javier_z_name        = "./Trained_Models/NN_Trainning_14_March_2026_10-52PM_Job16201/model_LowerValidationLoss.pth"
javier_y_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-17PM_Job17461/model_LowerValidationLoss.pth"
javier_x_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-15PM_Job17460/model_LowerValidationLoss.pth"
javier_p_name        = "./Trained_Models/NN_Trainning_2_April_2026_06-18PM_Job17462/model_LowerValidationLoss.pth"
javier_sub_comp      = SubModels_Composition(javier_model, 
                                    javier_z_name, 
                                    javier_y_name, 
                                    javier_x_name, 
                                    javier_p_name, 
                                    device='cpu', 
                                    bin_input=False, 
                                    is_eval=True)
models["Javier_subModels"]     = javier_sub_comp  


danny_f_model       = Extended_DannyKo()
danny_f_name        = "./Trained_Models/NN_Trainning_10_April_2026_01-25PM/model_LowerValidationLoss.pth"
danny_f_model.load_state_dict(torch.load(danny_f_name, map_location=torch.device('cpu'), weights_only=True))
danny_f_model.eval()
danny_z_name        = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
danny_y_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-14PM_Job16195/model_LowerValidationLoss.pth"
danny_x_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
danny_p_name        = "./Trained_Models/NN_Trainning_24_March_2026_03-59PM_Job16921/model_LowerValidationLoss.pth"
danny_f_model.z_model.load_state_dict(torch.load(danny_z_name, map_location=torch.device('cpu'), weights_only=True))
danny_f_model.y_model.load_state_dict(torch.load(danny_y_name, map_location=torch.device('cpu'), weights_only=True))
danny_f_model.x_model.load_state_dict(torch.load(danny_x_name, map_location=torch.device('cpu'), weights_only=True))
danny_f_model.p_model.load_state_dict(torch.load(danny_p_name, map_location=torch.device('cpu'), weights_only=True))
danny_f_model.z_model.eval()
danny_f_model.y_model.eval()
danny_f_model.x_model.eval()
danny_f_model.p_model.eval()
models["Danny_Final"]= danny_f_model


pinn_model          = MY_PIMODEL()
pinn_name           = "./Trained_Models/NN_Trainning_11_April_2026_01-39PM/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device('cpu'), weights_only=True))
pinn_model.eval()
danny_z_name        = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
danny_y_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-14PM_Job16195/model_LowerValidationLoss.pth"
danny_x_name        = "./Trained_Models/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
danny_p_name        = "./Trained_Models/NN_Trainning_24_March_2026_03-59PM_Job16921/model_LowerValidationLoss.pth"
pinn_model.z_model.load_state_dict(torch.load(danny_z_name, map_location=torch.device('cpu'), weights_only=True))
pinn_model.y_model.load_state_dict(torch.load(danny_y_name, map_location=torch.device('cpu'), weights_only=True))
pinn_model.x_model.load_state_dict(torch.load(danny_x_name, map_location=torch.device('cpu'), weights_only=True))
pinn_model.p_model.load_state_dict(torch.load(danny_p_name, map_location=torch.device('cpu'), weights_only=True))
pinn_model.z_model.eval()
pinn_model.y_model.eval()
pinn_model.x_model.eval()
pinn_model.p_model.eval()
models["My_Model"] = pinn_model

# ==============================================================================
# MAIN
# ============================================================================== 

for path in paths:
    
    for model_name, model in models.items():
        
        
        geometry     = (np.fromfile(path+raw_file, dtype=np.uint8).reshape(shape)>0)
        geometry_edt = edt(geometry).astype("float32")
        
        
        
        # Convert numpy array (Z,Y,X) to tensor (B=1,C=1, Z,Y,X)
        geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        print(f"Creating prediction with {model_name}")
        pred    = model.predict(geometry_edt)
        uz      = pred[0,0].numpy()
        uy      = pred[0,1].numpy()
        ux      = pred[0,2].numpy()
        pr      = pred[0,3].numpy()
        
        # Denormalize predictions
        pred    = vu.tensor_denorm(out=pred, inp=geometry_edt)
        
        # Prepare data for start file
        uz      = pred[0,0].numpy()
        uy      = pred[0,1].numpy()
        ux      = pred[0,2].numpy()
        pr      = pred[0,3].numpy()
                
        # Sanity Checks
        #  - Shape Matching
        if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape): 
            raise Exception("Prediction dont match specified .raw shape.")
        #  - NaN and Inf presence check
        if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
            raise ValueError(f"Model {model_name} predicted NaN or Inf values!")
        #  - Solid Matching (No-Slip Condition)
        solid_vel_mag =  np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)
        if np.any(solid_vel_mag > 1e-6):
            print(f"   [!] WARNING: {model_name} predicted velocity inside solid! Forcing to 0.0.")
            ux[~geometry] = 0.0
            uy[~geometry] = 0.0
            uz[~geometry] = 0.0
            
        #  - Zero-Field Check
        if np.max(np.abs(uz)) == 0.0 and np.max(np.abs(uy)) == 0.0 and np.max(np.abs(ux)) == 0.0:
            print(f"   [!] WARNING: {model_name} predicted a completely ZERO velocity field.")

        #  - LBM Stability Check (Max Velocity)
        max_v   = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
        if max_v > 0.7:
            print(f"   [!] DANGER: {model_name} predicted a max velocity of {max_v:.4f}. LBPM may be unstable (Mach limit).")
            
        # Write start file 
        print(f"   -> Creating Start.00000 file")
        sh.write_start_raw(
            filename = path+model_name+"/Start.00000",
            ux=ux, uy=uy, uz=uz, pr=pr
        )
        
        # Write the .db
        print(f"   -> Creating .db file")
        tau         = 1.5
        Re          = 0.1
        Dens        = 1.0
        p_drop       = vu.pressure_calculation(geometry, tau=tau, Re=Re, Dens=Dens)
        sh.write_lbpm_db(
            db_name = path+model_name+"/start_pressure.db",
            path    = "",
            tau     = tau,
            bc      = 3,
            din     = 1.0+3*p_drop,
            dout    = 1.0,
            nproc   = (1, 1, 1),
            n       = shape,
            N       = shape,
            analysis_interval = 1000,
            tolerance         = 1e-6,
            out_format        = "silo",
            Start             = True  
        )
        
        # Rewrite .raw
        geometry.astype(np.uint8).tofile(path+model_name+"/"+raw_file)
        
        # Use prediction and show primary statistics
        pred_perm = vu.permeability_calculation(pred, geometry_edt, denorm=False)
        perm_val = float(pred_perm)
        print(f"   -> Perm | {perm_val:.6e}")
        print(f"   -> Uz   | max: {uz.max():>13.6e} | mean: {uz.mean():>13.6e} | min: {uz.min():>13.6e}")
        print(f"   -> Uy   | max: {uy.max():>13.6e} | mean: {uy.mean():>13.6e} | min: {uy.min():>13.6e}")
        print(f"   -> Ux   | max: {ux.max():>13.6e} | mean: {ux.mean():>13.6e} | min: {ux.min():>13.6e}")
        print(f"   -> Pr   | max: {pr.max():>13.6e} | mean: {pr.mean():>13.6e} | min: {pr.min():>13.6e}")
        print()
        
        
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        im0 = axes[0, 0].imshow(uz[:, :, 60], cmap='plasma')
        axes[0, 0].set_title('Uz Velocity')
        fig.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(uy[:, :, 60], cmap='plasma')
        axes[0, 1].set_title('Uy Velocity')
        fig.colorbar(im1, ax=axes[0, 1])
        im2 = axes[1, 0].imshow(ux[:, :, 60], cmap='plasma')
        axes[1, 0].set_title('Ux Velocity')
        fig.colorbar(im2, ax=axes[1, 0])
        im3 = axes[1, 1].imshow(pr[:, :, 60], cmap='plasma')
        axes[1, 1].set_title('Pressure (Pr)')
        fig.colorbar(im3, ax=axes[1, 1])
        plt.tight_layout()
        plt.show()
        
        