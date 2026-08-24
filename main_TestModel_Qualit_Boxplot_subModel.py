import numpy              as np
import torch
import tensorflow         as tf
import matplotlib.pyplot  as plt
import pandas             as pd
import seaborn            as sns

from torch.utils.data     import DataLoader

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended

from Utilities            import dataset_reader as dr
from Danny_Original.architecture import Danny_KerasModel


#######################################################
#************ UTILS:                      ***********#
#######################################################

def mean_normalize(inp, x): 
    B, C, Z, Y, Xdim = x.shape
    mag     = torch.linalg.vector_norm(x, dim=1)  
    mask    = (inp > 0)  
    mask    = mask[:, 0] 

    means = []
    for b in range(B):
        vals    = mag[b][mask[b]]
        m       = vals.mean()
        means.append(m.unsqueeze(0))

    means = torch.stack(means, dim=0).view(B, 1, 1, 1, 1)

    return x / (means + 1e-12)


    
def print_n_params(model, pytorch=True):
    if pytorch:
        trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable   = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    else:
        trainable       = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable   = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    print("Trainable params:      ", trainable)
    print("Non-trainable params: ", non_trainable)
    print("Total params:          ", trainable + non_trainable)


#######################################################
#************ COMPARISONS:                 ***********#
####################################################### 

def Flux_Comparison(batch_inputs, batch_outputs, batch_targets):
    B = batch_inputs.shape[0]
    efs = []
    print("Flux Error:")
    for s_i in range(B):
        
        porous_mask = (batch_inputs[s_i] != 0)
        
        vel_z_true = batch_targets[s_i, 0]   # (D, H, W)
        vel_z_pred = batch_outputs[s_i, 0]
        
        vel_z_pred = vel_z_pred * porous_mask
        vel_z_pred = vel_z_pred * porous_mask
       
        q_z_true = vel_z_true.sum(axis=(1, 2))   # shape (D,)  soma sobre (H,W)
        q_z_pred = vel_z_pred.sum(axis=(1, 2))
        
        denom   = q_z_true.abs().sum() 

        e_fz    = (q_z_true - q_z_pred).abs().sum() / denom
        
        if batch_inputs.shape[1] >1:
            vel_y_true = batch_targets[s_i, 1]
            vel_x_true = batch_targets[s_i, 2]
    
            vel_y_pred = batch_outputs[s_i, 1]
            vel_x_pred = batch_outputs[s_i, 2]
            
            vel_x_true = vel_x_true * porous_mask
            vel_y_true = vel_y_true * porous_mask
            
            vel_x_pred = vel_x_pred * porous_mask
            vel_y_pred = vel_y_pred * porous_mask
            
            q_x_true = vel_x_true.sum(axis=(0, 1))   # shape (W,)  soma sobre (D,H)
            q_y_true = vel_y_true.sum(axis=(0, 2))   # shape (H,)  soma sobre (D,W)
            
            q_x_pred = vel_x_pred.sum(axis=(0, 1))
            q_y_pred = vel_y_pred.sum(axis=(0, 2))
            
            e_fx    = (q_x_true - q_x_pred).abs().sum() / denom
            e_fy    = (q_y_true - q_y_pred).abs().sum() / denom
            
        else:
            e_fx = 0.0
            e_fy = 0.0
            
        e_f = (e_fx + e_fy + e_fz)
        print(" -- Sample {}: {:.4f}".format(s_i, e_f.item()))
        efs.append(e_f.item())
    print(f"Mean: {np.mean(efs):.4f}, Std PE: {np.std(efs):.4f}")
    print("------------------------------------------------------------------")
    return efs

def Permeability_Comparison(batch_inputs, batch_outputs, batch_targets):

    vz_true     = batch_targets[:, 0:1, :, :, :]
    vz_pred     = batch_outputs[:, 0:1, :, :, :]
    print("Permeability Error (z):")
    pe_means = []
    for s_i in range(vz_pred.shape[0]):
        # Sample's void space
        s_i_mask    = batch_inputs[s_i:s_i+1]>=0
        # Sample's permeability
        vz_true_i   = vz_true[s_i:s_i+1][s_i_mask].mean()
        #print("!!!!!!!!!!!: ", vz_true_i)
        vz_pred_i   = vz_pred[s_i:s_i+1][s_i_mask].mean()
        pe_i        = ( 100*(vz_pred_i-vz_true_i).abs() / vz_true_i.abs() ).item()
        print(" -- Sample {}: {:.4f}".format(s_i, pe_i))
        pe_means.append(pe_i)
    print(f"Mean: {np.mean(pe_means):.4f}, Std PE: {np.std(pe_means):.4f}")
    print("------------------------------------------------------------------")
    return pe_means
    

def Magnitude_Deviation_Comparison(batch_inputs, batch_outputs, batch_targets):
    
    if batch_targets.shape[1] > 1:
    
        vz_true     = batch_targets[:, 0:1, :, :,:]
        vy_true     = batch_targets[:, 1:2, :, :,:]
        vx_true     = batch_targets[:, 2:3, :, :,:]
        
        vz_pred     = batch_outputs[:, 0:1, :, :,:]
        vy_pred     = batch_outputs[:, 1:2, :, :,:]
        vx_pred     = batch_outputs[:, 2:3, :, :,:]
        
        mag_true    = (vx_true**2 + vy_true**2 + vz_true**2).sqrt()
        mag_pred    = (vx_pred**2 + vy_pred**2 + vz_pred**2).sqrt()
        
    else:
        mag_true     = batch_targets[:, 0:1, :, :, :].abs()
        mag_pred     = batch_outputs[:, 0:1, :, :, :].abs()
        
    mag_mape     = ((mag_true-mag_pred)/mag_true).abs() * 100
    print("Magnitude Error Deviation: ")
    mag_mape_stds = []
    for s_i in range(mag_mape.shape[0]):
        
        mag_mape_i      = mag_mape[s_i:s_i+1]
        s_i_mask        = (batch_inputs[s_i:s_i+1]!=0)
        
        above_mean_mask = (mag_true[s_i:s_i+1] > mag_true[s_i:s_i+1][s_i_mask].mean())
        final_mask      = s_i_mask & above_mean_mask
        
        mean            = mag_mape_i[final_mask].mean()
        std             = mag_mape_i[final_mask].std()
        
        print(" -- Sample {}: mean={:.4f}%; std={:.4f}".format(s_i, mean.item(), std.item()))
        mag_mape_stds.append(std.item())
    print(f"Mean: {np.mean(mag_mape_stds):.4f}%, Std: {np.std(mag_mape_stds):.4f}%")
    print("------------------------------------------------------------------")
    return mag_mape_stds
    
def Magnitude_Comparison(batch_inputs, batch_outputs, batch_targets):
    
    if batch_targets.shape[1] > 1:
    
        vz_true     = batch_targets[:, 0:1, :, :,:]
        vy_true     = batch_targets[:, 1:2, :, :,:]
        vx_true     = batch_targets[:, 2:3, :, :,:]
        
        vz_pred     = batch_outputs[:, 0:1, :, :,:]
        vy_pred     = batch_outputs[:, 1:2, :, :,:]
        vx_pred     = batch_outputs[:, 2:3, :, :,:]
        
        mag_true    = (vx_true**2 + vy_true**2 + vz_true**2).sqrt()
        mag_pred    = (vx_pred**2 + vy_pred**2 + vz_pred**2).sqrt()
        
    else:
        mag_true     = batch_targets[:, 0:1, :, :, :].abs()
        mag_pred     = batch_outputs[:, 0:1, :, :, :].abs()
        
    mag_mape     = ((mag_true-mag_pred)/mag_true).abs() * 100
    print("Magnitude Error: ")
    mag_mape_means = []
    for s_i in range(mag_mape.shape[0]):
        
        mag_mape_i      = mag_mape[s_i:s_i+1]
        s_i_mask        = (batch_inputs[s_i:s_i+1]!=0)
        
        above_mean_mask = (mag_true[s_i:s_i+1] > mag_true[s_i:s_i+1][s_i_mask].mean())
        final_mask      = s_i_mask & above_mean_mask
        
        mean            = mag_mape_i[final_mask].mean()
        std             = mag_mape_i[final_mask].std()
        
        print(" -- Sample {}: mean={:.4f}%; std={:.4f}".format(s_i, mean.item(), std.item()))
        mag_mape_means.append(mean.item())
    print(f"Mean: {np.mean(mag_mape_means):.4f}%, Std: {np.std(mag_mape_means):.4f}%")
    print("------------------------------------------------------------------")
    return mag_mape_means
    

    
def Components_Comparison(batch_inputs, batch_outputs, batch_targets):
    fluid_mask  = (batch_inputs!=0).flatten()
    
    vz_true     = batch_targets[:, 0]
    vy_true     = batch_targets[:, 1]
    vx_true     = batch_targets[:, 2]
    
    vz_pred     = batch_outputs[:, 0]
    vy_pred     = batch_outputs[:, 1]
    vx_pred     = batch_outputs[:, 2]
    
    vx_error    = (vx_true - vx_pred).flatten()
    vy_error    = (vy_true - vy_pred).flatten()
    vz_error    = (vz_true - vz_pred).flatten()
    
    mag_error   = np.sqrt((vx_error)**2 + (vy_error)**2 + (vz_error)**2)
    
    mag_error   = mag_error[fluid_mask]
    vx_error    = vx_error[fluid_mask]
    vy_error    = vy_error[fluid_mask]
    vz_error    = vz_error[fluid_mask]
    
    vx_relerror = (vx_error**2/mag_error**2)*100
    vy_relerror = (vy_error**2/mag_error**2)*100
    vz_relerror = (vz_error**2/mag_error**2)*100
    
    print(f"Percentual Vel_X Error: mean={vx_relerror.mean():.4f}%, max={vx_relerror.max():.4f}%")
    print(f"Percentual Vel_Y Error: mean={vy_relerror.mean():.4f}%, max={vy_relerror.max():.4f}%")
    print(f"Percentual Vel_Z Error: mean={vz_relerror.mean():.4f}%, max={vz_relerror.max():.4f}%")
    print("------------------------------------------------------------------")


def Divergent_Residual(batch_inputs, batch_outputs):
    # For each spatial dimension: ux, uy, uz
    # Gets tuple of gradients for velocity of direction c.
    output_grads    = torch.gradient(batch_outputs, dim=(2,3,4))
    du_dz           = output_grads[0]  
    du_dy           = output_grads[1]  
    du_dx           = output_grads[2]  
    duz_dz          = du_dz[:, 0:1, :, :, :]
    duy_dy          = du_dy[:, 1:2, :, :, :]
    dux_dx          = du_dx[:, 2:3, :, :, :]
    div             = duz_dz + duy_dy + dux_dx
    abs_div         = (duz_dz.abs() + duy_dy.abs() + dux_dx.abs())
    
    
    print("Evaluating Divergent:")
    div_means = []
    for s_i in range(div.shape[0]):
        div_i       = div           [s_i:s_i+1, : ,:, :, :]
        abs_div_i   = abs_div       [s_i:s_i+1, : ,:, :, :]
        s_i_mask    = (batch_inputs  [s_i:s_i+1, : ,:, :, :]!=0) & (abs_div_i.abs() > 1e-12)
        norm_div_i  = 100*div_i[s_i_mask].abs() / (abs_div_i[s_i_mask]+1e-12)
        mean_i      = norm_div_i.mean()
        
        print(" -- Sample {}: {:.4f}%".format(s_i, mean_i.item()))
        div_means.append(mean_i.item())
    print(f"Mean: {np.mean(div_means):.4f}%, Std: {np.std(div_means):.4f}%")
    print("------------------------------------------------------------------")
    return div_means
    
def Tortuosity_Comparison(batch_inputs, batch_outputs, batch_targets):
    vz_true     = batch_targets[:, 0:1, :, :,:]
    vy_true     = batch_targets[:, 1:2, :, :,:]
    vx_true     = batch_targets[:, 2:3, :, :,:]
    
    vz_pred     = batch_outputs[:, 0:1, :, :,:]
    vy_pred     = batch_outputs[:, 1:2, :, :,:]
    vx_pred     = batch_outputs[:, 2:3, :, :,:]
    
    mag_true    = (vx_true**2 + vy_true**2 + vz_true**2).sqrt()
    mag_pred    = (vx_pred**2 + vy_pred**2 + vz_pred**2).sqrt()
    
    print("Tortuosity Comparison:")
    ets = []
    for s_i in range(mag_pred.shape[0]):
        s_i_mask    = batch_inputs[s_i:s_i+1]!=0
        mag_true_i  = mag_true[s_i:s_i+1]
        mag_pred_i  = mag_pred[s_i:s_i+1]
        vz_true_i   = vz_true[s_i:s_i+1]
        vz_pred_i   = vz_pred[s_i:s_i+1]
        tort_true   = mag_true_i[s_i_mask].mean() / vz_true_i.mean()
        tort_pred   = mag_pred_i[s_i_mask].mean() / vz_pred_i.mean()
        
        et = 100*(tort_true-tort_pred).abs() / tort_true
        print(" -- Sample {}: {:.4f}%".format(s_i, et.item()))
        ets.append(et.item())
    print(f"Mean: {np.mean(ets):.4f}%, Std: {np.std(ets):.4f}%")
    print("------------------------------------------------------------------")
    return ets
        
def Angular_Comparison(batch_inputs, batch_outputs, batch_targets):
    vz_true     = batch_targets[:, 0:1, :, :,:]
    vy_true     = batch_targets[:, 1:2, :, :,:]
    vx_true     = batch_targets[:, 2:3, :, :,:]
    
    vz_pred     = batch_outputs[:, 0:1, :, :,:]
    vy_pred     = batch_outputs[:, 1:2, :, :,:]
    vx_pred     = batch_outputs[:, 2:3, :, :,:]
    
    
    mag_true        = (vx_true**2 + vy_true**2 + vz_true**2).sqrt()
    mag_pred        = (vx_pred**2 + vy_pred**2 + vz_pred**2).sqrt()
    dot             = (vz_true*vz_pred + vy_true*vy_pred + vx_true*vx_pred) 
    den             = mag_true*mag_pred
    
    print("Angular error:")
    ae_means = []
    for s_i in range(mag_true.shape[0]):
        s_i_mask        = batch_inputs[s_i:s_i+1]!=0
        dot_i           = dot[s_i:s_i+1]
        den_i           = den[s_i:s_i+1]
        cos_sim_flat    = dot_i[s_i_mask] / den_i[s_i_mask]        
        cos_sim_flat    = torch.clamp(cos_sim_flat, min=-1.0, max=1.0)
        theta_rad       = torch.acos(cos_sim_flat)
        theta_deg       = torch.rad2deg(theta_rad)
        mean            = theta_deg.mean()
        std             = theta_deg.std()
        print(" -- Sample {}: mean={:.4f}%; std={:.4f}".format(s_i, mean.item(), std.item()))
        ae_means.append(mean.item())
    print(f"Mean: {np.mean(ae_means):.4f}%, Std: {np.std(ae_means):.4f}%")
    print("------------------------------------------------------------------")
    return ae_means

def Correlation_Comparison(batch_inputs, batch_outputs, batch_targets, npoints=5000):
    
    correlations = []
    
    print("Correlation comparison:")
    for b in range(batch_inputs.shape[0]):
        batch_input = batch_inputs[b]
        batch_output= batch_outputs[b]
        batch_target= batch_targets[b]
        
        fluid_mask  = (batch_input!=0)
        fluid_mask  = fluid_mask.expand(batch_output.shape[0],-1,-1,-1)
        
        x_flat = batch_output[fluid_mask].flatten()
        y_flat = batch_target[fluid_mask].flatten()
    
        correlation_matrix      = np.corrcoef(x_flat, y_flat)
        correlation_coefficient = correlation_matrix[0, 1]
        
        print(" -- Sample {}: {:.4f}".format(b, correlation_coefficient.item()))
        correlations.append(correlation_coefficient)
    print(f"Mean={np.mean(correlations):.4f}, Std={np.std(correlations):.4f}")
    print("------------------------------------------------------------------")
    return correlations


#######################################################
#************ GETTER COMPARISONS:          ***********#
#######################################################

def Test_Model_on_Dataset(dataloader, model, component, model_name, datasetname):
        
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']
    
    # Initialize lists to store metrics for EVERY sample in the dataset
    all_metrics = {
        "p_metrics": [], # Permeability
        "m_metrics": [], # Magnitude
        "a_metrics": [], # Angular
        "c_metrics": [], # Correlation
        "f_metrics": [], # Flux
        "t_metrics": [], # Tortuosity
        "d_metrics": []  # Divergent
    }

    # Iterate over the entire dataset
    for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        
        current_bs = batch_inputs.shape[0]
        print(f"Processing Batch {batch_idx+1} (Size: {current_bs})...")

        # 1. Prediction
        batch_inputs  = batch_inputs.clone().detach().to(dtype=torch.float32)
        batch_outputs = model.predict(batch_inputs)
            
        # 2. Casting and transformation
        batch_targets = batch_targets.clone().detach().to(dtype=torch.float32)
        batch_outputs = batch_outputs.clone().detach().to(dtype=torch.float32)
        
        batch_targets = mean_normalize(batch_inputs, batch_targets)
        batch_outputs = mean_normalize(batch_inputs, batch_outputs)
        
        # 3. Masking (Solid = 0)
        mask                = batch_inputs[:, 0:1, :, :, :] <= 0 
        mask                = mask.expand_as(batch_outputs)
        batch_outputs[mask] = 0.0

        # 4. Calculate & Collect Metrics
        if component==4: # If component is Z,Y,X
            all_metrics["p_metrics"].extend(Permeability_Comparison(batch_inputs, batch_outputs, batch_targets))
            all_metrics["m_metrics"].extend(Magnitude_Comparison   (batch_inputs, batch_outputs, batch_targets))
            all_metrics["a_metrics"].extend(Angular_Comparison     (batch_inputs, batch_outputs, batch_targets))
            all_metrics["c_metrics"].extend(Correlation_Comparison (batch_inputs, batch_outputs, batch_targets))
            all_metrics["f_metrics"].extend(Flux_Comparison        (batch_inputs, batch_outputs, batch_targets))
            all_metrics["t_metrics"].extend(Tortuosity_Comparison  (batch_inputs, batch_outputs, batch_targets))
            all_metrics["d_metrics"].extend(Divergent_Residual     (batch_inputs, batch_outputs))
            
        else: # If component is Z only
            bt_z = batch_targets[:, 0:1, :, :, :]
            bo_z = batch_outputs[:, 0:1, :, :, :]
            
            all_metrics["p_metrics"].extend(Permeability_Comparison(batch_inputs, bo_z, bt_z))
            all_metrics["m_metrics"].extend(Magnitude_Comparison   (batch_inputs, bo_z, bt_z))
            all_metrics["c_metrics"].extend(Correlation_Comparison (batch_inputs, bo_z, bt_z))
            all_metrics["f_metrics"].extend(Flux_Comparison        (batch_inputs, bo_z, bt_z))

    # --- Final Global Aggregation ---
    final_results = {}
    
    final_results["Permeability Error [%]"] = all_metrics["p_metrics"]
    final_results["Magnitude Error [%]"]    = all_metrics["m_metrics"]
    final_results["Correlation"]            = all_metrics["c_metrics"]
    final_results["Flux Error"]             = all_metrics["f_metrics"]
    
    if component==4: # If component is Z,Y,X
        final_results["Angular Error [Deg]"]    = all_metrics["a_metrics"]
        final_results["Tortuosity Error [%]"]   = all_metrics["t_metrics"]
        final_results["Divergent Residual [%]"] = all_metrics["d_metrics"]

    return final_results



#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################
# Choose component
# 0 - z models
# 1 - y models
# 2 - x models
# 3 - p models
# 4 - zyx models
# None - zyx-p models
component        = 0

batch_size       = 8
N_samples        = None # 'None' to consider all available samples
device           = 'cpu'

# DEFINE DATASETS
datasets        = {
    #"Trainning": "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_PressureWalls.h5",
    
    #"Spherical Pore":   "../NN_Datasets/ForceDriven/Test_SphPore_120_120_120.h5",
    #"Spherical Grain":  "../NN_Datasets/ForceDriven/Test_SphGrain_120_120_120.h5",
    #"Cylindrical Pore": "../NN_Datasets/ForceDriven/Test_CylinPore_120_120_120.h5",
    #"Cylindrical Grain":"../NN_Datasets/ForceDriven/Test_CylinGrain_120_120_120.h5",
    
    "Parker":       "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5",
    "Leopard":      "../NN_Datasets/ForceDriven/Test_Oliveira_Leopard_120_120_120.h5",
    "Kirby":        "../NN_Datasets/ForceDriven/Test_Oliveira_Kirby_120_120_120.h5",
    "Castle Gate":  "../NN_Datasets/ForceDriven/Test_Oliveira_CastleGate_120_120_120.h5",
    "Brown":        "../NN_Datasets/ForceDriven/Test_Oliveira_Brown_120_120_120.h5",
    "Upper Gray":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaUpperGray_120_120_120.h5",
    "Sinter Gray":  "../NN_Datasets/ForceDriven/Test_Oliveira_BereaSinterGray_120_120_120.h5",
    "Berea Buff":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaBuff_120_120_120.h5",
    "Berea":        "../NN_Datasets/ForceDriven/Test_Oliveira_Berea_120_120_120.h5",
    #"Bentheimer":   "../NN_Datasets/ForceDriven/Test_Oliveira_Bentheimer_120_120_120.h5",
    #"Bandera":      "../NN_Datasets/ForceDriven/Test_Oliveira_Bandera_120_120_120.h5",
    }


# DEFINE MODELS
models          = {}
# 1 Directional Flow Models
if component==0:
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_24_March_2026_04-02PM_Job16923/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA (Pr)"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA (Pr+Walls)"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-13PM_Job16071/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - Orig. Data Aug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
# Ux
elif component==2:
    
    # ARCHITECTURES COMPARISON (Ux)
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.x_model
    model_full_name = "./Trained_Models/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq"] = danny_model
    print_n_params(danny_model, pytorch=True)
        
# Uz, Uy, Uy
elif component==4:   
    baseline_model  = Danny_KerasModel()
    models["Baseline Danny (Ke) - Danny Data"] = baseline_model
    

#######################################################
#************ RUN ANALYSIS:                ***********#
#######################################################

# Lista para armazenar os dados no formato longo (long-format), ideal para pandas e seaborn
all_records = []

print("Loading baseline model")
if component==0:
    baseline_model  = Danny_KerasModel(component=0)

            
print("Starting test routine...")
for dataname, datapath in datasets.items():
    print("="*120)
    print("\n\n\n ", dataname, " results:\n\n")
    
    # Load data
    print("Loading data...")
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    dataset.component = component
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Compute metrics for each model
    for model_id, model in models.items():
        print("Testing model:  ", model_id)
        
        # metrics agora é um dicionário onde cada chave tem uma LISTA de valores
        metrics_lists = Test_Model_on_Dataset(dataloader, model, component=component, model_name=model_id, datasetname=dataname)
        print("\n-----------------------------------------------------\n")
        
        # Guardar cada valor de amostra separadamente
        for metric_name, values in metrics_lists.items():
            for v in values:
                all_records.append({
                    "Dataset": dataname,
                    "Model": model_id,
                    "Metric": metric_name,
                    "Value": float(v)
                })
    print()
    del dataloader


#######################################################
#************ SHOW RESULTS IN PLOTS:       ***********#
#######################################################

# Criar um DataFrame único com todos os dados
df_all = pd.DataFrame(all_records)

# Pegar as listas de métricas e modelos disponíveis
if not df_all.empty:
    metrics_list = df_all["Metric"].unique()
    dataset_order = sorted(df_all["Dataset"].unique())
    
    # Define estilo acadêmico limpo
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.3,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"]
        }
    )
    
    palette = sns.color_palette("colorblind")
    
    # Um gráfico por métrica (comparando TODOS os modelos juntos)
    for metric in metrics_list:
        df_metric = df_all[df_all["Metric"] == metric]
        
        plt.figure(figsize=(14, 6))
        
        ax = sns.boxplot(
            data=df_metric,
            x="Dataset",
            y="Value",
            hue="Model",
            order=dataset_order,
            palette=palette,
            width=0.7,
            showfliers=False
        )
        
        # Jitter leve (opcional, mantém distribuição visível sem poluir)
        sns.stripplot(
            data=df_metric,
            x="Dataset",
            y="Value",
            hue="Model",
            order=dataset_order,
            dodge=True,
            color="black",
            alpha=0.2,
            size=2
        )
        
        # Corrigir legenda duplicada (boxplot + stripplot)
        handles, labels = ax.get_legend_handles_labels()
        n_models = len(df_metric["Model"].unique())
        ax.legend(
            handles[:n_models],
            labels[:n_models],
            title="Model",
            frameon=False
        )
        
        # Títulos e labels
        plt.title(metric, fontsize=16, pad=15)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Dataset", fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        
        # Estética limpa
        sns.despine()
        plt.tight_layout()
        
        # Mostrar (ou salvar se quiser)
        plt.show()

else:
    print("Nenhum dado foi processado para gerar os gráficos.")