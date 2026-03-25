import numpy              as np
import torch
import tensorflow         as tf
import matplotlib.pyplot  as plt
import pandas             as pd
from torch.utils.data     import DataLoader

from Architectures.Models import Corrected_MS_Net, DannyKo_Net_Original
from Utilities            import dataset_reader as dr
from Danny_Original.architecture import Danny_KerasModel


#######################################################
#************ UTILS:                       ***********#
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

    print("Trainable params:     ", trainable)
    print("Non-trainable params: ", non_trainable)
    print("Total params:         ", trainable + non_trainable)


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

def Test_Model_on_Dataset(dataloader, model, directional, model_name, datasetname):
        
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
        
        print("output shape: ", batch_outputs.shape)
        print("target shape: ", batch_targets.shape)
            
        # 2. Casting and transformation
        batch_targets = batch_targets.clone().detach().to(dtype=torch.float32)
        batch_outputs = batch_outputs.clone().detach().to(dtype=torch.float32)
        
        batch_targets = mean_normalize(batch_inputs, batch_targets)
        batch_outputs = mean_normalize(batch_inputs, batch_outputs)
        
        
        
        # 3. Masking (Solid = 0)
        mask                = batch_inputs[:, 0:1, :, :, :] <= 0 
        mask                = mask.expand_as(batch_outputs)
        batch_outputs[mask] = 0.0

        # 5. Calculate & Collect Metrics
        if directional==4: # If directional is Z,Y,X
            # 3D Metrics
            all_metrics["p_metrics"].extend(Permeability_Comparison(batch_inputs, batch_outputs, batch_targets))
            all_metrics["m_metrics"].extend(Magnitude_Comparison   (batch_inputs, batch_outputs, batch_targets))
            all_metrics["a_metrics"].extend(Angular_Comparison     (batch_inputs, batch_outputs, batch_targets))
            all_metrics["c_metrics"].extend(Correlation_Comparison (batch_inputs, batch_outputs, batch_targets))
            all_metrics["f_metrics"].extend(Flux_Comparison        (batch_inputs, batch_outputs, batch_targets))
            all_metrics["t_metrics"].extend(Tortuosity_Comparison  (batch_inputs, batch_outputs, batch_targets))
            all_metrics["d_metrics"].extend(Divergent_Residual     (batch_inputs, batch_outputs))
            
            
        else: # If directional is Z only
            # 1D Metrics (Z-direction only)
            bt_z = batch_targets[:, 0:1, :, :, :]
            bo_z = batch_outputs[:, 0:1, :, :, :]
            
            all_metrics["p_metrics"].extend(Permeability_Comparison(batch_inputs, bo_z, bt_z))
            all_metrics["m_metrics"].extend(Magnitude_Comparison   (batch_inputs, bo_z, bt_z))
            all_metrics["c_metrics"].extend(Correlation_Comparison (batch_inputs, bo_z, bt_z))
            all_metrics["f_metrics"].extend(Flux_Comparison        (batch_inputs, bo_z, bt_z))
            
                
            

    # --- Final Global Aggregation ---
    final_results = {}
    
    final_results["Mean Permeability Error [%]"] = np.mean(all_metrics["p_metrics"])
    final_results["Mean Magnitude Error [%]"]    = np.mean(all_metrics["m_metrics"])
    final_results["Mean Correlation"]            = np.mean(all_metrics["c_metrics"])
    final_results["Mean Flux Error"]             = np.mean(all_metrics["f_metrics"])
    
    if directional==4: # If directional is Z,Y,X
        final_results["Mean Angular Error [Deg]"]    = np.mean(all_metrics["a_metrics"])
        final_results["Mean Tortuosity Error [%]"]       = np.mean(all_metrics["t_metrics"])
        final_results["Mean Divergent Residual [%]"]     = np.mean(all_metrics["d_metrics"])

    return final_results



#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################
# Choose directional
# 0 - z models
# 1 - y models
# 2 - x models
# 3 - p models
# 4 - zyx models
# None - zyx-p models
directional      = 0  

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
    #"Leopard":      "../NN_Datasets/ForceDriven/Test_Oliveira_Leopard_120_120_120.h5",
    #"Kirby":        "../NN_Datasets/ForceDriven/Test_Oliveira_Kirby_120_120_120.h5",
    #"Castle Gate":  "../NN_Datasets/ForceDriven/Test_Oliveira_CastleGate_120_120_120.h5",
    #"Brown":        "../NN_Datasets/ForceDriven/Test_Oliveira_Brown_120_120_120.h5",
    #"Upper Gray":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaUpperGray_120_120_120.h5",
    #"Sinter Gray":  "../NN_Datasets/ForceDriven/Test_Oliveira_BereaSinterGray_120_120_120.h5",
    #"Berea Buff":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaBuff_120_120_120.h5",
    #"Berea":        "../NN_Datasets/ForceDriven/Test_Oliveira_Berea_120_120_120.h5",
    #"Bentheimer":   "../NN_Datasets/ForceDriven/Test_Oliveira_Bentheimer_120_120_120.h5",
    #"Bandera":      "../NN_Datasets/ForceDriven/Test_Oliveira_Bandera_120_120_120.h5",
    }


# DEFINE MODELS
models          = {}
# 1 Directional Flow Models
if directional==0:
    
    # Baseline model
    print("\nLoading Danny Ko (Baseline)...")
    baseline_model  = Danny_KerasModel(uni_directional=0)
    print_n_params(baseline_model.model, pytorch=False)
    models["Baseline Danny (Ke) - Danny Data"] = baseline_model
    

    # DATASETS COMPARISON
    """
    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-11PM_Job16070/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - SO"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-13PM_Job16071/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - SOA"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_15_March_2026_03-30PM_Job16205/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - ST"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA"] = danny_model
    print_n_params(danny_model, pytorch=True)
    """
    
    
    # ARCHITECTURES COMPARISON
    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA"] = danny_model
    print_n_params(danny_model, pytorch=True)

    javier_model = Corrected_MS_Net()
    model_full_name = "./Trained_Models/NN_Trainning_14_March_2026_10-52PM_Job16201/model_LowerValidationLoss.pth"
    javier_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    javier_model.eval()
    javier_model.bin_input = False
    models["Javier Arq. - STA"] = javier_model
    print_n_params(javier_model, pytorch=True)

    model_aux       = DannyKo_Net_Original()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_16_March_2026_12-08PM_Job16226/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA (Corr)"] = danny_model
    print_n_params(danny_model, pytorch=True)

    
    
    
# 3 Directional Flow Models
elif directional==4:    
    baseline_model  = Danny_KerasModel()
    models["Baseline Danny (Ke) - Danny Data"] = baseline_model
    

#######################################################
#************ RUN ANALYSIS:                ***********#
#######################################################

results = {}
def stash_metrics(dataname: str, model_id, metrics: dict):
    for metric_name, value in metrics.items():
        results.setdefault(metric_name, {})
        results[metric_name].setdefault(model_id, {})
        results[metric_name][model_id][dataname] = float(value)
        
        
print("Loading baseline model")
if directional==0:
    baseline_model  = Danny_KerasModel(uni_directional=0)

            
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
    
    dataset.uni_directional = directional
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Compute metrics for each model
    for model_id, model in models.items():
        print("Testing model:  ", model_id)
        # Compute metri, cs
        metrics = Test_Model_on_Dataset(dataloader, model, directional=directional, model_name=model_id, datasetname=dataname)
        print("\n-----------------------------------------------------\n")
        # Register metrics achieved
        stash_metrics(dataname, model_id, metrics)
    print()
    
    del dataloader
    
#######################################################
#************ SAVE RESULTS IN DATAFRAME:   ***********#
#######################################################

dfs = {}
for metric_name, metric_cube in results.items():
    df = pd.DataFrame.from_dict(metric_cube, orient="index")
    df = df.reindex(columns=list(datasets.keys()))  # garante ordem das colunas
    df = df.T
    df['Mediana'] = df.median(axis=1)
    df.loc['Mediana'] = df.median(axis=0)
    dfs[metric_name] = df
    print()
    print(metric_name)
    print(df)
    print()

if directional==0:
    path = "../NN_Results/Z_"
else:
    path = "../NN_Results/"

GREEN_CELL = "green!25"
RED_CELL   = "red!25"

for metric_name, df in dfs.items():
    latex_path  = path + f"{metric_name.replace(' ', '_')}.tex"
    print("Saving dataframe in: ", latex_path)
    n_models    = df.shape[1] - 1
    col_fmt     = "P{3cm} " + " ".join(["P{2cm}"] *n_models ) + " | P{2cm}"
    
    # which columns to color (e.g. skip the first one = baseline)
    columns_to_color = list(df.columns)

    # ------------- choose rule based on metric_name -------------
    # defaults: no coloring
    mode        = "none" # "lower_better", "higher_better" or "none"
    green_thr   = None
    red_thr     = None
    
    # EXAMPLES – you will adapt these:
    if "Mean Permeability Error [%]" in metric_name: # ok
        mode = "lower_better"
        green_thr = 20
        red_thr   = 50  

    elif "Mean Magnitude Error [%]" in metric_name: # ok
        mode = "lower_better"
        green_thr = 20  
        red_thr   = 50 

    elif "Mean Correlation" in metric_name: # ok
        mode = "higher_better"
        green_thr = 0.8
        red_thr   = 0.5
        
    elif "Mean Flux Error" in metric_name: # ok
        mode = "lower_better"
        green_thr = 0.2
        red_thr   = 0.5 
        
    elif "Mean Angular Error [Deg]" in metric_name:# ok
        mode = "lower_better"
        green_thr = 15
        red_thr   = 75
        
    elif "Mean Tortuosity Error [%]" in metric_name:# ok
        mode = "lower_better"
        green_thr = 20
        red_thr   = 50
        
    elif "Mean Divergent Residual [%]" in metric_name:
        mode = "lower_better"
        green_thr = 5
        red_thr   = 20
        
        
            

    # ------------- formatter using the chosen rule -------------
    def make_formatter(col_name):
        def formatter(v):
            if pd.isna(v):
                return ""

            color_prefix = ""
            if col_name in columns_to_color and mode != "none":
                if mode == "lower_better" and green_thr is not None and red_thr is not None:
                    if v < green_thr:
                        color_prefix = f"\\cellcolor{{{GREEN_CELL}}}"
                    elif v > red_thr:
                        color_prefix = f"\\cellcolor{{{RED_CELL}}}"
                elif mode == "higher_better" and green_thr is not None and red_thr is not None:
                    if v > green_thr:
                        color_prefix = f"\\cellcolor{{{GREEN_CELL}}}"
                    elif v < red_thr:
                        color_prefix = f"\\cellcolor{{{RED_CELL}}}"

            return f"{color_prefix}{v:.4f}"
        return formatter

    formatters = {col: make_formatter(col) for col in df.columns}

    latex_body = df.to_latex(
        column_format=col_fmt,
        escape=False,
        formatters=formatters
    )
    
    latex_body = latex_body.replace("\nMediana", "\n\\hline\nMediana")
    metric_name = metric_name.replace("%", "\\%")

    wrapped_latex = (
        "\\begin{table}[h!]\n"
        "    \\centering\n"
        "    \\footnotesize\n"
        f"    \\caption{{{metric_name}}}\n"
        
        
        "    \\label{tab:results}\n"
        + latex_body +
        "\\end{table}\n"
    )

    with open(latex_path, "w") as f:
        f.write(wrapped_latex)
