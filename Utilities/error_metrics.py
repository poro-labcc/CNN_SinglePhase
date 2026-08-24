import numpy              as np
import torch
from Utilities import velocity_usage as vu


#######################################################
# Testing Errors (not intended for NN training)       #
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
    print(f"Mean: {np.mean(efs):.4f}, Std: {np.std(efs):.4f}")
    print("------------------------------------------------------------------")
    return efs

def Bias_Comparison(batch_inputs, batch_outputs, batch_targets):
    print("In Bias Comparasion")
    if batch_outputs.shape[1] > 1:
        print("Using magnitude")
        v_true     = (batch_targets[:, 0:1, :, :, :]**2+batch_targets[:, 1:2, :, :, :]**2+batch_targets[:, 2:3, :, :, :]**2).sqrt()
        v_pred     = (batch_outputs[:, 0:1, :, :, :]**2+batch_outputs[:, 1:2, :, :, :]**2+batch_outputs[:, 2:3, :, :, :]**2).sqrt()
    else:
        print("Using Single Component")
        v_true     = batch_targets[:, 0:1, :, :, :]
        v_pred     = batch_outputs[:, 0:1, :, :, :]
        
    print("Out: ", batch_outputs.mean(), "; Tar: ", batch_targets.mean())

    print("Bias Error:")
    pe_means = []
    for s_i in range(v_pred.shape[0]):
        # Sample's void space
        s_i_mask    = batch_inputs[s_i:s_i+1]>0
        # Sample's Bias
        vz_true_i   = v_true[s_i:s_i+1][s_i_mask].mean()
        vz_pred_i   = v_pred[s_i:s_i+1][s_i_mask].mean()
        
        print("!!!!!  Pred: ", vz_pred_i, "; True ", vz_true_i)
        pe_i        = ( 100*(vz_pred_i-vz_true_i).abs() / vz_true_i.abs() ).item()
        print(" -- Sample {}: {:.4f}".format(s_i, pe_i))
        pe_means.append(pe_i)
    print(f"Mean: {np.mean(pe_means):.4f}, Std: {np.std(pe_means):.4f}")
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

    print("Evaluating Divergent:")
    div_means = []
    for s_i in range(batch_inputs.shape[0]):
        output  = batch_outputs[s_i:s_i+1, :, :,:,:]
        mask    = (batch_inputs[s_i:s_i+1, :, :,:,:]>0)
        duz_dz  = vu.d_dz(output, mask, c=0)
        duy_dy  = vu.d_dy(output, mask, c=1)
        dux_dx  = vu.d_dx(output, mask, c=2)
        div     = (duz_dz + duy_dy + dux_dx)
        abs_div = (duz_dz.abs() + duy_dy.abs() + dux_dx.abs())
        mask    = abs_div>0
        mean_i  = (100*div[mask]/abs_div[mask]).abs().mean()
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

def Correlation_Comparison(batch_inputs, batch_outputs, batch_targets):
    
    correlations = []
    
    print("Correlation comparison:")
    for b in range(batch_inputs.shape[0]):
        # Get one sample
        batch_input = batch_inputs[b]
        batch_output= batch_outputs[b]
        batch_target= batch_targets[b]
        
        if batch_output.shape[0]>1:
            output_mag = (batch_output[0:1]**2+batch_output[1:2]**2+batch_output[2:3]**2).sqrt()
            target_mag = (batch_target[0:1]**2+batch_target[1:2]**2+batch_target[2:3]**2).sqrt()
        else:
            output_mag = batch_output
            target_mag = batch_target
        
        # Get mask and expand to all channels
        fluid_mask  = (batch_input>0)
        
        x_flat = output_mag[fluid_mask].flatten()
        y_flat = target_mag[fluid_mask].flatten()
    
        correlation_matrix      = np.corrcoef(x_flat, y_flat)
        correlation_coefficient = correlation_matrix[0, 1]
        
        print(" -- Sample {}: {:.4f}".format(b, correlation_coefficient.item()))
        correlations.append(correlation_coefficient)
    print(f"Mean={np.mean(correlations):.4f}, Std={np.std(correlations):.4f}")
    print("------------------------------------------------------------------")
    return correlations
 
