import numpy          as np
from   scipy.stats    import gaussian_kde
import matplotlib.pyplot as plt
import seaborn        as sns
import pandas         as pd
import os

from   Utilities      import loss_functions as lf
from   Utilities      import Domain_Plotter as pl
from   Utilities      import array_handler as ah

def compute_performance(loss_functions, loader, predictions, model_path):

    evaluation_results = []
    
    
    for i, (batch_input, batch_target) in enumerate(loader):
        batch_output    = predictions[i]
        sample_losses   = {}
        # Evaluate each loss function for the current sample
        for loss_name, loss_info in loss_functions.items():
            loss_obj                    = loss_info["obj"]            
            loss_value                  = loss_obj(batch_output, batch_target).item()
            sample_losses[loss_name]    = loss_value
        # Add the sample's results to our list
        evaluation_results.append(sample_losses)
        
    
    # Create the DataFrame from the list of dictionaries
    df              = pd.DataFrame(evaluation_results)
    # Calculate the mean and standard deviation for each column
    mean_row        = df.mean().to_frame().T
    std_row         = df.std().to_frame().T
    # Rename the index for the new rows
    mean_row.index  = ['Mean']
    std_row.index   = ['Std']
    # Concatenate the new rows to the DataFrame
    final_df        = pd.concat([df, mean_row, std_row])
    
    os.makedirs(model_path, exist_ok=True)
    final_df.to_csv(model_path+"model_performance.csv")
    print(f"\nSummary table saved as 'model_performance.csv' in {model_path}")
    
    return final_df

    
def analyze_permeabilities(phyVisc, bodyForce, voxel_phyRes, loader, outputs, path):
    rel_errors = []
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_outputs                       = outputs[i]  
        
        scale_net_target  = batch_targets.squeeze(0).squeeze(0).detach().cpu().numpy()
        scale_net_output  = batch_outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
            
        print(f"Sample {i}:")
        target_perm = np.mean(scale_net_target)*phyVisc*voxel_phyRes**2/np.abs(bodyForce)
        output_perm = np.mean(scale_net_output)*phyVisc*voxel_phyRes**2/np.abs(bodyForce)
        print("-> mean target: ", np.mean(scale_net_target))
        print("-> mean output: ", np.mean(scale_net_output))
        
        rel_error = 100*np.abs(target_perm-output_perm)/target_perm
        print(f"-> rel_error [%]: {rel_error}")
        
        rel_errors.append(rel_error)
        
    return rel_errors
            
def analyze_input_target_output_domain(loader, outputs, path):
    
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output                       = outputs[i]  
        
        arr_input   = batch_inputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        arr_target  = batch_targets.squeeze(0).squeeze(0).detach().cpu().numpy()
        arr_output  = batch_output.squeeze(0).squeeze(0).detach().cpu().numpy()

        arr_target = ah.Set_solids_to_value(arr_target, arr_input)
        arr_output = ah.Set_solids_to_value(arr_output, arr_input)
                

        t_mean, t_std = np.mean(arr_target), np.std(arr_target)
        o_mean, o_std = np.mean(arr_output), np.std(arr_output)
        
        low_t,  high_t  = t_mean - 5.0*t_std, t_mean + 5.0*t_std
        low_o,  high_o  = o_mean - 5.0*o_std, o_mean + 5.0*o_std
        
        # Option A: Python built-ins
        global_min = min(low_t, low_o)
        global_max = max(high_t, high_o)
            

        # Plot 
        dimz, dimy, dimx = arr_input.shape
        
        pl.Plot_Continuous_Domain_2D(
            values=arr_input[dimz//2,:,:],
            filename=path+f"Sample{i}/{dimz}_xyFront_INPUT",
            colormap="plasma",
            vmin=np.min(arr_input[dimz//2,:,:]),
            vmax=np.max(arr_input[dimz//2,:,:]),
            show_colorbar=True,
            special_colors={0: (1,1,1,1), 1: (0,0,0,1)}
        )
        pl.Plot_Continuous_Domain_2D(
            values=arr_target[dimz//2,:,:],
            filename=path+f"Sample{i}/{dimz}_xyFront_TARGET",
            colormap="plasma",
            vmin=global_min,
            vmax=global_max,
            show_colorbar=True,
            special_colors={0: (1,1,1,1)}
        )
        pl.Plot_Continuous_Domain_2D(
            values=arr_output[dimz//2,:,:],
            filename=path+f"Sample{i}/{dimz}_xyFront_OUTPUT",
            remove_value=None,              
            colormap="plasma",
            vmin=global_min,
            vmax=global_max,
            show_colorbar=True,
            special_colors={0: (1,1,1,1)}
        )
        
        
        pl.Plot_Continuous_Domain_2D(
            values=arr_input[:,dimy//2,:].T,
            filename=path+f"Sample{i}/{dimz}_xzSide_INPUT",
            colormap="plasma",
            vmin=np.min(arr_input[:,:,:]),
            vmax=np.max(arr_input[:,:,:]),
            show_colorbar=True,
            special_colors={0: (1,1,1,1), 1: (0,0,0,1)}
        )
        pl.Plot_Continuous_Domain_2D(
            values=arr_target[:,dimy//2,:].T,
            filename=path+f"Sample{i}/{dimz}_xzSide_TARGET",
            colormap="plasma",
            vmin=global_min,
            vmax=global_max,
            show_colorbar=True,
            special_colors={0: (1,1,1,1)}
        )
        pl.Plot_Continuous_Domain_2D(
            values=arr_output[:,dimy//2,:].T,
            filename=path+f"Sample{i}/{dimz}_xzSide_OUTPUT",
            remove_value=None,              
            colormap="plasma",
            vmin=global_min,
            vmax=global_max,
            show_colorbar=True,
            special_colors={0: (1,1,1,1)}
        )
        
        pl.plot_line_in_domain(arr_target, arr_output, save_path=path+f"Sample{i}/{dimz}_outputLine")
    
        pl.plot_plane_sums(arr_input, arr_target, arr_output, save_path=path+f"Sample{i}/{dimz}_fluxLine")            
        
            

        
def analyze_domain_error(loader, outputs, path):
    # Performance of each sample
    rel_error_all_samples = {}
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output = outputs[i]
        
        
        scale_net_input   = batch_inputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        scale_net_target  = batch_targets.squeeze(0).squeeze(0).detach().cpu().numpy()
        scale_net_output  = batch_output.squeeze(0).squeeze(0).detach().cpu().numpy()
            
        # For each sample's scale
        #for j, (scale_net_input, scale_net_target, scale_net_output) in  enumerate(zip(net_input_list, net_target_list, net_output_list)):
            
        dimx, dimy, dimz                = scale_net_output.shape
        
        error_abs                       = np.abs(scale_net_output - scale_net_target)
        
        # PLOT RELATIVE ERROR IN LOG SCALE
        void_mask                       = scale_net_target != 0.0
        
        relative_errorLog               = scale_net_target.copy()
        relative_errorLog[void_mask]    = np.log10(      (error_abs[void_mask] / np.abs(scale_net_target[void_mask]))+1 )            
        relative_errorLog[~void_mask]   = -1

        relative_error                  = scale_net_target.copy()
        relative_error[void_mask]       = 100* np.abs(   error_abs[void_mask] / scale_net_target[void_mask] )
        relative_error[~void_mask]      = -1           
        
        
        pl.Plot_Continuous_Domain_2D(
            values=relative_errorLog[dimx//2,:,:],
            title="Relative Error in Void Space: $log_10(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$",
            filename=path+f"Sample{i}/Absolute Log_REL_ERRO_Scale_{dimx}",
            colormap="inferno",
            show_colorbar=True,
            special_colors={-1: (1,1,1,1)}
        )
        
    
        
        
        pl.Plot_Continuous_Domain_2D(
            values=relative_error[dimx//2, :, :],
            filename=path+f"Sample{i}/{dimx}_Percentual Error",
            colormap="jet",
            show_colorbar=True,
            vmax=100,
            vmin=0,
            special_colors={-1: (1,1,1,1)}
        )

        """
        pl.plot_distributions(
            {"Relative Error in Void Space: $log_10(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$": relative_error[void_mask]},
            save_path=path+f"Sample{i}/Distribution Log_REL_ERROR_Scale_{dimx}"
        )
        """
        if dimx not in rel_error_all_samples:
            rel_error_all_samples[dimx] = []
        rel_error_all_samples[dimx].extend(relative_error[void_mask])
            
        
        
        
def analyze_population_distributions(loader, outputs, path):

    # Performance of all samples (per scale)        
    net_targets_inVoid = [] 
    net_outputs_inVoid = []
    net_inputs_inVoid  = []
    net_errors_inVoid = []
    net_relerrors_inVoid = []
    
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        
        batch_outputs = outputs[i]
        
        # Eliminate batch and channel dimensions
        net_input     = batch_inputs.squeeze(0).squeeze(0).detach().cpu().numpy()
        net_target    = batch_targets.squeeze(0).squeeze(0).detach().cpu().numpy()
        net_output    = batch_outputs.squeeze(0).squeeze(0).detach().cpu().numpy()

        dimx, dimy, dimz    = net_input.shape

        # Calculate error
        void_mask           = net_input>0 
        error               = np.abs(net_output - net_target)
        rel_error           = error[void_mask] / np.abs(net_target[void_mask])
        
        target_inVoid       = net_target[void_mask].tolist()
        input_inVoid        = net_input[void_mask].tolist()
        output_inVoid       = net_output[void_mask].tolist()
        error_inVoid        = error[void_mask].tolist()
        rel_error_inVoid    = rel_error.tolist()
        
        # Store only in-void values
        net_inputs_inVoid.append(net_input[void_mask])
        net_targets_inVoid.append(net_target[void_mask])
        net_outputs_inVoid.append(net_output[void_mask])
        net_errors_inVoid.append(error[void_mask])
        net_relerrors_inVoid.append(rel_error)
        

    # --------- Per-sample histograms ---------
    dic = {}
    for n, tar in enumerate(net_targets_inVoid):
        dic[f"Sample {n}"] = tar
        
    pl.plot_distributions(
        dic,
        save_path=path + "Targets_Histogram",
        normalize=False
    )
    
    dic = {}
    for n, inp in enumerate(net_inputs_inVoid):
        dic[f"Sample {n}"] = inp
    
    pl.plot_distributions(
        dic,
        save_path=path + "Inputs_Histogram",
        normalize=False
    )

    # --------- Global histograms ---------
    all_target       = np.concatenate(net_targets_inVoid)
    all_output       = np.concatenate(net_outputs_inVoid)
    all_rel_error    = np.concatenate(net_relerrors_inVoid)
    all_input        = np.concatenate(net_inputs_inVoid)
    all_error        = np.concatenate(net_errors_inVoid)
    
    npoints = 5000
    pl.plot_scatter_sampled(
        all_target, all_output,
        npoints=npoints,
        xlabel="Target", ylabel="Output",
        title=f'2D Scatter Plot: Target vs. Prediction ({npoints} Sampled Points)',
        save_path=path + "TargetVsOuput"
    )
    
    pl.plot_distributions(
        {
            "Targets": all_target,
            "Outputs": all_output,
        },
        save_path=path + "TargetsVsOutputs_Histogram",
        normalize=False
    )
    
    

        
        
    
    
def sanity_check(loader, outputs):
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output = outputs[i]
        # For each sample's scale
        for scale, (net_input, net_target, net_output) in  enumerate(zip(batch_inputs, batch_targets, batch_output)):
            # Convert tensors to array
            net_input  = net_input.squeeze(0).squeeze(0).detach().cpu().numpy()
            net_target = net_target.squeeze(0).squeeze(0).detach().cpu().numpy()
            net_output = net_output.squeeze(0).squeeze(0).detach().cpu().numpy() 
            
            solids_True             = net_input==0
            voids_True              = net_input!=0
            any_solid_w_velocity    = not np.any(net_target[solids_True] != 0) # If any velocity is non zero in solid, 
            any_void_wo_velocity    = not np.any(net_target[voids_True] == 0)  # If there is any void without velocity
            
            if any_solid_w_velocity: raise Exception("Error: some solid cell (from Input) has velocity in Target")
            if any_void_wo_velocity: raise Exception("Error: some void cell (from Input) has no velocity in Target")
                
        