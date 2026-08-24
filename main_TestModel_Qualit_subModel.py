import torch
import numpy              as np
import matplotlib.pyplot  as plt
import tensorflow         as tf
from scipy.stats          import gaussian_kde
from matplotlib.ticker    import LogLocator, LogFormatterSciNotation
from torch.utils.data     import DataLoader

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended

from Utilities            import dataset_reader as dr
from Danny_Original.architecture import Danny_KerasModel

  
#######################################################
#************ UTILS:                       ***********#
#######################################################

def mean_normalize(inp, x): 
    B, C, Z, Y, X = x.shape
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

def get_masked_slices(inp, tar, slice_idx, axis='front'):
    """Extracts and masks 2D slices from 3D volumes based on orientation."""
    if axis == 'front':
        # XY Plane (slice along Z)
        i_slc = inp[slice_idx, :, :].cpu().numpy()
        t_slc = tar[slice_idx, :, :].cpu().numpy()
    elif axis == 'side':
        # XZ Plane (slice along Y)
        i_slc = inp[:, :, slice_idx].cpu().numpy()
        t_slc = tar[:, :, slice_idx].cpu().numpy()
    
    mask = (i_slc == 0)
    return np.ma.array(t_slc, mask=mask)


#######################################################
#************ COMPARISONS:                 ***********#
####################################################### 

import os
def Plot_Front_Comparison(models, datapath, component, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models to 'Plot_Front_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)    # Remove batch dim,
    tar_z           = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='front') # Put zeros on solid
    
    # Prepare color range
    vmin, vmax      = np.percentile(tar_z_masked.compressed(), [1, 99])
    
    folder = "Plot_Front_Comparison_"+save_tag
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        # Save Target
        plt.figure(figsize=(6, 6))
        plt.imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        # Increased height from 5 to 6 to fit horizontal colorbars nicely
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].set_title("Target (Front View)")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
            
        #tar = mean_normalize(inp, tar)
        #out = mean_normalize(inp, out)
        
        out_z       = out.squeeze(0)[0]          # Remove batch dim, get first channel
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='front') # Put zeros on solid
        vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
            im = axes[i].imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} (Front)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()


def Plot_Side_Comparison(models, datapath, component, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models to 'Plot_Side_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx] # Shape (C,Z,Y,X)
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32) # Add channel for prediction
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)    # Remove batch dim,
    tar_z           = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='side') # Put zeros on solid
    
    # Prepare color range
    vmin, vmax      = np.percentile(tar_z_masked.compressed(), [1, 99])
    
    folder = "Plot_Side_Comparison_"+save_tag
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        # Save Target
        plt.figure(figsize=(6, 6))
        plt.imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        # Increased height from 5 to 6
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].axis('off')
        axes[0].set_title("Target (Side)")
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        #tar = mean_normalize(inp, tar)
        #out = mean_normalize(inp, out)
        
        out_z       = out.squeeze(0)[0]   
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='side') # Put zeros on solid
        vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} (Side)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()

def Plot_Error_Comparison(models, datapath, sample_idx=0, slice_idx=60, axis='front', save_mode=False, save_tag=""):
    """Saves Absolute Error maps to folder. (No target here as error is relative)."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)[0] # Remove batch dim, get first channel
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='front') # Put zeros on solid
    
    
    folder = f"Plot_Error_Comparison_{axis}_"+save_tag
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if not save_mode:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), constrained_layout=True)

    for i, (name, model) in enumerate(models.items()):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        out_z         = out.squeeze(0)[0]          # Remove batch dim, get first channel
        o_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='front') # Put zeros on solid
        
        error_map = np.abs(tar_z_masked - o_z_masked)

        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(error_map, cmap='Reds')
            plt.title(f"{name} Error ({axis})")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(error_map, cmap='Reds')
            axes[i].set_title(f"{name} Error")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    if not save_mode: plt.show()
    

def Plot_Mean_Velocity_Scatter(models, datapath, batch_size=4, npoints=5000, 
                               xlabel="Target Mean Velocity", ylabel="Predicted Mean Velocity", 
                               title="Mean Velocity Scale Accuracy", 
                               save_tag="default", save_mode=False, log=True):
    """
    Computes mean velocities per sample in batches.
    Matches the dataset loading and normalization logic of the other plotting functions.
    """
    # --- 1. Load Data (Updated to LazyDatasetTorch) ---
    dataset = dr.LazyDatasetTorch(h5_path=datapath, 
                                  list_ids=None, 
                                  x_dtype=torch.float32,
                                  y_dtype=torch.float32)
    
    # Dataloader automatically adds the Batch dimension (B, C, Z, Y, X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    folder = "Plot_Mean_Velocity_" + save_tag
    if save_mode and not os.path.exists(folder): 
        os.makedirs(folder)

    # --- 2. Better composite figure sizing ---
    if not save_mode:
        n_models = len(models)
        # Adjust figure size based on number of models
        fig_width = min(21, 7 * n_models)  # Cap maximum width
        fig_height = 6  # Slightly reduced from 7 to give more breathing room
        fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height), constrained_layout=True)
        if n_models == 1: 
            axes = [axes]
        # Add more spacing for titles
        fig.suptitle("", fontsize=16)  # Empty suptitle to trigger layout adjustment

    # --- 3. Iterate through Models ---
    for i, (name, model) in enumerate(models.items()):
        all_gt_means = []
        all_pred_means = []
        
        # --- 4. Process in Batches ---
        with torch.no_grad():
            for batch_inp, batch_tar in loader:
                # Ensure float32 (safety measure)
                batch_inp = batch_inp.to(dtype=torch.float32)
                batch_tar = batch_tar.to(dtype=torch.float32)

                # Normalize using the consistent util function
                output      = model.predict(batch_inp) if hasattr(model, 'predict') else model(batch_inp)
                
                dims        = tuple(range(1, batch_tar.ndim))
                all_gt_means.append(batch_tar.abs().mean(dim=dims).cpu().numpy())
                all_pred_means.append(output.abs().mean(dim=dims).cpu().numpy())

        x_data = np.concatenate(all_gt_means)
        y_data = np.concatenate(all_pred_means)
    
        # --- 5. Sampling and Density Logic ---
        np.random.seed(42)
        total_points        = len(x_data)
        indices             = np.random.choice(total_points, size=min(npoints, total_points), replace=False)
        x_sample, y_sample  = x_data[indices], y_data[indices]
        
        valid               = (x_sample > 0) & (y_sample > 0)
        x_sample, y_sample  = x_sample[valid], y_sample[valid]
    
        data_points         = np.vstack([np.log10(x_sample), np.log10(y_sample)]) if log else np.vstack([x_sample, y_sample])
        
        # Handle case where too few points remain after filtering
        if data_points.shape[1] < 2:
            print(f"Warning: Not enough valid points for {name}. Skipping density plot.")
            continue
            
        kde = gaussian_kde(data_points)
        density = kde(data_points)
        sort_idx = density.argsort()

        # --- 6. Frame Setup ---
        if save_mode:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
        else:
            ax = axes[i]

        # --- 7. Plotting Assets ---
        sc = ax.scatter(x_sample[sort_idx], y_sample[sort_idx], 
                        c=density[sort_idx], cmap='plasma', s=35, alpha=0.6)
        
        # Consistent Square Limits
        combined = np.concatenate([x_sample, y_sample])
        lo_lin, hi_lin = combined.min() * 0.8, combined.max() * 1.2
        line_vals = np.logspace(np.log10(lo_lin), np.log10(hi_lin), 100) if log else np.linspace(lo_lin, hi_lin, 100)
        
        ax.plot(line_vals, line_vals, color='gray', linestyle='--', linewidth=2, label='y=x', zorder=3)
        ax.set_xlim(lo_lin, hi_lin)
        ax.set_ylim(lo_lin, hi_lin)

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_locator(LogLocator(base=10))

        # Correlation and Labels - Make text smaller for composite view
        corr = np.corrcoef(x_data, y_data)[0, 1]
        fontsize_text = 12 if not save_mode else 16  # Smaller for composite
        ax.text(0.05, 0.95, f'$R = {corr:.4f}$', transform=ax.transAxes, 
                fontsize=fontsize_text, verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.8, pad=3))

        # Adjust label sizes for composite view
        label_fontsize = 12 if not save_mode else 16
        title_fontsize = 11 if not save_mode else 15
        
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        
        # Handle title more carefully for composite view
        if save_mode:
            ax.set_title(f"{name}\n{title}", fontsize=title_fontsize, fontweight='bold')
        else:
            # Shorter title for composite view to prevent overlapping
            ax.set_title(name, fontsize=title_fontsize, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.grid(True, which="major", linestyle="-", alpha=0.3)
        
        # --- 8. Colorbar handling ---
        if save_mode:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label('Density', fontsize=12)
        else:
            # For composite view, make colorbar smaller and position it better
            cbar = plt.colorbar(sc, ax=ax, fraction=0.08, pad=0.04, shrink=0.8)
            cbar.set_label('Density', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

        if save_mode:
            plt.savefig(f"{folder}/{name.replace(' ', '_')}_scatter.png", 
                       dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    if not save_mode:
        # Adjust layout one more time before showing
        plt.show()
        
        
#######################################################
#************ MAIN:                        ***********#
#######################################################

z_direction_only    = True
device              = 'cpu'
batch_size          = 1
save_mode           = False
sample_idexes       = [2]
#datapath            = "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5" 
datapath            = "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_Pressure.h5"

save_tag            = "Danny"
shape               = (120,120,120)
component           = 3 # Uz=0, Uy=1, Ux=2, P=3
models          = {}
# 1 Directional Flow Models
if component==0:
    
    """
    # Baseline model
    print("\nLoading Danny Ko (Baseline)...")
    baseline_model  = Danny_KerasModel(uni_directional=0)
    print_n_params(baseline_model.model, pytorch=False)
    models["Baseline Danny (Ke) - Danny Data"] = baseline_model
    
    # Dataset Danny Simulado - Sem Augmentation - Dados Alinhados
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-11PM_Job16070/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - Orig. Data noAug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-13PM_Job16071/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - Orig. Data Aug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_15_March_2026_03-30PM_Job16205/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    #nnt.load_model_from_checkpoint(danny_model, "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_13_March_2026_02-13PM_Job16072/", 1200)
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - My Data noAug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA"] = danny_model
    print_n_params(danny_model, pytorch=True)
    """
    
    # Comparing Javier and Danny Models
    """
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_March_2026_02-16PM_Job16074/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny Arq. - STA"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    javier_model = MS_Net()
    model_full_name = "./Trained_Models/NN_Trainning_14_March_2026_10-52PM_Job16201/model_LowerValidationLoss.pth"
    javier_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    javier_model.eval()
    javier_model.bin_input = False
    models["Javier Arq. - STA"] = javier_model
    print_n_params(javier_model, pytorch=True)
    """
    
    
    # Do Pressure (no Walls) removed deconvolution residual ?
    #"""
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
    #"""
    
elif component==2:
    
    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_14_March_2026_03-15PM_Job16196/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - Orig. Data Aug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    

elif component==3:

    model_aux       = Extended_DannyKo()
    danny_model     = model_aux.z_model
    model_full_name = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_24_March_2026_03-59PM_Job16921/model_LowerValidationLoss.pth"
    danny_model.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model.eval()
    danny_model.bin_input = True
    models["Danny - Orig. Data Aug"] = danny_model
    print_n_params(danny_model, pytorch=True)
    
    
# 3 Directional Flow Models
else:    
    baseline_model  = Danny_KerasModel()
    models["Baseline Danny (Ke) - Danny Data"] = baseline_model
    

# --- Execution Block ---

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']

# 3. Run Visualization
#Plot_Mean_Velocity_Scatter(models, datapath, npoints=5000, xlabel="Target Mean Velocity", 
#                                   ylabel="Predicted Mean Velocity", title="Mean Velocity Scale Accuracy", 
#                                   save_tag="default", save_mode=False, log=False)


for sample_idx in sample_idexes:
    
    Plot_Front_Comparison(models, datapath, component, sample_idx= sample_idx, slice_idx=shape[0]//2, save_mode=save_mode, save_tag = save_tag)
    Plot_Side_Comparison (models, datapath, component, sample_idx= sample_idx, slice_idx=shape[2]//2, save_mode=save_mode, save_tag = save_tag)
    #Plot_Error_Comparison(models, datapath, slice_idx=60, save_mode=save_mode, save_tag = save_tag, axis='side')
