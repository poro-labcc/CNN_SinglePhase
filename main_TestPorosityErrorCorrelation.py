import os
import numpy              as np
import torch
import matplotlib.pyplot  as plt
import pandas             as pd
import porespy            as ps
from scipy.stats          import pearsonr
from torch.utils.data     import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Architectures.Unet   import Extended_DannyKo
from Utilities            import dataset_reader as dr

#######################################################
#************ UTILITY FUNCTIONS:           ***********#
#######################################################
def analyze_thickness_correlation(dataloader, model, component, datasetname, device='cpu'):
    
    
    returns = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            current_bs = batch_inputs.shape[0]
            print(f"Processing Batch {batch_idx+1}...")

            batch_inputs_dev  = batch_inputs.to(dtype=torch.float32, device=device)
            
            #inlets              = np.zeros_like(vol, dtype=bool)
            #inlets[0, :, :]     = 1  
            #outlets             = np.zeros_like(vol, dtype=bool)    
            #outlets[-1, :, :]   = 1 
            #filt_vol            = ps.filters.trim_nonpercolating_paths(vol, inlets=inlets, outlets=outlets)
            
            
            if hasattr(model, 'predict'):
                batch_outputs = model.predict(batch_inputs_dev)
            else:
                batch_outputs = model(batch_inputs_dev)
            
            # Move outputs to CPU for NumPy correlation functions
            batch_outputs = batch_outputs.cpu()
            
            for samp in range(current_bs):
                void_mask = batch_inputs[samp, 0, ...] > 0
                out_samp = batch_outputs[samp]
                tar_samp = batch_targets[samp]
                
                # Analyse error
                if out_samp.shape[0] >= 3: 
                    out_mag = torch.sqrt(out_samp[0]**2 + out_samp[1]**2 + out_samp[2]**2)
                    tar_mag = torch.sqrt(tar_samp[0]**2 + tar_samp[1]**2 + tar_samp[2]**2)
                else:
                    out_mag = out_samp[0]
                    tar_mag = tar_samp[0]
                x_flat      = out_mag[void_mask].numpy().flatten()
                y_flat      = tar_mag[void_mask].numpy().flatten()
                corr_matrix = np.corrcoef(x_flat, y_flat)
                r_coeff     = corr_matrix[0, 1]
                
                # Analyse thickness accordance
                void_mask = void_mask.numpy().astype(bool)
                dissimilarity_perc  = np.mean(void_mask)
                
                returns.append((r_coeff, dissimilarity_perc))
    
    return returns

#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################

component  = 0
batch_size = 4  
N_samples  = None 
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path defined as requested
model_folder = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_13_March_2026_02-16PM_Job16074/"
model_name   = "model_LowerValidationLoss.pth"
model_path   = os.path.join(model_folder, model_name)

# Ensure you have multiple datasets active here to see the combined plot
datasets = {
    "Training Data":     "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_Pressure.h5",
    #"Cylindrical Grain": "../NN_Datasets/ForceDriven/Test_CylinGrain_120_120_120.h5",
    #"Cylindrical Pore": "../NN_Datasets/ForceDriven/Test_CylinPore_120_120_120.h5",
    #"Spherical Grain": "../NN_Datasets/ForceDriven/Test_SphGrain_120_120_120.h5",
    #"Spherical Pore": "../NN_Datasets/ForceDriven/Test_SphPore_120_120_120.h5",
    "Parker":       "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5",
    "Leopard":      "../NN_Datasets/ForceDriven/Test_Oliveira_Leopard_120_120_120.h5",
    "Kirby":        "../NN_Datasets/ForceDriven/Test_Oliveira_Kirby_120_120_120.h5",
    "Castle Gate":  "../NN_Datasets/ForceDriven/Test_Oliveira_CastleGate_120_120_120.h5",
    "Brown":        "../NN_Datasets/ForceDriven/Test_Oliveira_Brown_120_120_120.h5",
    "Upper Gray":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaUpperGray_120_120_120.h5",
    "Sinter Gray":  "../NN_Datasets/ForceDriven/Test_Oliveira_BereaSinterGray_120_120_120.h5",
    "Berea Buff":   "../NN_Datasets/ForceDriven/Test_Oliveira_BereaBuff_120_120_120.h5",
    #"Berea":        "../NN_Datasets/ForceDriven/Test_Oliveira_Berea_120_120_120.h5",
    "Bentheimer":   "../NN_Datasets/ForceDriven/Test_Oliveira_Bentheimer_120_120_120.h5",
    #"Bandera":      "../NN_Datasets/ForceDriven/Test_Oliveira_Bandera_120_120_120.h5",
    
}

# Load Model
print("\nLoading Model...")
model_aux = Extended_DannyKo()
danny_model = model_aux.z_model
danny_model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
danny_model.eval()
danny_model.bin_input = True
danny_model.to(device)


#######################################################
#************ RUN ANALYSIS:                ***********#
#######################################################

all_results = {}

for dataname, datapath in datasets.items():
    print(f"\nAnalyzing Dataset: {dataname}")
    
    # Load Dataloader
    ids_to_load = np.arange(N_samples) if N_samples is not None else None
    dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=ids_to_load, x_dtype=torch.float32, y_dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run analysis and store in dictionary
    res = analyze_thickness_correlation(dataloader, danny_model, component, dataname, device)
    all_results[dataname] = res
    

# =====================================================
# PLOT: % Voxels Outside Thickness Range vs. Correlation (R)
# =====================================================
print("\nGenerating scatter plot...")

fig, ax = plt.subplots(figsize=(8, 6))
color_palette = plt.cm.tab10.colors

# Define an output directory
plot_out_dir = "../NN_Results/Plots/Thickness_Correlation/"
os.makedirs(plot_out_dir, exist_ok=True)

for idx, (dataname, res_list) in enumerate(all_results.items()):
    c = color_palette[idx % len(color_palette)]
    y_r_coeffs = [item[0] for item in res_list]
    x_dissim   = [item[1] for item in res_list]
    
    # Scatter points
    ax.scatter(x_dissim, y_r_coeffs, alpha=0.4, s=60, color=c, edgecolors='black', 
               linewidth=0.5, label=dataname, zorder=3)

ax.set_xlabel(r"Voxels Outside Thickness Range 5-17 (%)", fontsize=13)
ax.set_ylabel(r"Pearson Correlation Coefficient ($R$)", fontsize=13)
ax.set_title("Model Reliability vs. Out-of-Distribution Geometry", fontsize=14, pad=15)
ax.axhline(1.0, color='grey', linestyle=':', alpha=0.5, label='Perfect Prediction')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), frameon=True, fontsize=10)

plt.tight_layout()
plot_path = os.path.join(plot_out_dir, "Dissimilarity_vs_R_Correlation.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved scatter plot to: {plot_path}")