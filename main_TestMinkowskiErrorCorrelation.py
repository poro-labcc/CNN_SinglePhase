import os
import numpy              as np
import torch
import matplotlib.pyplot  as plt
import pandas             as pd
import porespy            as ps
from scipy.stats          import pearsonr
from skimage.measure      import euler_number  # Added for Euler characteristic
from torch.utils.data     import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import quantimpy.minkowski as mk
from Architectures.Unet   import Extended_DannyKo
from Utilities            import dataset_reader as dr

#######################################################
#************ UTILITY FUNCTIONS:            ***********#
#######################################################
def analyze_geometry_correlation(dataloader, model, component, datasetname, device='cpu'):
    
    returns = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            current_bs = batch_inputs.shape[0]
            print(f"Processing Batch {batch_idx+1}...")

            batch_inputs_dev  = batch_inputs.to(dtype=torch.float32, device=device)
            
            if hasattr(model, 'predict'):
                batch_outputs = model.predict(batch_inputs_dev)
            else:
                batch_outputs = model(batch_inputs_dev)
            
            # Move outputs to CPU for NumPy correlation functions
            batch_outputs = batch_outputs.cpu()
            
            for samp in range(current_bs):
                # Ensure boolean mask for geometry functions
                void_mask = (batch_inputs[samp, 0, ...] > 0).numpy().astype(bool)
                out_samp = batch_outputs[samp]
                tar_samp = batch_targets[samp]
                
                # 1. Analyse Error (Pearson R)
                if out_samp.shape[0] >= 3: 
                    out_mag = torch.sqrt(out_samp[0]**2 + out_samp[1]**2 + out_samp[2]**2)
                    tar_mag = torch.sqrt(tar_samp[0]**2 + tar_samp[1]**2 + tar_samp[2]**2)
                else:
                    out_mag = out_samp[0]
                    tar_mag = tar_samp[0]
                
                # Apply mask (convert back to tensor mask for masking outputs)
                void_mask_tensor = torch.tensor(void_mask)
                x_flat      = out_mag[void_mask_tensor].numpy().flatten()
                y_flat      = tar_mag[void_mask_tensor].numpy().flatten()
                
                corr_matrix = np.corrcoef(x_flat, y_flat)
                r_coeff     = corr_matrix[0, 1]
                

                # Evaluating Geomtrical Factors
                # Ensure your mask is integer (0 for solid, 1 for void/fluid)
                void_mask_int = void_mask.astype(bool)
                
                # Calculate the 4 Minkowski functionals
                W0, W1, W2, W3 = mk.functionals(void_mask_int)
                
                print(f"Volume Fraction (Porosity): {W0}")
                print(f"Surface Area: {W1}")
                print(f"Integral Mean Curvature: {W2}")
                print(f"Euler-Poincaré Characteristic: {W3}")

                euler_num = euler_number(void_mask, connectivity=3)
                
                # Store all metrics as a dictionary for easy unpacking
                returns.append({
                    'r_coeff': r_coeff,
                    #'euler': euler_num,
                    'Wink. 0': W0,
                    'Wink. 1': W1,
                    'Wink. 2': W2,
                    'Wink. 3': W3,
                })
    
    return returns

#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################

component  = 0
batch_size = 10  
N_samples  = 100 
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path defined as requested
model_folder = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_13_March_2026_02-16PM_Job16074/"
model_name   = "model_LowerValidationLoss.pth"
model_path   = os.path.join(model_folder, model_name)

# Ensure you have multiple datasets active here to see the combined plot
datasets = {
    "Training Data":     "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_Pressure.h5",
    "Cylindrical Grain": "../NN_Datasets/ForceDriven/Test_CylinGrain_120_120_120.h5",
    "Cylindrical Pore": "../NN_Datasets/ForceDriven/Test_CylinPore_120_120_120.h5",
    "Spherical Grain": "../NN_Datasets/ForceDriven/Test_SphGrain_120_120_120.h5",
    "Spherical Pore": "../NN_Datasets/ForceDriven/Test_SphPore_120_120_120.h5",
    #"Parker":       "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5",
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
    res = analyze_geometry_correlation(dataloader, danny_model, component, dataname, device)
    all_results[dataname] = res
    

# =====================================================
# PLOT: Geometry Metrics vs. Correlation (R)
# =====================================================
import os
import matplotlib.pyplot as plt

print("\nGenerating subplots...")

# 1. Reduced figsize to a more manageable size (12, 9)
# 2. Used layout='constrained' which is much smarter than tight_layout() for shared legends
fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
axes = axes.flatten()
color_palette = plt.cm.tab10.colors

# Define an output directory
plot_out_dir = "../NN_Results/Plots/Thickness_Correlation/"
os.makedirs(plot_out_dir, exist_ok=True)

# Define the metrics to plot in each subplot
metrics_config = [
    ('Wink. 0',   r"Minkowski $W_0$ (Volume)"),
    ('Wink. 1',   r"Minkowski $W_1$ (Surface Area)"),
    ('Wink. 2',   r"Minkowski $W_2$ (Mean Curvature)"),
    ('Wink. 3',   r"Minkowski $W_3$ (Euler Characteristic)")
]

for ax, (metric_key, x_label) in zip(axes, metrics_config):
    for idx, (dataname, res_list) in enumerate(all_results.items()):
        c = color_palette[idx % len(color_palette)]
        
        y_r_coeffs = [item['r_coeff'] for item in res_list]
        x_metric   = [item[metric_key] for item in res_list]
        
        # Plot scatter with dataname as the label
        ax.scatter(x_metric, y_r_coeffs, alpha=0.6, s=60, color=c, edgecolors='black', 
                   linewidth=0.5, label=dataname, zorder=3)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(r"Pearson Correlation ($R$)", fontsize=12)
    ax.axhline(1.0, color='grey', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

# Extract handles and labels, using a dictionary to prevent any duplicates
handles, labels = axes[0].get_legend_handles_labels()
unique_legend = dict(zip(labels, handles))

# Place the legend robustly at the top, allowing it to span columns cleanly
fig.legend(unique_legend.values(), unique_legend.keys(), 
           loc='outside lower center', ncol=min(4, len(unique_legend)), 
           frameon=True, fontsize=11)

# Set the title (no need for y=1.08 hacks thanks to constrained_layout)
fig.suptitle("Model Reliability vs. Minkowski Functionals", fontsize=16, fontweight='bold')

plot_path = os.path.join(plot_out_dir, "Geometry_vs_R_Correlation_Subplots.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved subplot grid to: {plot_path}")