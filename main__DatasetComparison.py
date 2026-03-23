import torch
import numpy as np
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

import matplotlib.pyplot as plt
import os
from matplotlib import patheffects # Add this import at the top of your script
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from   scipy.ndimage import distance_transform_edt


def Plot_Velocity_Front_Comparison(datasets, sample_idx=0, slice_idx=60, vel_channel=0, save_mode=False, save_tag=""):
    """
    Plots a slice of the velocity field side-by-side (Front View) with Flux Annotation.
    """
    num_plots = len(datasets)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), constrained_layout=True)
    
    if num_plots == 1: axes = [axes]
        
    comp_name = {0: "Uz", 1: "Uy", 2: "Ux"}.get(vel_channel, f"Ch {vel_channel}")
    folder = "Velocity_Front_Comparison_" + save_tag
    if save_mode and not os.path.exists(folder): 
        os.makedirs(folder)

    for i, (ds_name, ds) in enumerate(datasets.items()):
        _, targets = ds[sample_idx]
        targets = targets.numpy()
        
        vel_slice = targets[vel_channel, slice_idx, :, :]
        vel_masked = np.ma.masked_where(vel_slice == 0, vel_slice)
        
        cmap = plt.colormaps["plasma"].copy()
        cmap.set_bad("white")  
        im = axes[i].imshow(vel_masked, cmap=cmap)
        
        # --- HIGH-VISIBILITY FRONTAL ANNOTATION ---
        # Using \otimes to represent flow into the screen for Uz
        flux_text = r"$U_z$ $\otimes$"
        ann = axes[i].annotate(r'$U_z$ $\otimes$', 
                         xy=(0.5, 0.2),      # Same base position
                         xytext=(0.5, 0.08),  # No arrow length needed here
                         color='#00FF00', 
                         fontsize=18, 
                         fontweight='bold',
                         ha='center', 
                         va='bottom',
                         xycoords='axes fraction',
                         textcoords='axes fraction')

        ann.set_path_effects([
            patheffects.withStroke(linewidth=4, foreground='black')
        ])
        # ------------------------------------------
        
        clean_name = ds_name.replace('.h5', '')
        axes[i].set_title(f"{clean_name}\n({comp_name} - Front View)")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if save_mode:
        plt.savefig(f"{folder}/Sample_{sample_idx}_{comp_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def Plot_Velocity_Side_Comparison(datasets, sample_idx=0, slice_idx=60, vel_channel=0, save_mode=False, save_tag=""):
    """
    Plots a slice of the velocity field side-by-side for all datasets (Side View).
    """
    num_plots = len(datasets)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), constrained_layout=True)
    
    if num_plots == 1: axes = [axes]
        
    comp_name = {0: "Uz", 1: "Uy", 2: "Ux"}.get(vel_channel, f"Ch {vel_channel}")
    folder = "Velocity_Side_Comparison_" + save_tag
    if save_mode and not os.path.exists(folder): 
        os.makedirs(folder)

    for i, (ds_name, ds) in enumerate(datasets.items()):
        _, targets = ds[sample_idx]
        targets = targets.numpy()
        
        # Slice along the X-axis (Width)
        vel_slice = targets[vel_channel, :, :, slice_idx]
        
        vel_masked = np.ma.masked_where(vel_slice == 0, vel_slice)
        cmap = plt.colormaps["plasma"].copy()
        cmap.set_bad("white")  
        im = axes[i].imshow(vel_masked, cmap=cmap)
        ann = axes[i].annotate('$U_z$', 
                         xy=(0.5, 0.2),         # Arrow tip
                         xytext=(0.5, 0.08),     # Text position (closer to tail)
                         arrowprops=dict(
                             facecolor='#00FF00', 
                             edgecolor='black', 
                             linewidth=1.5,     # Border for the arrow
                             shrink=0.05, 
                             width=5, 
                             headwidth=15
                         ),
                         color='#00FF00', 
                         fontsize=18, 
                         fontweight='bold',
                         ha='center', 
                         va='center',
                         xycoords='axes fraction',
                         textcoords='axes fraction')

        # Add a black border/outline to the text
        ann.set_path_effects([
            patheffects.withStroke(linewidth=3, foreground='black')
        ])
        
        clean_name = ds_name.replace('.h5', '')
        axes[i].set_title(f"{clean_name}\n({comp_name} - Side View)")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if save_mode:
        plt.savefig(f"{folder}/Sample_{sample_idx}_{comp_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def Compare_Histograms(
        datasets,
        sample_indices,
        bins=150,
        save_folder="Normalization_comparisons", # Default folder
        filename="global_histogram_comparison.png"
    ):
    """
    Compares Uz (axis=0) and velocity magnitude distributions
    across ALL provided samples for each dataset and saves to a folder.
    """
    # 1. Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Adding a bold title with path effects for visibility
    #title = fig.suptitle("Global Velocity Distribution Comparison (All Samples)", fontsize=16, fontweight='bold')
    #title.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

    for ds_name, ds in datasets.items():
        uz_all = []
        mag_all = []
        
        for sample_idx in sample_indices:
            _, targets = ds[sample_idx]
            targets = targets.numpy()

            uz = targets[0]
            uy = targets[1]
            ux = targets[2]
            mag = np.sqrt(uz**2 + uy**2 + ux**2)
            
            uz_flat = uz.flatten()
            mag_flat = mag.flatten()
            fluid_mask = mag_flat > 1e-20

            uz_all.append(uz_flat[fluid_mask])
            mag_all.append(mag_flat[fluid_mask])

        # Concatenate all samples
        uz_all = np.concatenate(uz_all)
        mag_all = np.concatenate(mag_all)
        short_name = ds_name.replace(".h5", "")

        # Print global statistics
        print(f"\nDataset: {short_name}")
        print(f"   Uz  -> Mean: {uz_all.mean():.5f} | Std: {uz_all.std():.5f}")
        print(f"   Mag -> Mean: {mag_all.mean():.5f} | Std: {mag_all.std():.5f}")
        print("occ: ", len(uz_all))
        # Plot        
        axes[0].hist(uz_all, bins=bins, alpha=0.5, density=True, label=short_name)
        axes[1].hist(mag_all, bins=bins, alpha=0.5, density=True, label=short_name)

    # Formatting
    axes[0].set_xlabel("Componente $u_z$ ", fontsize=24)
    axes[0].set_ylabel("Densidade de Probabilidade", fontsize=18)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_xlabel("Magnitude $|\\vec{u}$|", fontsize=24)
    axes[1].set_ylabel("Densidade de Probabilidade", fontsize=18)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    
    #plt.tight_layout()
    axes[0].set_xlim((-0.05, 1.0))
    axes[1].set_xlim((-0.05, 1.0))
    

    # 2. Save logic
    full_save_path = os.path.join(save_folder, filename)
    plt.savefig(full_save_path, dpi=300)
    print(f"Histogram saved to: {full_save_path}")
    plt.close()
        
def Compare_Permeability(
        datasets,
        sample_indices,
    ):
    """
    Calculates and prints the permeability for all provided samples in each dataset.
    Permeability K = (mu * <Uz>) / F_z (Darcy's Law for force-driven flow)
    """
    print("\n" + "="*80)
    print(f"PERMEABILITY ANALYSIS")
    print("="*80)

    for ds_name, ds in datasets.items():
        short_name = ds_name.replace(".h5", "").split('/')[-1] # Clean up name
        print(f"\nDataset: {short_name}")
        
        perm_list = []
        rmax_list = []
        
        for sample_idx, (inputs, targets) in enumerate(ds):
            inputs_np   = inputs.numpy() 
            targets_np  = targets.numpy() # (3, Z, Y, X)
            
            # Denormalize
            Dens        = 1.0
            tau         = 1.5
            Re          = 0.1 
            visc        = (tau-0.5)/3
            Kn          = 0.2
            porous_mask = (inputs_np[0] > 0)
            force       = vu.force_calculation(porous_mask, tau=tau, Re=Re, Dens=Dens)
            r_max       = distance_transform_edt(porous_mask).max()
            D           = (2*r_max*0.65)
            targets_np  = targets_np*force*D**2*Kn/visc
            
            uz          = targets_np[0]
            k_star      = vu.calculate_permeability(porous_mask, uz)
                
            perm_list.append(k_star)
            rmax_list.append(r_max)
            print(f" ----  Sample {sample_idx}: {k_star}; force {force}")

        # Dataset Summary
        perm_arr = np.array(perm_list)
        rmax_arr = np.array(rmax_list)
        sort_idx = np.argsort(perm_arr)
        perm_arr = perm_arr[sort_idx]
        rmax_arr = rmax_arr[sort_idx]
        
        plt.scatter(perm_arr, rmax_arr)
        plt.show()
        print(f"   >>> {short_name} Summary | Mean K: {np.mean(perm_arr):.5f} | Std K: {np.std(perm_arr):.5f}")
        
    print("="*80 + "\n")
#######################################################
#************ INPUTS                      *************#
#######################################################
dataset_folder = "../NN_Datasets/"

# Comparar reconstrucao dados baseline
"""
dataset_names = [
    "Train_Original_Danny_noAug.h5", 
    "PressureDriven/Train_Danny_120_120_120_PressureWalls.h5",
    "ForceDriven/Train_Danny_120_120_120_Force.h5"
]
"""


datasets        = {
    #"Spherical Pore":   "ForceDriven/Test_SphPore_120_120_120.h5",
    #"Spherical Grain":  "ForceDriven/Test_CylinGrain_120_120_120.h5",
    #"Cylindrical Pore": "ForceDriven/Test_CylinPore_120_120_120.h5",
    #"Cylindrical Grain":"ForceDriven/Test_CylinGrain_120_120_120.h5",
    
    "Parker":       "ForceDriven/Test_Oliveira_Parker_120_120_120.h5",
    #"Leopard":      "ForceDriven/Test_Oliveira_Leopard_120_120_120.h5",
    #"Kirby":        "ForceDriven/Test_Oliveira_Kirby_120_120_120.h5",
    #"Castle Gate":  "ForceDriven/Test_Oliveira_CastleGate_120_120_120.h5",
    #"Brown":        "ForceDriven/Test_Oliveira_Brown_120_120_120.h5",
    #"Upper Gray":   "ForceDriven/Test_Oliveira_BereaUpperGray_120_120_120.h5",
    #"Sinter Gray":  "ForceDriven/Test_Oliveira_BereaSinterGray_120_120_120.h5",
    #"Berea Buff":   "ForceDriven/Test_Oliveira_BereaBuff_120_120_120.h5",
    #"Berea":        "ForceDriven/Test_Oliveira_Berea_120_120_120.h5",
    #"Bentheimer":   "ForceDriven/Test_Oliveira_Bentheimer_120_120_120.h5",
    #"Bandera":      "ForceDriven/Test_Oliveira_Bandera_120_120_120.h5",
    }


# The specific sample indices you want to analyze
samples_to_plot = [0, 1, 2] 


#######################################################
#************ INITIALIZE DATASETS         ************#
#######################################################
datasets_data   = {}

print("Initializing datasets...")
for ds_name, ds_path in datasets.items():
    dataset_full_name = dataset_folder + ds_path
    datasets_data[ds_name] = dr.LazyDatasetTorch(
        h5_path=dataset_full_name, 
        list_ids=None, 
        x_dtype=torch.float32,
        y_dtype=torch.float32
    )
#######################################################
#****** SAMPLE-BY-SAMPLE VEL. Field ANALYSIS   *******#
#######################################################
"""
for sample_idx in samples_to_plot:
    print(f"Generating Side-by-Side Plots for Sample {sample_idx}...")
    
    # Compare Uz (vel_channel=0) Front View
    Plot_Velocity_Front_Comparison(
        datasets=datasets_data, 
        sample_idx=sample_idx, 
        slice_idx=60, 
        save_mode=True, 
    )
    
    # Compare Uz (vel_channel=0) Side View
    Plot_Velocity_Side_Comparison(
        datasets=datasets_data, 
        sample_idx=sample_idx, 
        slice_idx=60, 
        save_mode=True, 
    )
"""
#######################################################
#****** SAMPLE-BY-SAMPLE HISTOGRAM ANALYSIS   ********#
#######################################################
"""
Compare_Histograms(
    datasets=datasets_data, 
    sample_indices=samples_to_plot,
    bins=500,
    save_folder="Normalization_comparisons" # Specify your folder here
)
"""

#######################################################
#****** SAMPLE-BY-SAMPLE PERMEABILITY ANALYSIS  ******#
#######################################################
Compare_Permeability(
        datasets_data,
        samples_to_plot,
)

