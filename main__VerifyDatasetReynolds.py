import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps 
from torch.utils.data import DataLoader
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import distance_transform_edt

# Importe os seus leitores e utilitários
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

def find_local_maxima(dist_transform):
    
    dt = np.asarray(dist_transform)
    
    # Initialize boolean arrays for each direction (padded with False at the borders)
    max_z = np.zeros_like(dt, dtype=bool)  # Axis 0
    max_y = np.zeros_like(dt, dtype=bool)  # Axis 1
    max_x = np.zeros_like(dt, dtype=bool)  # Axis 2
    
    # 1. Check Z direction (Axis 0)
    # Center is strictly greater than the slice before it AND the slice after it
    max_z[1:-1, :, :] = ((dt[1:-1, :, :] > dt[:-2, :, :]) & (dt[1:-1, :, :] > dt[2:, :, :]))
    
    # 2. Check Y direction (Axis 1)
    max_y[:, 1:-1, :] = (dt[:, 1:-1, :] > dt[:, :-2, :]) & (dt[:, 1:-1, :] > dt[:, 2:, :])
    
    # 3. Check X direction (Axis 2)
    max_x[:, :, 1:-1] = (dt[:, :, 1:-1] > dt[:, :, :-2]) & (dt[:, :, 1:-1] > dt[:, :, 2:])
    
    # Count how many axes flag the voxel as a maximum (converts True to 1, False to 0)
    max_count = (max_z & max_y).astype(np.int8) + (max_y & max_x).astype(np.int8) + (max_z & max_x).astype(np.int8)
    
    # Define global maximum: true in at least 2 directions AND must be inside the pore (dt > 0)
    local_maxima_mask = (max_count >= 2) & (dt > 0)
    
    return local_maxima_mask    

"""
def calculate_local_reynols_maximas(porous_mask, vel_mag, mu, dens):
    edt     = distance_transform_edt(porous_mask).astype("float32")
    maximas = find_local_maxima(edt)
    
    thick   = ps.filters.local_thickness(porous_mask)
    
    reynolds = thick*vel_mag*dens/mu
    valid_reynolds = reynolds[maximas]
    
    return valid_reynolds, maximas
"""

def calculate_local_reynols(porous_mask, vel_mag, mu, dens):
    
    thick   = ps.filters.local_thickness(porous_mask)
    
    reynolds = thick*vel_mag*dens/mu

    return reynolds[porous_mask], porous_mask
    
# -------------------------------------------------------------------
# 1) Boxplot with Jitter for Multiple Datasets
# -------------------------------------------------------------------
def plot_reynolds_boxplot(datasets_dict: dict, 
                                   batch_size: int = 4, 
                                   save_path: str = None):
    
    # --- Configuration for Academic Style ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    all_means = []
    all_maxs = []
    all_pcts = []
    labels = []
    
    print("--- Starting Extraction for Academic Plots ---")
    
    for dataset_name, datapath in datasets_dict.items():
        print(f"Processing: {dataset_name}")
        # Note: Replace dr.LazyDatasetTorch and vu.tensor_denorm with your actual imports
        dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, 
                                      x_dtype=torch.float32, y_dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        sample_means, sample_maxs, sample_pcts = [], [], []
        
        with torch.no_grad():
            for batch_inp, batch_tar in loader:
                for b in range(batch_inp.shape[0]):
                    mask_np     = (batch_inp[b, 0] > 0).cpu().numpy()
                    denorm_tar  = vu.tensor_denorm(batch_tar[b:b+1], batch_inp[b:b+1])
                    v_mag       = torch.sqrt(denorm_tar[0,0]**2 + denorm_tar[0,1]**2 + denorm_tar[0,2]**2).cpu().numpy()
                    
                    valid_re, _ = calculate_local_reynols(porous_mask=mask_np, vel_mag=v_mag, mu=1.0/3.0, dens=1.0)
                    
                    if valid_re.size > 0:
                        sample_means.append(valid_re.mean())
                        sample_maxs.append(np.percentile(valid_re, 75))
                        sample_pcts.append(valid_re.max())
        
        if sample_means:
            all_means.append(np.array(sample_means))
            all_maxs.append(np.array(sample_maxs))
            all_pcts.append(np.array(sample_pcts))
            labels.append(dataset_name)

    # Plot Setup: 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
    
    titles = [
        "Mean local Reynolds Number\nper sample", 
        "75th Percentile of local Reynolds\nNumber per sample", # Updated title
        "Maximum local Reynolds Number\nper sample",
    ]
    data_list   = [all_means, all_maxs, all_pcts]
    colors      = ['#1f77b4', '#d62728', '#2ca02c'] 

    formatted_labels = [str(label).replace(" ", "\n") for label in labels]
    
    for i, ax in enumerate(axes):
        # Create boxplot with academic styling
        
        bplot = ax.boxplot(data_list[i], tick_labels=formatted_labels, patch_artist=True, 
                           showfliers=False, widths=0.5, zorder=2)
        
        # Style boxes, medians, and whiskers
        for patch in bplot['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor(colors[i])
            patch.set_linewidth(1.5)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        for element in ['whiskers', 'caps']:
            for line in bplot[element]:
                line.set_color('black')
                line.set_linewidth(1)
        
        for j, data in enumerate(data_list[i]):
            x = np.random.uniform(j + 1 - 0.15, j + 1 + 0.15, size=len(data))
            ax.scatter(x, data, alpha=0.6, facecolors='none', edgecolors='black', 
                       s=30, linewidths=1.0, zorder=3)
            
        ax.set_ylabel(titles[i])
        ax.set_axisbelow(True) 
        
        # Scientific formatter for Reynolds axes
        if i < 2:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(formatter)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Academic figure saved to {save_path}")
    else:
        plt.show()

# ==========================================
# Exemplo de Execução
# ==========================================
 
datasets = {
    "Spherical Pores":   "../NN_Datasets_Grad/Train_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":  "../NN_Datasets_Grad/Train_Silveira_SphGrain_SAug_DNorm.h5",
    "Leopard":           "../NN_Datasets_Grad/Train_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":       "../NN_Datasets_Grad/Train_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":  "../NN_Datasets_Grad/Train_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray": "../NN_Datasets_Grad/Train_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea Buff":        "../NN_Datasets_Grad/Train_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Berea":             "../NN_Datasets_Grad/Train_Oliveira_Berea_SAug_DNorm.h5",
    "Bentheimer":        "../NN_Datasets_Grad/Train_Oliveira_Bentheimer_SAug_DNorm.h5",
}

plot_reynolds_boxplot(datasets_dict=datasets, 
                      batch_size=12, 
                      save_path="Local_Reynolds.png")
