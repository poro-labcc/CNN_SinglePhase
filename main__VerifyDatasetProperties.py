import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import porespy as ps
from torch.utils.data import DataLoader

# Import local utilities
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

def analyze_and_plot_properties(datasets_dict: dict, batch_size: int = 4, save_csv_path: str = None, save_plot_path: str = None):
    """
    Extracts Porosity, Permeability, Q1/Mean/Max Local Thickness per sample,
    generates a tabular summary, and plots jittered boxplots for each property.
    """
    
    # ==========================================
    # 1. DATA EXTRACTION (Create Per-Sample Table)
    # ==========================================
    records = []
    
    print("--- Starting Extraction ---")
    for dataset_name, datapath in datasets_dict.items():
        print(f"Processing Dataset: {dataset_name}")
        
        # Load Dataset
        dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, 
                                      x_dtype=torch.float32, y_dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_inp, batch_tar in loader:
                B = batch_inp.shape[0]
                
                for b in range(B):
                    # Extract porous mask
                    mask_np = (batch_inp[b, 0] > 0).cpu().numpy()
                    
                    # Calculate Porosity
                    porosity = np.mean(mask_np)
                    
                    # Calculate Permeability
                    denorm_tar = vu.tensor_denorm(batch_tar[b:b+1], batch_inp[b:b+1])
                    perm = vu.permeability_calculation(denorm_tar, 
                                                       batch_inp[b:b+1], 
                                                       tau=1.5,   
                                                       Re=0.1, 
                                                       dens=1.0,
                                                       denorm=False).item()
                    
                    # Calculate Local Thickness
                    thick = ps.filters.local_thickness(mask_np)
                    valid_thick = thick[mask_np]
                    
                    if valid_thick.size > 0:
                        q1_thick   = np.percentile(valid_thick, 25)
                        mean_thick = valid_thick.mean()
                        max_thick  = valid_thick.max()
                    else:
                        q1_thick = mean_thick = max_thick = np.nan
                        
                    # Append sample data
                    records.append({
                        "Dataset": dataset_name,
                        "Porosity": porosity,
                        "Permeability": perm,
                        "Q1 Local Thickness": q1_thick,
                        "Mean Local Thickness": mean_thick,
                        "Max Local Thickness": max_thick
                    })

    # Create the complete per-sample table
    df_samples = pd.DataFrame(records)    
    df_samples_export = df_samples.copy()
    df_samples_export.columns = [
        c.replace(" ", "_") for c in df_samples_export.columns
    ]
    df_samples_export["Dataset"] = (
        df_samples_export["Dataset"]
        .str.replace(" ", "_", regex=False)
    )
    if save_csv_path:
        df_samples_export.to_csv(save_csv_path, index=False)
        print(f"\nPer-sample table saved to {save_csv_path}")



    print("\n--- Summary Table (Mean per Dataset) ---")
    df_summary = df_samples.groupby("Dataset", as_index=False).mean()
    df_summary_export = df_summary.copy()
    df_summary_export.columns = [
        c.replace(" ", "_") for c in df_summary_export.columns
    ]
    df_summary_export["Dataset"] = (
        df_summary_export["Dataset"]
        .str.replace(" ", "_", regex=False)
    )
    
    # Prints
    try:
        # Prints a nice grid (e.g., | Dataset | Porosity | ...)
        print(df_summary.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
        # Fallback if 'tabulate' library isn't installed: Left-justify with generous spacing
        print(df_summary.to_string(index=False, justify='left', col_space=18))
    
    # Automatically save the summary table to its own CSV to avoid copy-paste formatting issues
    if save_csv_path:
        summary_csv = save_csv_path.replace(".csv", "_Summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSummary table saved to {summary_csv} (Open this file in Excel!)")
        
    print("-" * 40)

    # ==========================================
    # 2. BOXPLOT GENERATION (Academic Style)
    # ==========================================
    print("\nGenerating Academic Boxplots...")
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 11,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    variables = [
        "Porosity", 
        "Permeability", 
        "Q1 Local Thickness", 
        "Mean Local Thickness", 
        "Max Local Thickness"
    ]
    
    datasets_labels = df_samples["Dataset"].unique()
    
    # NEW: Format labels to jump lines by replacing spaces with newline characters
    formatted_labels = [str(label).replace(" ", "\n") for label in datasets_labels]
    
    colors = plt.cm.tab10.colors  
    
    # Setup 5 rows, 1 column
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 5 * len(variables)), constrained_layout=True)
    
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Group data per dataset for the boxplot list format
        data_list = [df_samples[df_samples["Dataset"] == ds][var].dropna().values for ds in datasets_labels]
        
        # Create base boxplot using the newly formatted multi-line labels
        bplot = ax.boxplot(data_list, tick_labels=formatted_labels, patch_artist=True, 
                           showfliers=False, widths=0.5, zorder=2)
        
        # Style boxes
        for j, patch in enumerate(bplot['boxes']):
            patch.set_facecolor('white')
            patch.set_edgecolor(colors[j % len(colors)])
            patch.set_linewidth(1.5)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        for element in ['whiskers', 'caps']:
            for line in bplot[element]:
                line.set_color('black')
                line.set_linewidth(1)
        
        # Add Jittered Scatter Dots
        for j, data in enumerate(data_list):
            x = np.random.uniform(j + 1 - 0.15, j + 1 + 0.15, size=len(data))
            ax.scatter(x, data, alpha=0.6, facecolors='none', edgecolors='black', 
                       s=30, linewidths=1.0, zorder=3)
            
        ax.set_ylabel(var)
        ax.set_axisbelow(True) 
        
        # Apply Logarithmic scale for Permeability
        if var == "Permeability":
            ax.set_yscale('log')

    # Save or Show
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        print(f"Academic figure saved to {save_plot_path}")
    else:
        plt.show()

    
datasets = {
    "Spherical Pores":   "../NN_Datasets_Grad/Train_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":  "../NN_Datasets_Grad/Train_Silveira_SphGrain_SAug_DNorm.h5",
    "Cylindrical Pores": "../NN_Datasets_Grad/Train_Silveira_CylinPore_SAug_DNorm.h5",
    "Cylindrical Grains":"../NN_Datasets_Grad/Train_Silveira_CylinGrain_SAug_DNorm.h5",
    "Leopard":           "../NN_Datasets_Grad/Train_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":       "../NN_Datasets_Grad/Train_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":  "../NN_Datasets_Grad/Train_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray": "../NN_Datasets_Grad/Train_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea Buff":        "../NN_Datasets_Grad/Train_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Berea":             "../NN_Datasets_Grad/Train_Oliveira_Berea_SAug_DNorm.h5",
    "Bentheimer":        "../NN_Datasets_Grad/Train_Oliveira_Bentheimer_SAug_DNorm.h5",
}

analyze_and_plot_properties(
    datasets_dict=datasets, 
    batch_size=8,
    save_csv_path="Dataset_Properties_Table.csv",
    save_plot_path="Dataset_Properties_Boxplots.png"
)