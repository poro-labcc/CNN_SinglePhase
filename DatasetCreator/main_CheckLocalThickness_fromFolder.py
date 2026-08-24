import os
import glob
import csv
import numpy as np
import porespy as ps
from utils import plot_lt_distribution, check_local_thickness
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
#BASE_DIRECTORIES    = ["../../Simulations/Train_Danny_SphPore_120_120_120/"]
#BASE_DIRECTORIES    = ["../../Simulations/Test_CylinPore_120_120_120/"]

BASE_DIRECTORIES    = ["../../Simulations/Test_CylinGrain_120_120_120/"]

#BASE_DIRECTORIES    = ["/home/gabriel/remote/hal/dissertacao/simulations/Test_SphPore_120_120_120/Samples/"]

RAW_FILENAME        = "domain.raw"
VOL_SHAPE           = (120, 120, 120)
VOL_DTYPE           = np.uint8

# Verification Limits
MIN_R           = 5.0
MAX_R           = 17.0
TARGET_PERCENT  = 70.0


for base_dir in BASE_DIRECTORIES:
    print(f"Processing Directory: {base_dir}")
    raw_files = glob.glob(os.path.join(base_dir, "**", RAW_FILENAME), recursive=True)
    if not raw_files: raise Exception("Dataset not found in ", base_dir)

    dataset_summary = []
    all_fluid_pixels = []
    
    for file_idx, raw_path in enumerate(raw_files):
        parent_folder = os.path.basename(os.path.dirname(raw_path))
        print(f"File {file_idx}: {parent_folder}")
        
        # Load raw volume
        vol = np.fromfile(raw_path, dtype=VOL_DTYPE).reshape(VOL_SHAPE)
        
        # Explore plots
        #plot_lt_distribution(vol, MIN_R, MAX_R, TARGET_PERCENT)
        
        # Explore defined threshold
        success = check_local_thickness(
            im=vol, 
            min_radius=MIN_R, 
            max_radius=MAX_R, 
            target_percentage=TARGET_PERCENT
        )
        
        # Collect statistics
        lt                  = ps.filters.local_thickness(vol)
        # Where fluid cells are located
        fluid_pixels        = lt[lt > 0]    
        mean_thickness      = np.mean(fluid_pixels)
        std_thickness       = np.std(fluid_pixels)
        
        dataset_summary.append({
            "file_idx": file_idx,
            "folder": parent_folder,
            "path": raw_path,
            "resolved": success,
            "mean_thickness": mean_thickness,
            "std_thickness": std_thickness
        })
        
        all_fluid_pixels.append(fluid_pixels)
        

    # -----------------------------
    # Save CSV inside dataset folder
    # -----------------------------
    output_csv = os.path.join(base_dir, "local_thickness_summary.csv")

    if dataset_summary:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dataset_summary[0].keys())
            writer.writeheader()
            writer.writerows(dataset_summary)
            
    # -----------------------------
    # Generate and Save Global Histogram
    # -----------------------------
    if all_fluid_pixels:
        print("Generating global histogram...")
        # Concatenate all 1D arrays into a single massive 1D array
        combined_pixels = np.concatenate(all_fluid_pixels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(combined_pixels, bins=50, color='dodgerblue', edgecolor='black', alpha=0.7)
        
        # Formatting
        dataset_name = os.path.basename(os.path.normpath(base_dir))
        ax.set_title(f"Global Local Thickness Distribution\n{dataset_name}", fontsize=14)
        ax.set_xlabel("Local Thickness (voxels)", fontsize=12)
        ax.set_ylabel("Voxel Count", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add vertical lines to show your verification limits
        ax.axvline(MIN_R, color='red', linestyle='dashed', linewidth=2, label=f'Min R ({MIN_R})')
        ax.axvline(MAX_R, color='red', linestyle='dashed', linewidth=2, label=f'Max R ({MAX_R})')
        ax.legend()
        
        # Save the plot
        hist_path = os.path.join(base_dir, "global_thickness_histogram.png")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=300)
        plt.close(fig)
        
        print(f"Saved Global Histogram: {hist_path}")

    print(f"Saved CSV: {output_csv}")

print("\nBatch processing finished.")