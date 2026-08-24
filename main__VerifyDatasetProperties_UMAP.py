import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# UMAP package
# Install with: pip install umap-learn
import umap.umap_ as umap

# ==========================================================
# Configuration
# ==========================================================
CSV_FILE = "Dataset_Properties_Table.csv"

FEATURES = [
    "Porosity",
    "Permeability",
    "Q1_Local_Thickness",
    "Mean_Local_Thickness",
    "Max_Local_Thickness",
]

LABEL_COLUMN = "Dataset"

# ----------------------------------------------------------
# FILTER & SHAPES
# ----------------------------------------------------------
# Replace these with the actual names of the datasets you want to ignore completely
#datasets_to_ignore = ["Cylindrical_Grains", "Cylindrical_Pores"]
datasets_to_ignore = []
# Replace these with the actual names of datasets you want plotted as triangles
datasets_as_triangles = ["Cylindrical_Grains", "Cylindrical_Pores", "Spherical_Grains", "Spherical_Pores"]

# ==========================================================
# Read data
# ==========================================================
df = pd.read_csv(CSV_FILE)

# Keep only required columns and drop missing values
df = df[[LABEL_COLUMN] + FEATURES].dropna()

# Keep only the rows where the Dataset name is NOT in the ignore list
df = df[~df[LABEL_COLUMN].isin(datasets_to_ignore)]

# ==========================================================
# Prepare features
# ==========================================================
X = df[FEATURES].values
labels = df[LABEL_COLUMN]

# Standardize before PCA/UMAP
X_scaled = StandardScaler().fit_transform(X)

# ==========================================================
# PCA
# ==========================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ==========================================================
# UMAP
# ==========================================================
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.6,
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

# ==========================================================
# Plot
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Unique datasets (after filtering)
datasets = sorted(labels.unique())

# Color map
cmap = plt.cm.get_cmap("tab20", len(datasets))

# ---------------- PCA ----------------
ax = axes[0]
for i, dataset in enumerate(datasets):
    mask = labels == dataset
    
    # Check if this dataset should be a triangle ('^') or a dot ('o')
    marker_style = '^' if dataset in datasets_as_triangles else 'o'
    
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=60,
        alpha=0.2,           # Lowered to 0.5 to make it more transparent
        color=cmap(i),
        marker=marker_style, # Apply the chosen shape
        label=dataset,
    )

ax.set_title("PCA")
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
ax.grid(True)

# ---------------- UMAP ----------------
ax = axes[1]
for i, dataset in enumerate(datasets):
    mask = labels == dataset
    
    # Check if this dataset should be a triangle ('^') or a dot ('o')
    marker_style = '^' if dataset in datasets_as_triangles else 'o'
    
    ax.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=60,
        alpha=0.2,           # Lowered to 0.5 to make it more transparent
        color=cmap(i),
        marker=marker_style, # Apply the chosen shape
        label=dataset,
    )

ax.set_title("UMAP")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.grid(True)

# ==========================================================
# Legend & Layout Fix
# ==========================================================
# Extract handles and labels from the first axis
handles, labels_ = axes[0].get_legend_handles_labels()

# Attach the legend to the rightmost axis (axes[1]) instead of the figure
axes[1].legend(
    handles,
    labels_,
    title="Dataset",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5), # Places it 5% outside the right edge of the UMAP plot
    frameon=True,
)

# Standard tight_layout handles the rest automatically
plt.tight_layout()
plt.show()