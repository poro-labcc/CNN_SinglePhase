import matplotlib.pyplot as plt
import os
import matplotlib as mpl 

def plot_histogram(pores_raidus, throats_radius, save_path=None):
    
    
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",   # if installed
            "Times",
            "Liberation Serif",  # common on Linux
            "Nimbus Roman",
            "DejaVu Serif",      # Matplotlib default-ish
            "serif",             # generic fallback
        ],
    
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    
        "axes.edgecolor": "0.2",
        "xtick.color": "0.2",
        "ytick.color": "0.2",
        "text.color": "0.15",
    
        "axes.grid": False,
    })

    def stats(a):
        return a.min(), a.mean(), a.std(), a.max()
    
    p_min, p_mean, p_std, p_max = stats(pores_raidus)
    t_min, t_mean, t_std, t_max = stats(throats_radius)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, dpi=300)
    
    for ax in axes:
        ax.set_box_aspect(1)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    bbox = dict(boxstyle="round,pad=0.35", facecolor="0.92", edgecolor="0.3", alpha=0.95)
    hist_color = "0.35"  # dark grey
    
    # Pore
    axes[0].hist(pores_raidus, bins="auto", color=hist_color, alpha=0.9)
    axes[0].axvline(p_mean, color="black", linewidth=2.2)
    axes[0].set_title("Pore radius")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.02, 0.98,
        f"min =  {p_min:.6g}\nmean = {p_mean:.6g}\nstd =  {p_std:.6g}\nmax = {p_max:.6g}",
        transform=axes[0].transAxes,
        va="top",
        bbox=bbox
    )
    
    # Throat
    axes[1].hist(throats_radius, bins="auto", color=hist_color, alpha=0.9)
    axes[1].axvline(t_mean, color="black", linewidth=2.2)
    axes[1].set_title("Throat radius")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")
    axes[1].text(
        0.02, 0.98,
        f"min = {t_min:.6g}\nmean = {t_mean:.6g}\nstd = {t_std:.6g}\nmax = {t_max:.6g}",
        transform=axes[1].transAxes,
        va="top",
        bbox=bbox
    )
    
    plt.show()
    
    if save_path is not None:
        save_path = os.path.splitext(save_path)[0] + ".png"
    
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

