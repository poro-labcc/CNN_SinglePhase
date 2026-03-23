import torch
import numpy as np
from pathlib import Path
import imageio.v2 as imageio  # stable v2 API
# Assuming these are available in your env
from Utilities import dataset_reader as dr
from Architectures.Models import MS_Net,Corrected_MS_Net, DannyKo_Net_Original
import os
import matplotlib.colors as mcolors
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
from glob import glob
from Utilities            import nn_trainner as nnt
from PIL import Image, ImageDraw, ImageFont
from Utilities import dataset_reader as dr


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']

def Set_solids_to_value(array, bin_array, value=0, solid_value=0):
    arr = array.copy()
    arr[bin_array==solid_value] = value
    return arr

def Plot_Continuous_Domain_2D(
    values,
    filename,
    title="",
    remove_value=None,              # values to make transparent
    colormap="viridis",
    vmin=None,
    vmax=None,
    clip_percentiles=None,          # e.g. (2, 98) for robust scaling
    show_colorbar=True,
    special_colors=None,            # dict: {value: color}
    dpi=300
):
    """
    Save a PNG heatmap of a 2D continuous field, with optional special colors.

    Parameters
    ----------
    values : np.ndarray
        2D array (H, W) or (1, H, W) with continuous values.
    filename : str
        Path (without extension) to save the image.
    remove_value : float or list/tuple[float], optional
        Values to be masked (transparent).
    colormap : str
        Matplotlib colormap name for continuous values.
    vmin, vmax : float, optional
        Explicit color limits. If None, computed from data (or percentiles).
    clip_percentiles : tuple, optional
        If vmin/vmax not given, use percentiles of data.
    show_colorbar : bool
        Whether to show colorbar.
    special_colors : dict, optional
        Mapping {value: matplotlib_color} for specific discrete values.
    dpi : int
        Output resolution.
    """
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # If input is 3D (1, H, W), take first slice
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"`values` must be 2D or (1,H,W). Got {values.shape}")

    # Build mask: NaNs + remove_value(s)
    mask = np.isnan(values)
    if remove_value is not None:
        if np.isscalar(remove_value):
            mask |= (values == remove_value)
        else:
            mask |= np.isin(values, list(remove_value))

    # Prepare data for plotting
    data = np.ma.masked_array(values, mask=mask)

    # Determine vmin/vmax if not given
    finite_vals = data.compressed()
    if finite_vals.size == 0:
        raise ValueError("All values masked or NaN; nothing to plot.")
    if (vmin is None or vmax is None):
        if clip_percentiles is not None:
            lowp, highp = clip_percentiles
            vmin_auto, vmax_auto = np.percentile(finite_vals, [lowp, highp])
        else:
            vmin_auto, vmax_auto = np.min(finite_vals), np.max(finite_vals)
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
        if vmin == vmax:
            vmin, vmax = vmin - 1e-8, vmax + 1e-8

    # Base colormap
    cmap = mpl.colormaps[colormap].copy()
    if hasattr(cmap, "set_bad"):
        cmap.set_bad((0, 0, 0, 0))  # transparent for masked

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

    # Overlay special colors
    if special_colors:
        for val, color in special_colors.items():
            mask_special = (values == val)
            if np.any(mask_special):
                overlay = np.zeros((*values.shape, 4))
                overlay[mask_special] = mcolors.to_rgba(color)
                ax.imshow(overlay, interpolation="none")

    # Aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize="large")

    fig.tight_layout()
    out_path = f"{filename}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path



def get_files_dict(directory):
    files_dict = {}
    
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        
        if os.path.isfile(full_path):
            match = re.search(r'(\d+)$', f)
            if match:
                number = int(match.group(1))
                files_dict[number] = f
    
    # Return dictionary sorted by key
    return dict(sorted(files_dict.items()))

def mean_normalize(inp, x): 
    B, C, Z, Y, Xdim = x.shape
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

def get_masked_slices(inp, tar, out, slice_idx, axis='front'):
    """Extracts and masks 2D slices from 3D volumes based on orientation."""
    if axis == 'front':
        # XY Plane (slice along Z)
        i_slc = inp[0, 0, slice_idx].cpu().numpy()
        t_slc = tar[0, 0, slice_idx].cpu().numpy()
        o_slc = out[0, 0, slice_idx].cpu().numpy()
    elif axis == 'side':
        # XZ Plane (slice along Y)
        i_slc = inp[0, 0, :, slice_idx, :].cpu().numpy()
        t_slc = tar[0, 0, :, slice_idx, :].cpu().numpy()
        o_slc = out[0, 0, :, slice_idx, :].cpu().numpy()
    
    mask = (i_slc == 0)
    return np.ma.array(t_slc, mask=mask), np.ma.array(o_slc, mask=mask)
#######################################################
#******************** INPUTS *************************#
#######################################################

NN_DATASETS_DIR     = Path("../NN_Datasets")
DATASET_NAME        = "ForceDriven/Test_CylinPore_120_120_120.h5"
EXAMPLES_SHAPE      = (120, 120, 120)
sample_idx            = 0

# LOAD MODEL
model_base_path = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_15_March_2026_03-30PM_Job16205/"
model_aux       = DannyKo_Net_Original()
model           = model_aux.z_model





#######################################################
#***************** SETUP        **********************#
#######################################################

FRAMES_DIR = Path(model_base_path+"frames/")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

dataset_full_path   = NN_DATASETS_DIR / DATASET_NAME

dataset    = dr.LazyDatasetTorch(h5_path=dataset_full_path, 
                                list_ids=None, 
                                x_dtype=torch.float32,
                                y_dtype=torch.float32)


DEVICE = torch.device("cpu")  # or torch.device("cuda") if desired

#######################################################
#************ EVALUATE MULTIPLE CHECKPOINTS **********#
#######################################################


# get all checkpoints
# sort
files = get_files_dict(model_base_path)
   

print("Computing Ouput from each epoch ...")
for epoch, checkpoint_name in files.items():
    print("Plotting frame from epoch ", epoch)
    model,_ = nnt.load_model_from_checkpoint(model, model_base_path, epoch=epoch, device='cpu')
    model.bin_input = True
    
    net_input, net_target  = dataset[sample_idx]#.to(dtype=torch.float32)
    net_input, net_target = net_input.unsqueeze(0).to(dtype=torch.float32), net_target.unsqueeze(0).to(dtype=torch.float32)
    net_output = model.predict(net_input)
    
    net_target = mean_normalize(net_input, net_target)
    net_output = mean_normalize(net_input, net_output)
    
    t_masked, o_masked = get_masked_slices(net_input, net_target, net_output, slice_idx=EXAMPLES_SHAPE[0]//2, axis='side')

    net_input  = net_input[0,0, :,:,:].numpy()
    net_target = net_target[0,0, :,:,:].numpy()
    net_output = net_output[0,0, :,:,:].numpy()
    
    vmin, vmax = np.percentile(t_masked.compressed(), [1, 99])
    
    dimx, dimy, dimz = net_target.shape
    
    solid_mask                      = net_input==0
    net_input[solid_mask]           = -1
    net_target[solid_mask]          = -1
    net_output[solid_mask]          = -1

    fname_no_ext = FRAMES_DIR / f"frame_{epoch:03d}"  
    Plot_Continuous_Domain_2D(
        values=net_output[:, dimy // 2, :],
        filename=str(fname_no_ext),
        colormap="plasma",
        show_colorbar=True,
        vmax=vmax,
        vmin=vmin,
        special_colors={-1: (1, 1, 1, 1)},
    )


# Build GIF from the frames we actually saved
images = [imageio.imread(p) for p in sorted(FRAMES_DIR.glob("frame_*.png"))]
imageio.mimsave(FRAMES_DIR / "animation.gif", images, duration=0.4, loop=0)
print(f"Saved GIF to: {FRAMES_DIR / 'animation.gif'}")



# Save a static PNG of the *final* net_input (same colormap/limits) ---
left_fname_no_ext = FRAMES_DIR / "static_input_final"
Plot_Continuous_Domain_2D(
    values=net_target[dimz // 2, :, :],
    filename=str(left_fname_no_ext),
    colormap="plasma",
    show_colorbar=True,
    vmax=vmax,  # keep consistent with the right frames
    vmin=vmin,
    special_colors={-1: (1, 1, 1, 1)},
)
left_png = left_fname_no_ext.with_suffix(".png")
left_img        = Image.open(left_png).convert("RGBA")
combined_frames = []
right_paths     = sorted(FRAMES_DIR.glob("frame_*.png"))
for idx,rp in enumerate(right_paths):
    right_img = Image.open(rp).convert("RGBA")
    # Ensure same height (just in case your plotter sample_outputs differ)
    if right_img.height != left_img.height:
        new_w = int(right_img.width * (left_img.height / right_img.height))
        right_img = right_img.resize((new_w, left_img.height), Image.BICUBIC)
    canvas = Image.new("RGBA", (left_img.width + right_img.width, left_img.height), (255, 255, 255, 255))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width, 0))
    
    # ----- Panel Titles -----
    left_title  = "Targets"
    right_title = "Neural Network Output"
    draw = ImageDraw.Draw(canvas)
    try:
        font_size = max(18, left_img.height // 18)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        title_font = ImageFont.load_default()
    
    margin = 20
    # LEFT TITLE
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), left_title, font=title_font)
        tw_left, th_left = x1 - x0, y1 - y0
    else:
        tw_left, th_left = draw.textsize(left_title, font=title_font)
    
    x_left = (left_img.width - tw_left) // 2
    y_left = margin
    draw.rectangle(
        (x_left - 10, y_left - 8, x_left + tw_left + 10, y_left + th_left + 8),
        fill=(0, 0, 0, 160)
    )
    draw.text((x_left, y_left), left_title, font=title_font, fill=(255, 255, 255, 255))
    # RIGHT TITLE
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), right_title, font=title_font)
        tw_right, th_right = x1 - x0, y1 - y0
    else:
        tw_right, th_right = draw.textsize(right_title, font=title_font)
    x_right = left_img.width + (right_img.width - tw_right) // 2
    y_right = margin
    draw.rectangle(
        (x_right - 10, y_right - 8, x_right + tw_right + 10, y_right + th_right + 8),
        fill=(0, 0, 0, 160)
    )
    draw.text((x_right, y_right), right_title, font=title_font, fill=(255, 255, 255, 255))
    
    # ----- Percent label (top-left of right panel) -----
    
    epoch = int(rp.stem.split("_")[-1])
    txt     = f"Epoch: {epoch}"
    draw    = ImageDraw.Draw(canvas)
    try:
        font_size = max(14, left_img.height // 24)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Measure text
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
        tw, th = x1 - x0, y1 - y0
    else:
        tw, th = draw.textsize(txt, font=font)
    pad, margin = 10, 15
    x_text = left_img.width + margin   # inside the right panel
    y_text = margin
    draw.rectangle(
        (x_text - pad, y_text - pad, x_text + tw + pad, y_text + th + pad),
        fill=(0, 0, 0, 160)
    )
    draw.text((x_text, y_text), txt, font=font, fill=(255, 255, 255, 255))    
    combined_frames.append(np.array(canvas))

# Makes videos
out_gif = FRAMES_DIR / "comparison.gif"
imageio.mimsave(out_gif, combined_frames, duration=0.4, loop=0)
print(f"Saved comparison GIF to: {out_gif}")

out_mp4 = FRAMES_DIR / "comparison.mp4"
fps = 1 / 0.8
imageio.mimsave(
    out_mp4,
    combined_frames,
    fps=fps,
    codec="libx264",
    quality=8
)
print(f"Saved comparison video to: {out_mp4}")