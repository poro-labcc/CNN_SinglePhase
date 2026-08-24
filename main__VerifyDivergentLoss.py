import torch
from   torch.utils.data import DataLoader
import torch.nn as nn
from Utilities import dataset_reader as dr
from Utilities import loss_functions as lf
from Danny_Original.architecture import Danny_KerasModel
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


#datapath   = "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5" 
datapath    = "../NN_Datasets/PressureDriven/Train_Danny_120_120_120_Pressure.h5"
batch_size = 1

baseline_model = Danny_KerasModel()


dataset = dr.LazyDatasetTorch(
    h5_path=datapath, 
    list_ids=None, 
    x_dtype=torch.float32,
    y_dtype=torch.float32
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

fn   = lf.Divergent_2()
fn_2 = lf.Divergent()
fn_3 = lf.MassConservation()
fn_4 = lf.NavierStokesLoss()

with torch.no_grad():
    for batch_idx, (inp, tar) in enumerate(loader):
        
        inp = inp.to(dtype=torch.float32)
        tar = tar.to(dtype=torch.float32)
        out = baseline_model.predict(inp)
        out = torch.concatenate([out, tar[:,3:4,:,:,:]],dim=1)
    

        # Gradient using torch.gradient() channel by channel
        tar_div  = torch.gradient(tar, dim=2)[0][:, 0]
        tar_div += torch.gradient(tar, dim=3)[0][:, 1]
        tar_div += torch.gradient(tar, dim=4)[0][:, 2]
        
        # Usa .item() para extrair o valor escalar do tensor e formata com 16 casas decimais
        print(" TARGET: ")
        print(f"Pytorch gradient() [no-walls]:  {tar_div.abs().mean().item():.16f}")        
        print(f"Divergent          [no-walls]:  {fn(tar, tar).item():.16f}")
        print(f"Divergent_2:                   {fn_2(tar, tar).item():.16f}")
        print(f"Mass Conservation:             {fn_3(tar, inp).item():.16f}")
        print(f"Navier Stokes:                 {fn_4(tar, inp).item():.16f}")
        print(" OUTPUT: ")
        print(f"Divergent          [no-walls]:  {fn(out, tar).item():.16f}")
        print(f"Divergent_2:                   {fn_2(out, tar).item():.16f}")
        print(f"Mass Conservation:             {fn_3(out, inp).item():.16f}")
        print(f"Navier Stokes:                 {fn_4(out, inp).item():.16f}")
 
        
# Verifying mask scheme used to derivate
shape = (1, 1, 30, 30, 30)
bin_solid = torch.ones(shape)

bin_solid[:, :, 0, :, :] = 0  
bin_solid[:, :, -1, :, :] = 0 
bin_solid[:, :, :, 0, :] = 0  
bin_solid[:, :, :, -1, :] = 0 
bin_solid[:, :, :, :, 0] = 0  
bin_solid[:, :, :, :, -1] = 0 
bin_solid[:, :, 13:18, 13:18, 13:18] = 0


mag_z_right = bin_solid[:, 0, 2:  , 1:-1, 1:-1] > 1e-16
mag_z_left  = bin_solid[:, 0, :-2 , 1:-1, 1:-1] > 1e-16
mag_cent    = bin_solid[:, 0, 1:-1, 1:-1, 1:-1] > 1e-16

central     = ( mag_cent & mag_z_right  &  mag_z_left)  
lat_forw    = ( mag_cent & mag_z_right  & ~mag_z_left) 
lat_back    = ( mag_cent & ~mag_z_right  &  mag_z_left)

slice_idx = 14 

central_slc     = central[0, :, slice_idx, :].float().cpu().numpy()
lat_forw_slc    = lat_forw[0, :, slice_idx, :].float().cpu().numpy()
lat_back_slc    = lat_back[0, :, slice_idx, :].float().cpu().numpy()

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

# Plot Left
bin_slice = bin_solid[0,0,:, slice_idx, :]
im0 = axes[0].imshow(bin_slice, cmap='gray', origin='lower')
axes[0].set_title("Binazry Image")
axes[0].set_ylabel("Z")
axes[0].set_xlabel("X")
axes[0].set_xticks(range(bin_slice.shape[1])) 
axes[0].set_yticks(range(bin_slice.shape[0])) 
axes[0].tick_params(labelsize=6)

bin_slice_red = bin_solid[0,0,1:-1, slice_idx, 1:-1]
im0 = axes[1].imshow(bin_slice_red, cmap='gray', origin='lower')
axes[1].set_title("Binazry Image")
axes[1].set_ylabel("Z")
axes[1].set_xlabel("X")
axes[1].set_xticks(range(bin_slice_red.shape[1])) 
axes[1].set_yticks(range(bin_slice_red.shape[0])) 
axes[1].tick_params(labelsize=6)

custom_cmap = ListedColormap(['#ff6666', '#66cc66'])
im0 = axes[2].imshow(central_slc, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
axes[2].set_title("Where to apply Central Diff")
axes[2].set_ylabel("Z")
axes[2].set_xlabel("X")
axes[2].set_xticks(range(central_slc.shape[1])) # Eixo X: do 0 até a largura da imagem
axes[2].set_yticks(range(central_slc.shape[0])) # Eixo Y: do 0 até a altura da imagem
axes[2].tick_params(labelsize=6)

# Plot Center
im1 = axes[3].imshow(lat_forw_slc, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
axes[3].set_title("Where to apply Central Z+ Diff")
axes[3].set_xlabel("X")
axes[3].set_xticks(range(lat_forw_slc.shape[1])) # Eixo X: do 0 até a largura da imagem
axes[3].set_yticks(range(lat_forw_slc.shape[0])) # Eixo Y: do 0 até a altura da imagem
axes[3].tick_params(labelsize=6)

# Plot Right
im2 = axes[4].imshow(lat_back_slc, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
axes[4].set_title("Where to apply Central Z- Diff")
axes[4].set_xlabel("X")
axes[4].set_xticks(range(lat_back_slc.shape[1])) # Eixo X: do 0 até a largura da imagem
axes[4].set_yticks(range(lat_back_slc.shape[0])) # Eixo Y: do 0 até a altura da imagem
axes[4].tick_params(labelsize=6)
    
plt.suptitle(f"Cells to apply Z-Axis Derivative")
plt.tight_layout()
plt.show()