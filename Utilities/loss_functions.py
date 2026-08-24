import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.nn import BCELoss, functional
import torch
import numpy as np
import matplotlib.pyplot as plt
from Utilities import velocity_usage as vu
#######################################################
#************ LOSS FUNCTION UTILITIES ****************#
#######################################################

# Apply threshold
class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (torch.sigmoid(x) > self.threshold).float()
    
# Defines the specific pixels that the loss function might look into
# Mode: 
# - flatten if a mask operates and a flatten tensor is propagated (default)
# - overwrite if output is overwrited by the target in the mask locations, then the structure is kept
class Mask_LossFunction(nn.Module):
    def __init__(self, lossFunction, mask_law=None, mode="flatten"):
        super(Mask_LossFunction, self).__init__()
        
        self.lossFunction = lossFunction
        
        self.mode = mode
        
        if mask_law is None: 
            self.mask_law = self._default_mask_law
        else:
            self.mask_law = mask_law
            
    # Do not consider cells with 0 value  
    # The loss function used must be a mean across the tensor lenght, 
    # so that the quantity of solid cells do not affect the loss
    def _default_mask_law(self,output, target, threshold=0): 
        # Mask consider only target != 0, i.e, non-solid cells
        #mask = (target > threshold) | (target < -threshold)
        mask = torch.abs(target) > threshold
        return mask
    
    def forward(self, output, target):
        
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        if self.mode == 'flatten':
            loss    = 0.0
            C       = output.shape[1]
            for c in range(C):
                mask_c   = self.mask_law(output[:, c], target[:, c])
                loss    += self.lossFunction(output[:, c][mask_c], target[:, c][mask_c])
            return loss / C
                
        elif self.mode == 'overwrite':
            mask = self.mask_law(output, target)
            temp_output = output.clone()
            temp_output[~mask] = target[~mask]
            output = temp_output            
            return self.lossFunction(output, target)
        
        else: raise Exception(f"Mask_LossFunction mode {self.mode} not implemented. Must be one of flatten or overwrite")

        
    
    
    
    
#######################################################
#************ LOSS FUNCTIONS  ************************#
#######################################################

# MY COMPOSED FUNCTIONS
    
# Permeability Relative Percentual Error
class MeanBiasError(nn.Module):
    def __init__(self):
        super(MeanBiasError, self).__init__()
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        mean_error = 100*( (output.mean() - target.mean())/target.mean() ).abs()
        #print(f"Means: Output={output.mean().item()}; Target={target.mean().item()}")
        return mean_error
    

class PearsonCorr(nn.Module):
    def __init__(self, N_samples=None, eps=1e-16, reverse=False):
        super(PearsonCorr, self).__init__()
        self.eps=eps
        self.N_samples=N_samples 
        self.reverse = reverse
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        # 1. Flatten the tensors to treat all cells as a single distribution
        o = output.flatten()
        t = target.flatten()
        
        if self.N_samples is not None:         
            indx = torch.randperm(o.size(0))[:self.N_samples]
            o = o[indx]
            t = t[indx]
        
        # 2. Centering (Subtract the mean)
        o_mu = o - o.mean()
        t_mu = t - t.mean()
        
        numerator   = torch.sum(t_mu * o_mu)
        denominator = torch.sqrt(torch.sum(t_mu**2)) * torch.sqrt(torch.sum(o_mu**2))
        corr = (numerator / (denominator + self.eps))
        
        return corr if not self.reverse else 1-corr
    

    
class KGE(nn.Module):
    def __init__(self):
        super(KGE, self).__init__()
        
        self.corr = PearsonCorr(2000)
      
    def forward(self, output, target):
        mean_pred = torch.mean(output)
        mean_true = torch.mean(target)
        bias      = 1.0 - (mean_pred / mean_true)
        inv_corr  = 1.0 - self.corr(output, target)
        
        return torch.sqrt(bias**2 + inv_corr**2)
    
    
class Divergent_2(nn.Module):
    def __init__(self):
        super(Divergent_2, self).__init__()
        
    def forward(self, output, target):
        if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[2:]:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        if output.shape[1] < 3 or target.shape[1] < 3:
            raise ValueError(f"Shape mismatch: output {output.shape} or target {target.shape} need to have at least 3 channels (z,y,x)")
    
        
        mag     = (target[:, 0:1]**2 + target[:, 1:2]**2 + target[:, 2:3]**2)[:, 0, 1:-1, 1:-1, 1:-1 ]
        
        mag_cent= mag > 1e-16
        
        out_div =  mag_cent * (output[:, 0, 2:  , 1:-1, 1:-1] - output[:, 0,  :-2, 1:-1, 1:-1]) / 2.0
        out_div += mag_cent * (output[:, 1, 1:-1, 2:  , 1:-1] - output[:, 1, 1:-1,  :-2, 1:-1]) / 2.0
        out_div += mag_cent * (output[:, 2, 1:-1, 1:-1, 2:  ] - output[:, 2, 1:-1, 1:-1,  :-2]) / 2.0
        
        """
        slice_idx   = 60
        
        rho_val     = target[0, 3, :, :, slice_idx].cpu().numpy()
        error_val   = out_div[0, :, :, slice_idx].cpu().numpy() 

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = ax[0].imshow(mag[0,:,:, slice_idx].cpu().numpy(), cmap='jet')
        ax[0].set_title("Velocity Magnitude")
        plt.colorbar(im0, ax=ax[0])
        
        im1 = ax[1].imshow(rho_val, cmap='viridis')
        ax[1].set_title("Density ($\\rho$)")
        plt.colorbar(im1, ax=ax[1])
        
        # Colormap divergente para o erro (seismic: azul é negativo, vermelho positivo)
        v_limit = np.max(np.abs(error_val)) * 0.5 # Ajuste de contraste
        im2 = ax[2].imshow(error_val, cmap='seismic', vmin=-v_limit, vmax=v_limit)
        ax[2].set_title("$\\nabla \cdot \mathbf{u}$")
        plt.colorbar(im2, ax=ax[2])
        
        plt.suptitle(f"Physical Consistency Check - Slice {slice_idx}")
        plt.tight_layout()
        plt.show()
        """
        return out_div[mag>1e-16].abs().mean()
    




class Divergent(nn.Module):
    def __init__(self):
        super(Divergent, self).__init__()
        
    def forward(self, output, target):
        if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[2:]:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        if output.shape[1] < 3 or target.shape[1] < 3:
            raise ValueError(f"Shape mismatch: output {output.shape} or target {target.shape} need to have at least 3 channels (z,y,x)")
        
        mag     = target[:, 0:1]**2 + target[:, 1:2]**2 + target[:, 2:3]**2
        
        out_div = vu.d_dz(output, mag, c=0) + vu.d_dy(output, mag, c=1) + vu.d_dx(output, mag, c=2)
        
        
        """
        # Plot slice
        slice_idx   = 60
        v_mag       = torch.sqrt(target[0,0]**2 + target[0,1]**2 + target[0,2]**2)[1:-1, 1:-1, slice_idx].cpu().numpy()
        rho_val     = target [0, 3, :, :, slice_idx].detach().cpu().numpy()
        error_val   = out_div[0,    :, :, slice_idx].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = ax[0].imshow(v_mag, cmap='jet')
        ax[0].set_title("Velocity Magnitude")
        plt.colorbar(im0, ax=ax[0])
        
        im1 = ax[1].imshow(rho_val, cmap='viridis')
        ax[1].set_title("Density ($\\rho$)")
        plt.colorbar(im1, ax=ax[1])
        
        v_limit = np.max(np.abs(error_val)) * 0.5 # Ajuste de contraste
        im2 = ax[2].imshow(error_val, cmap='seismic', vmin=-v_limit, vmax=v_limit)
        ax[2].set_title("$\\nabla \cdot \mathbf{u}$ in void space")
        plt.colorbar(im2, ax=ax[2])
        
        plt.suptitle(f"Physical Consistency Check - Slice {slice_idx}")
        plt.tight_layout()
        plt.show()
        """
        
        return out_div[mag[:,0, 1:-1, 1:-1, 1:-1]>1e-16].abs().mean()

    
class MassConservation(nn.Module):
    def _denorm(self, out, inp): return out
        
    def __init__(self, fun_denorm=None):
        super(MassConservation, self).__init__()
        self.fun_denorm = fun_denorm if fun_denorm is not None else self._denorm

    def forward(self, output, inp):
        
        field = self.fun_denorm(output.clone(), inp)
        
        # Pressure to Density logic
        field[:, 3] *= 3.0 
        
        # 1. Divergence of velocity:  (dUz/dz + dUy/dy + dUx/dx) * rho
        mass_cons  = vu.d_dz(field, inp, c=0) + vu.d_dy(field, inp, c=1) + vu.d_dx(field, inp, c=2)
        mass_cons *= field[:,3, 1:-1, 1:-1, 1:-1]
        
        # 2. Advection of density: 
        mass_cons += field[:,0, 1:-1, 1:-1, 1:-1] * vu.d_dz(field, inp, c=3) # Uz * dP/dz
        mass_cons += field[:,1, 1:-1, 1:-1, 1:-1] * vu.d_dy(field, inp, c=3) # Uy * dP/dy
        mass_cons += field[:,2, 1:-1, 1:-1, 1:-1] * vu.d_dx(field, inp, c=3) # Ux * dP/dx
        
        
        """
        # Plot slice
        slice_idx   = 60
        mag         = torch.sqrt(field[0,0]**2 + field[0,1]**2 + field[0,2]**2)[1:-1, 1:-1, slice_idx].cpu().numpy()
        rho_val     = field[0, 3, :, :, slice_idx].cpu().numpy()
        error_val   = mass_cons[0, :, :, slice_idx].cpu().numpy() # slice_idx-1 porque mass_cons é menor

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = ax[0].imshow(mag, cmap='jet')
        ax[0].set_title("Velocity Magnitude")
        plt.colorbar(im0, ax=ax[0])
        
        im1 = ax[1].imshow(rho_val, cmap='viridis')
        ax[1].set_title("Density ($\\rho$)")
        plt.colorbar(im1, ax=ax[1])
        
        v_limit = np.max(np.abs(error_val)) * 0.5 # Ajuste de contraste
        im2 = ax[2].imshow(error_val, cmap='seismic', vmin=-v_limit, vmax=v_limit)
        ax[2].set_title("$\\rho(\\nabla \\cdot \\mathbf{u}) + \\mathbf{u} \\cdot \\nabla \\rho$")
        plt.colorbar(im2, ax=ax[2])
        
        plt.suptitle(f"Physical Consistency Check - Slice {slice_idx}")
        plt.tight_layout()
        plt.show()
        """
            
        return mass_cons[inp[:, 0,  1:-1, 1:-1, 1:-1]>1e-16].abs().mean()
        
        
    

class NavierStokesLoss(nn.Module):
    def _denorm(self, out, inp): return out
        
    def __init__(self, fun_denorm=None, rho=1.0, mu=0.01):
        super(NavierStokesLoss, self).__init__()
        self.fun_denorm = fun_denorm if fun_denorm is not None else self._denorm
        self.rho = rho
        self.mu  = mu

    def forward(self, output, inp):

        field = self.fun_denorm(output.clone(), inp)

        # Extraindo as velocidades apenas no miolo fluído (B, Z-2, Y-2, X-2)
        # Note que passamos '0', '1' e '2' explicitamente para o canal, mantendo as 4 dimensões restantes
        uz_c = field[:, 0, 1:-1, 1:-1, 1:-1]
        uy_c = field[:, 1, 1:-1, 1:-1, 1:-1]
        ux_c = field[:, 2, 1:-1, 1:-1, 1:-1]
        
        # Z-Momentum
        loss_mom_z  = self.rho * (uz_c * vu.d_dz(field, inp, c=0) + uy_c * vu.d_dy(field, inp, c=0) + ux_c * vu.d_dx(field, inp, c=0))
        loss_mom_z += vu.d_dz(field, inp, c=3) # dP/dz 
        loss_mom_z -= self.mu * (vu.d2_dx2(field, inp, c=0) + vu.d2_dy2(field, inp, c=0) + vu.d2_dz2(field, inp, c=0))
                     
        # Y-Momentum
        loss_mom_y  = self.rho * (uz_c * vu.d_dz(field, inp, c=1) + uy_c * vu.d_dy(field, inp, c=1) + ux_c * vu.d_dx(field, inp, c=1)) 
        loss_mom_y += vu.d_dy(field, inp, c=3) # dP/dy
        loss_mom_y -= self.mu * ( vu.d2_dx2(field, inp, c=1) + vu.d2_dy2(field, inp, c=1) + vu.d2_dz2(field, inp, c=1) )
                     
        # X-Momentum
        loss_mom_x  = self.rho * (uz_c * vu.d_dz(field, inp, c=2) + uy_c * vu.d_dy(field, inp, c=2) + ux_c * vu.d_dx(field, inp, c=2))
        loss_mom_x += vu.d_dx(field, inp, c=3) # dP/dx
        loss_mom_x -= self.mu * ( vu.d2_dx2(field, inp, c=2) + vu.d2_dy2(field, inp, c=2) + vu.d2_dz2(field, inp, c=2) )

        total_loss = torch.mean(loss_mom_z**2) + torch.mean(loss_mom_y**2) + torch.mean(loss_mom_x**2)
        
        """
        # Plot slice
        slice_idx   = 60
        v_mag       = torch.sqrt(field[0,0]**2 + field[0,1]**2 + field[0,2]**2)[1:-1, 1:-1, slice_idx].detach().cpu().numpy()
        rho_val     = field[0, 3, 1:-1, 1:-1, slice_idx].detach().cpu().numpy()
        error_val   = loss_mom_z[0, :, :, slice_idx - 1].detach().cpu().numpy() 

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = ax[0].imshow(v_mag, cmap='jet')
        ax[0].set_title("Velocity Magnitude")
        plt.colorbar(im0, ax=ax[0])
        
        im1 = ax[1].imshow(rho_val, cmap='viridis')
        ax[1].set_title("Density ($\\rho$)")
        plt.colorbar(im1, ax=ax[1])
        
        v_limit = np.max(np.abs(error_val)) * 0.5 
        im2 = ax[2].imshow(error_val, cmap='seismic', vmin=-v_limit, vmax=v_limit)
        ax[2].set_title("$\\rho (\\mathbf{u} \\cdot \\nabla) u_z + \\frac{\\partial P}{\\partial z} - \\mu \\nabla^2 u_z$")
        plt.colorbar(im2, ax=ax[2])
        
        plt.suptitle(f"Physical Consistency Check - Slice {slice_idx}")
        plt.tight_layout()
        plt.show()
        """
        
        return total_loss
    
    
class MSE_Divergent(nn.Module):
    def __init__(self, div_weight: np.uint = 1):
        super(MSE_Divergent, self).__init__()
        self.div    = Divergent()
        self.mse    = nn.MSELoss()
        self.alpha  = div_weight
        
    def forward(self, output, target):
        loss =  self.mse(output[:,0], target[:,0])
        loss += self.mse(output[:,1], target[:,1])
        loss += self.mse(output[:,2], target[:,2])
        loss += self.alpha * self.div(output, target)
        return loss