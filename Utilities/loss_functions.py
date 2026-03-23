import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.nn import BCELoss, functional
import torch
import numpy as np


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
        mask = self.mask_law(output, target)
        
        if self.mode == 'flatten':
            return self.lossFunction(output[mask], target[mask])
        
        elif self.mode == 'overwrite':
            temp_output = output.clone()
            temp_output[~mask] = target[~mask]
            output = temp_output            
            return self.lossFunction(output, target)
        
        else: raise Exception(f"Mask_LossFunction mode {self.mode} not implemented. Must be one of flatten or overwrite")

        
    
    
    
    
#######################################################
#************ LOSS FUNCTIONS  ************************#
#######################################################

# MY COMPOSED FUNCTIONS


class Log10MaskedLoss(nn.Module):

    def __init__(self, base_loss: nn.Module, eps: float = 1e-9, multi: float = 1e9):
        super().__init__()
        self.base_loss = base_loss
        self.eps = eps
        self.multi =  multi

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
        
        std_loss = self.base_loss(output, target)
        log_loss = self.base_loss(torch.log10((output.abs()+ self.eps) *self.multi )*(output.sign()), 
                              torch.log10((target.abs()+ self.eps) *self.multi )*(target.sign()))
        
        return std_loss + 0.001*log_loss
                              

    
# Permeability Relative Percentual Error
class PRPE(nn.Module):
    def __init__(self):
        super(PRPE, self).__init__()
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        mean_error = 100*( (output.mean() - target.mean())/target.mean() ).abs()
        return mean_error
    

class PearsonCorr(nn.Module):
    def __init__(self, N_samples, eps=1e-8):
        super(PearsonCorr, self).__init__()
        self.eps=eps
        self.N_samples=N_samples 
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        # 1. Flatten the tensors to treat all cells as a single distribution
        x = output.flatten()
        y = target.flatten()
        
        indx = torch.randperm(x.size(0))[:self.N_samples]
        x = x[indx]
        y = y[indx]
        
        # 2. Centering (Subtract the mean)
        x_mu = x - x.mean()
        y_mu = y - y.mean()
        
        numerator   = torch.sum(x_mu * y_mu)
        denominator = torch.sqrt(torch.sum(x_mu**2)) * torch.sqrt(torch.sum(y_mu**2))
                
        return (numerator / (denominator + self.eps))
    
        
    
# The 'weights' parameter should be a list of lists or a 2D torch.Tensor
# with shape (num_channels, num_spatial_dims).
#
# - The number of rows (list elements) must equal the number of channels.
#   (e.g., U, V, W for velocity)
#
# - The number of columns (elements in each inner list) must equal
#   the number of spatial dimensions.
#   (e.g., 2 for (x, y) or 3 for (x, y, z))
#
# The weight element i,j actuates on dU_i / de_j (e.g, w(1,3) = dUx/dz)
            
class MeanJacobianError(nn.Module):
    def __init__(self):
        super(MeanJacobianError, self).__init__()

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")
        
        # Determine the number of spatial dimensions.
        if output.ndim == 4:  # (B, C, X, Y)
            spatial_dims = 2
        elif output.ndim == 5:  # (B, C, X, Y, Z)
            spatial_dims = 3
        else:
            raise ValueError(f"Tensor shape not supported. Expected 4 or 5 dimensions (B, C, spatial...) but got {output.shape}")

        num_channels = output.shape[1]
        loss = 0.0

        # Loop over each output channel (e.g., u, v, w)
        for c in range(num_channels):
            # Extract a single channel from the output and target tensors
            # Shape becomes (B, X, Y, Z) or (B, X, Y)
            output_channel = output[:, c, ...]
            target_channel = target[:, c, ...]

            # Compute the gradients of the current channel with respect to all spatial dimensions.
            # torch.gradient returns a tuple, one tensor for each spatial dimension.
            # e.g., for 3D, it returns (grad_x, grad_y, grad_z)
            output_grads = torch.gradient(output_channel, dim=tuple(range(1, 1 + spatial_dims)))
            target_grads = torch.gradient(target_channel, dim=tuple(range(1, 1 + spatial_dims)))

            # Sum the mean squared error for each gradient component.
            for i in range(spatial_dims):
                # We are now comparing d(output_channel)/d(spatial_dim[i]) with d(target_channel)/d(spatial_dim[i])
                loss += ((output_grads[i] - target_grads[i])**2).mean()
                

        return loss / (num_channels * spatial_dims)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
class IterativeGradLoss(nn.Module):

    def __init__(self, weights, edge_order=2):
        super().__init__()
        if not isinstance(weights, (list, tuple)) or len(weights) < 1:
            raise ValueError("`weights` must be a non-empty list/tuple.")
        self.weights = [float(w) for w in weights]
        self.edge_order = int(edge_order)

    @staticmethod
    def _check_shapes(output, target):
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")
        if output.ndim not in (4, 5):
            raise ValueError(f"Expected 4D (B,C,X,Y) or 5D (B,C,X,Y,Z), got {tuple(output.shape)}")

    @staticmethod
    def _order_loss(list_out, list_tgt):
        if len(list_out) != len(list_tgt):
            raise RuntimeError("Derivative lists size mismatch.")
        loss = list_out[0].new_tensor(0.0)
        for a, b in zip(list_out, list_tgt):
            loss = loss + F.mse_loss(a, b)
        return loss / len(list_out)

    def forward(self, output, target):
        self._check_shapes(output, target)
        C = output.shape[1]
        total = output.new_tensor(0.0)

        for c in range(C):
            # Start with the base scalar field for this channel: (B, X, Y[, Z])
            out_cur = [output[:, c, ...]]
            tgt_cur = [target[:, c, ...]]

            # Spatial meta
            spatial_dims = out_cur[0].ndim - 1
            dims = tuple(range(1, 1 + spatial_dims))

            # Build orders iteratively
            for order_idx, w in enumerate(self.weights, start=1):
                # Take gradient of every tensor in the previous order along all axes
                out_next, tgt_next = [], []
                for o_prev, t_prev in zip(out_cur, tgt_cur):
                    # torch.gradient returns a tuple of len(spatial_dims)
                    o_grads = torch.gradient(o_prev, dim=dims, edge_order=self.edge_order)
                    t_grads = torch.gradient(t_prev, dim=dims, edge_order=self.edge_order)
                    # Append each axis derivative as a separate component
                    out_next.extend(o_grads)
                    tgt_next.extend(t_grads)

                # Compute this order's loss:
                # For order 1, this averages MSE over gradient components (axes).
                # For order 2, this averages over all Hessian entries, etc.
                if w != 0.0:
                    # (optional) normalize by spatial_dims to keep order-1 on similar scale
                    # but since we average over all components already, no extra /spatial_dims needed
                    total = total + w * self._order_loss(out_next, tgt_next)

                # Prepare for next order (gradients of gradients, etc.)
                out_cur, tgt_cur = out_next, tgt_next

        # Average across channels so batch/scale is consistent
        return total / C
"""


class IterativeGradLoss(nn.Module):
    """
    Weighted sum of errors of different orders, starting with MSE on the inputs.
    
    weights[0] -> 0-th order (MSE on the tensors themselves)
    weights[1] -> 1st order (∇u components)
    weights[2] -> 2nd order (Hessian entries: ∂²/∂x_i∂x_j)
    
    The loss for each order k is:
    - calculated as the average MSE across all tensors of that order.
    - multiplied by weights[k].
    - accumulated into the total loss.
    """
    def __init__(self, weights, edge_order=2):
        super().__init__()
        if not isinstance(weights, (list, tuple)) or len(weights) < 1:
            raise ValueError("`weights` must be a non-empty list/tuple.")
        self.weights = [float(w) for w in weights]
        self.edge_order = int(edge_order)

    @staticmethod
    def _check_shapes(output, target):
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")
        if output.ndim not in (4, 5):
            raise ValueError(f"Expected 4D (B,C,X,Y) or 5D (B,C,X,Y,Z), got {tuple(output.shape)}")

    @staticmethod
    def _order_loss(list_out, list_tgt):
        """Average MSE over a list of tensors (same structure/order)."""
        if len(list_out) != len(list_tgt):
            raise RuntimeError("Derivative lists size mismatch.")
        loss = list_out[0].new_tensor(0.0) # Create loss sum initially as 0.0 with same type (float) as list_out[0]
        for a, b in zip(list_out, list_tgt): # Unzip tuple (gradx, grady, gradz)
            
            loss += F.mse_loss(a, b)
        return loss / len(list_out)

    def forward(self, output, target):
        self._check_shapes(output, target)
        C = output.shape[1]    # Number of channels (3=(Uz,Uy,Ux), 2=(Ux,Uy), 1=(Uz))
        total = output.new_tensor(0.0)
        
        # Start with the base tensors for each channel: Separate the tensor of each direction as a list element
        out_cur = [output[:, c, ...] for c in range(C)] # [Ux, Uy, Uz]
        tgt_cur = [target[:, c, ...] for c in range(C)]

        # 0th-order loss: MSE over the original velocity directions
        total += self.weights[0] * self._order_loss(out_cur, tgt_cur)

        # For each desired order: Build derivates iteratively
        
        # for each channel
        for c in range(C):
            for order_idx, w in enumerate(self.weights[1:], start=1):  
                
                
                output[:, c, ...]
                target[:, c, ...]
        """
        for order_idx, w in enumerate(self.weights[1:], start=1): 
            #print(f"Computing Order {order_idx}")
            out_next, tgt_next = [], []
            # For each previous variable, add it gradients to be computed next
            for o_prev, t_prev in zip(out_cur, tgt_cur):  
                spatial_dims    = o_prev.ndim - 1 # Total tensor dimensions (B,C, X,Y) or (B,C, Z,X,Y)
                dims            = tuple(range(1, 1 + spatial_dims))  # Sets derivative of each spatial dimension (last 1 2 or 3)
                
                # Take gradient of every tensor in the previous order along all axes
                # Append each axis derivative as a separate component of a list
                out_next.extend(torch.gradient(o_prev, dim=dims, edge_order=self.edge_order))
                tgt_next.extend(torch.gradient(t_prev, dim=dims, edge_order=self.edge_order))

            # Compute this order's loss and add it to the total
            total += w * self._order_loss(out_next, tgt_next)

            # Prepare for next order (gradients of gradients, etc.)
            out_cur, tgt_cur = out_next, tgt_next
        """
        return total
    
class STAFE(nn.Module):
    def __init__(self, dim =0):
        super(STAFE, self).__init__()
        
    def forward(self, output, target):
        # Ensure the output and target tensors have the same shape.
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")    
        
        return torch.sum( torch.abs(torch.sum(output, dim=0)-torch.sum(target, dim=0)) )  / (torch.sum( torch.abs(torch.sum(target, dim=0)) ))
    
class MultiScaleLoss(nn.Module):
    # 'normalize_mode' can be : 
    #  - 'none' to return the raw sum, 
    #  - 'n_scales' to divide by the num. of scales and return the mean across scales
    #  - 'var' to divide by the variance of higher resolution's scale target
    def __init__(self, loss_fn, n_scales=4, norm_mode='none'):
        """
        Args:
            loss_fn: any PyTorch loss function (e.g., nn.MSELoss(), nn.L1Loss())
        """
        super(MultiScaleLoss, self).__init__()
        self.loss_fn    = loss_fn
        self.norm_mode  = norm_mode
        self.scales     = n_scales
        

    def forward(self, y_pred, y):
        """
        Args:
            y_pred (List[Tensor]): predictions at each scale
            y (List[Tensor] or Tensor): ground truths at each scale
        Returns:
            loss: total multiscale loss
        """
        # If the ground truth is a tensor: make it multi-scale
        if torch.is_tensor(y):
            y = self.get_coarsened_list(y)
        
        # Validate input types
        if not isinstance(y_pred, (list, tuple)):
            raise TypeError(f"Expected y_pred to be list or tuple, got {type(y_pred)}")
        if not isinstance(y, (list, tuple)):
            raise TypeError(f"Expected y to be list or tuple, got {type(y)}")
        if len(y_pred) != len(y):
            raise ValueError(f"Mismatch in number of scales: {len(y_pred)} predictions vs {len(y)} targets")
        
        total_loss  = y_pred[-1].new_tensor(0.0)
        y_vars      = torch.var(y[-1], dim=list(range(1, y[-1].ndim))) # Compute var over batch dimension, reducing all others        
        y_max       = torch.amax(y[-1].abs(), dim=list(range(1, y[-1].ndim)))
        y_avg       = torch.mean(y[-1].abs(), dim=list(range(1, y[-1].ndim)))
        
        # For each scale
        for scale, (y_hats, y_trues) in enumerate(zip(y_pred, y)): # Iterate over listed scales
            if y_hats.shape != y_trues.shape:
                raise ValueError(f"Shape mismatch at scale {scale}: {y_hats.shape} vs {y_trues.shape}")
            
            # For each sample
            for sample_idx, (y_hat, y_true) in enumerate(zip(y_hats, y_trues)):
                # Get the scaled image loss, then include it to the total
                
                if self.norm_mode=='var':       total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_vars[sample_idx])
                elif self.norm_mode=='max':     total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_max[sample_idx])
                elif self.norm_mode=='avg':     total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_avg[sample_idx])
                else:                           total_loss += self.loss_fn(y_hat, y_true)/len(y_pred)
                
        return total_loss
    
    def get_coarsened_list(self, x):    
        ds_x = []
        ds_x.append(x)
        for i in range( self.scales-1 ): 
            ds_x.append( self.scale_tensor( ds_x[-1], scale_factor=1/2 ) )
        return ds_x[::-1] # returns the reversed list (small images first)
    
    def scale_tensor(self, x, scale_factor=1):
        
        # Downscale images
        if scale_factor<1:
            return nn.AvgPool3d(kernel_size = int(1/scale_factor))(x)
        
        # Upscale images
        elif scale_factor>1:
            for repeat in range (0, int(np.log2(scale_factor)) ):  # number of repeatsx2
                for ax in range(2,5): # (B,C,  H,W,D), repeat only the 3D axis, not batch and channel
                    x=x.repeat_interleave(repeats=2, axis=ax)
            return x
        
        # Do not change images
        elif scale_factor==1:
            return x
        
        else: raise ValueError(f"Scale factor not understood: {scale_factor}")
       
    


class LogTransform(nn.Module):
    def __init__(self, fun, order=3):
        super(LogTransform, self).__init__()
        self.fun   = fun
        self.order = order
    def forward(self, output, target):
        # Ensure the output and target tensors have the same shape.
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")    
        error = self.fun(output, target)
        return torch.pow(error, 1.0 / self.order) #error ** (1.0 / self.order)
            
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, output, target):
        # Ensure the output and target tensors have the same shape.
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")    
        errors = (target-output)
        
        loss = errors.abs()
        loss += 1 - torch.exp(- (2000.0 * errors)**2.0)
        loss += 1 - torch.exp(- (20.0 * errors)**2.0)
        loss += 1 - torch.exp(- (2.0 * errors)**2.0)
        loss += 1 - torch.exp(- (0.2 * errors)**2.0)

        return loss.mean()