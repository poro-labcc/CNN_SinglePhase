import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#######################################################
#************ BASE MODEL *****************************#
# The basic information that any proposed model to    #
# this image-to-image problem needs to carry          #
#######################################################

class BASE_MODEL(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__()
        self.bin_input = bin_input
        
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask


#######################################################
#************ USEFUL FUNCTIONS ***********************#
#######################################################

def Calculate_PaddingSame(input_size, kernel_size, stride, dilation=1):
    """Calculate padding for 'SAME' padding mode."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    output_size = (input_size + stride - 1) // stride  # ceil division
    padding = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    return padding

#######################################################
#************ MODEL BLOCKS: Tensor Modification    ***#
#######################################################

class ChannelWiseMult(nn.Module):
    def forward(self, x):
        return torch.prod(x, dim=1, keepdim=True)
    
class Channel_Concat(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("At least one tensor is required for concatenation.")
        if len(tensors) == 1:
            return tensors[0]
        
        # Ensure all tensors have the same spatial dimensions
        shapes = [t.shape for t in tensors]
        batch_size = shapes[0][0]
        spatial_dims = shapes[0][2:]  # H, W, D
        
        for i, t in enumerate(tensors):
            if t.shape[0] != batch_size or t.shape[2:] != spatial_dims:
                raise ValueError(f"Tensor {i} has mismatched shape: {t.shape}. Expected batch: {batch_size}, spatial: {spatial_dims}")
        
        # Concatenate along channel dimension
        return torch.cat(tensors, dim=1)
    
def pad_same(x, kernel_size, stride, dilation=1):
    """
    Apply 'same' padding for Conv3D (padding before convolution).
    """
    
    if isinstance(kernel_size, tuple): kernel_size  = kernel_size[0]
    if isinstance(stride, tuple)     : stride       = stride[0]
    
    i_h, i_w, i_d = x.size()[-3:]
    # Calculate padding for each dimension
    pad_h = Calculate_PaddingSame(i_h, kernel_size, stride, dilation)
    pad_w = Calculate_PaddingSame(i_w, kernel_size, stride, dilation)
    pad_d = Calculate_PaddingSame(i_d, kernel_size, stride, dilation)
    
    # Apply asymmetric padding
    # F.pad order: (depth_last, depth_first, width_last, width_first, height_last, height_first)
    return F.pad(x, [
        pad_d // 2, pad_d - pad_d // 2,  # depth
        pad_w // 2, pad_w - pad_w // 2,  # width
        pad_h // 2, pad_h - pad_h // 2   # height
    ])

def crop_same(x, target_size):
    """
    Crop output of ConvTranspose3D to match target_size for 'same' padding.
    """
    current_h, current_w, current_d = x.size()[-3:]
    target_h, target_w, target_d = target_size
    
    # Calculate cropping amounts
    crop_h = current_h - target_h
    crop_w = current_w - target_w
    crop_d = current_d - target_d
    
    # Apply cropping (symmetric if possible, otherwise prefer more from end)
    h_start = crop_h // 2
    h_end = current_h - (crop_h - crop_h // 2)
    
    w_start = crop_w // 2
    w_end = current_w - (crop_w - crop_w // 2)
    
    d_start = crop_d // 2
    d_end = current_d - (crop_d - crop_d // 2)
    
    return x[..., h_start:h_end, w_start:w_end, d_start:d_end]


