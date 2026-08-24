import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Functional import Channel_Concat


# Corrected version of MS-NET, modifying norm order and CeLU according to the paper
"""
The original code is present in :
    https://github.com/je-santos/ms_net
The original code to generate each individual conv model was modified from:
    https://github.com/tamarott/SinGAN
"""

# Class used to transform a loss function into multi scale
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
       
        
class JavierSantos(nn.Module):
    
    def __init__(
                 self, 
                 nc_out       =  1, # Number of Output Channels (Uz=1, Uz+Uy+Ux=3, etc)
                 num_scales   =  4, # Number of Resolutions / Sub-Models 
                 num_features =  1, # Number of Input Feature Channels
                 num_filters  =  2, # Increase of Filters
                 f_mult       =  4, # Increase of Channels 
                 bin_input    = False,
                 ):
        
        super(JavierSantos, self).__init__()
        
        self.scales   = num_scales
        self.feats    = num_features
        self.n_out    = nc_out
        self.models   = nn.ModuleList( 
                                JavierSantos.get_SubModels( 
                                    nc_out,
                                    num_scales,
                                    num_features,
                                    num_filters,
                                    f_mult,
                                    ) 
                                )

        self.bin_input = bin_input

        
        
    @staticmethod
    def get_SubModels(nc_out, scales, features, filters, f_mult):
        
        """
        Returns an array with n-trainable models (ConvNets)
        """
        
        models   = []         # empty list to store the models
        nc_in    = features   # number of inputs on the first layer
        
        # list of number filters in each model (scale)
        num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
        print("Number of Filters: ", num_filters)
        for it in range( scales ): # creates a model for each scale
            if it==1: nc_in+=1     # adds an additional input to the subsecuent models 
                                   # to convolve the domain + previous(upscaled) result 
                                   
            models.append( 
                JavierSantos.Scale_SubModel( 
                    nc_out   = nc_out,
                    nc_in    = nc_in,
                    ncf      = num_filters[it])
            )
                
        return models  
    
    def get_Masks(self, x, scales):
        """
        x: euclidean distance 3D array at the finest scale
        Returns array with masks
        
        Notes:
            for n scales we need n masks (the last one is binary)
        """    
        masks    = [None]*(scales)
        pooled   = [None]*(scales)
        
        pooled[0] = (x>0).float() # 0s at the solids, 1s at the empty space
        masks[0]  = pooled[0].squeeze(0)
        
        
        for scale in range(1,scales):
            pooled[scale] = nn.AvgPool3d(kernel_size = 2)(pooled[scale-1])
            denom = pooled[scale].clone()   # calculate the denominator for the mask
            denom[denom==0] = 1e8  # regularize to avoid divide by zero
            for ax in range(2,5):   # repeat along 3 axis
                denom=denom.repeat_interleave( repeats=2, axis=ax ) # Upscale
            # Calculate the mask as Mask = Image / Upscale( Downscale(Img) )
            masks[ scale ] = torch.div( pooled[scale-1], denom ).squeeze(0) 
        return masks[::-1] # returns a list with masks. smallest size first
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)[-1] 

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
        
        
    def forward(self, x):
        # The coarsest network receives only the domain representation 
        # at the coarsest scale, while the subsequent ones receive two
        # the domain representation at the appropriate scale, 
        # and the prediction from the previous scale. 
        # As mentioned above, the input’s linear size is reduced by 
        # a factor of two between every scale.
        # x_list is the sample's input, a list of coarsened versions of an image
        x_list  = self.get_coarsened_list(x)
        masks   = self.get_Masks( (x_list[-1]>0).float(), self.scales)
        
        assert x_list[0].shape[1] == self.feats, \
        f'The number of features provided {x_list[0].shape[1]} \
            does not match with the input size {self.feats}'
            
        # Carry-out the first prediction (pass through the coarsest model)
        # Calculate the lower resolution output (no upscale mask applied)
        y = [ self.models[0]( x_list[0] ) ]
        
        for scale,[ model,x ] in enumerate(zip( self.models[1:],x_list[1:] )):

            y_up = self.scale_tensor( y[scale], scale_factor=2 )*masks[scale]
            # Residual operation: what the model must learn to add in the previous solution, 
            # based on distance map and previous solution
            y.append( model( torch.cat((x,y_up),dim=1) ) + y_up )
            
        return y
    
    def get_coarsened_list(self, x):    
        if self.bin_input: x = (x > 0).to(torch.float32)
        
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
        
    class ConvBlock3D( nn.Sequential ):
        def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm, activation):
            super().__init__()
            
            
            self.add_module( 'conv',
                             nn.Conv3d( in_channel, 
                                        out_channel,
                                        kernel_size=ker_size,
                                        stride=stride,
                                        padding=padd ) ),
            if norm == True:
                self.add_module( 'i_norm', nn.InstanceNorm3d( out_channel ) )
            if activation== True:
                self.add_module( 'CeLU', nn.CELU( inplace=False , alpha=2) )

    class Scale_SubModel(nn.Module):
        def __init__(self, nc_in, ncf, nc_out):
            super().__init__()
            
            # default parameters
            ker_size   = 3   # kernel side-lenght
            padd_size  = 1   # padding size
            ncf_min    = ncf # min number of convolutional filters
            num_layers = 5   # number of conv layers
            stride     = 1
            
            self.reflec_pad = num_layers            
            self.reflector  = nn.ReflectionPad3d((0, 0, 0, 0, self.reflec_pad, self.reflec_pad))

            # first block
            self.head = JavierSantos.ConvBlock3D( 
                in_channel  = nc_in,
                out_channel = ncf,
                ker_size    = ker_size,
                padd        = padd_size,
                stride      = stride,
                norm        = True,
                activation  = True )
            
            # Body of the model: stack 'num_layers' conv blocks
            self.body = nn.Sequential()
            for i in range( num_layers-1 ):
                new_ncf = int( ncf/2**(i+1) )
                if i==num_layers-2:
                    convblock = JavierSantos.ConvBlock3D( 
                        in_channel  = max(2*new_ncf,ncf_min),
                        out_channel = max(new_ncf,ncf_min),
                        ker_size    = ker_size,
                        padd        = padd_size,
                        stride      = stride,
                        norm        = True,
                        activation  = False
                    )
                else:
                    convblock = JavierSantos.ConvBlock3D( 
                        in_channel  = max(2*new_ncf,ncf_min),
                        out_channel = max(new_ncf,ncf_min),
                        ker_size    = ker_size,
                        padd        = padd_size,
                        stride      = stride,
                        norm        = True,
                        activation  = True
                    )
                    
                
                self.body.add_module( f'block{i+1}', convblock )
                
            self.tail = nn.Sequential(
                JavierSantos.ConvBlock3D( 
                    in_channel  = max(new_ncf,ncf_min),
                    out_channel = nc_out,
                    ker_size    = 1,
                    padd        = 0,
                    stride      = stride,
                    norm        = False,
                    activation  = False
                ))
            
        def crop_3d(self, x):
            if self.reflec_pad == 0:
                return x
            return x[:, :, self.reflec_pad:-self.reflec_pad, :, :] # Keep Batch, Keep Channel, Pad z, Keep Y and Z axis

        def forward(self,x):
            x = self.head(x)
            x = self.body(x)
            x = self.tail(x)
            return x 



# Javier Santo's model extended to include pressure 
class JavierSantos_Extended(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
        
        self.bin_input = bin_input
        
        self.x_model    = JavierSantos(nc_out = 1, num_features = 1)
        
        self.y_model    = JavierSantos(nc_out = 1, num_features = 1)
        
        self.z_model    = JavierSantos(nc_out = 1, num_features = 1)
        
        self.p_model    = JavierSantos(nc_out = 1, num_features = 1)
        
        self.concat     = Channel_Concat()
        
        self.main_model = JavierSantos(nc_out = 4, num_features = 4)
    
    # Modified to freeze sub-models
    def train(self, mode=True):
        # 1. Call the standard train method for the main_model
        super().train(mode)
        
        # 2. Force the sub-models back to eval mode immediately
        self.x_model.eval()
        self.y_model.eval()
        self.z_model.eval()
        self.p_model.eval()
        return self
    
    def forward(self, x):
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            x_out = self.x_model.predict(x) 
            y_out = self.y_model.predict(x) 
            z_out = self.z_model.predict(x)
            p_out = self.p_model.predict(x)
                
        combined = self.concat(z_out, y_out, x_out, p_out)
        return combined + self.main_model(combined)
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
         
