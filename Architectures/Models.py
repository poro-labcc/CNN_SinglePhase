import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Architectures.FunctionalBlocks import (
    BASE_MODEL,
    ConvBlock,
    PoolingBlock,
    UpSampleBlock,
    ChannelConcat_Block,
    InceptionBlock
)

# MS-NET
"""
The original code is present in :
    https://github.com/je-santos/ms_net
The original code to generate each individual conv model was modified from:
    https://github.com/tamarott/SinGAN
"""
# Corrected
class Corrected_MS_Net(nn.Module):
    
    def __init__(
                 self, 
                 net_name     = 'test1', 
                 num_scales   =  4,
                 num_features =  1, 
                 num_filters  =  2, 
                 f_mult       =  4,  
                 summary      = False,
                 bin_input    = False,
                 ):
        
        super(Corrected_MS_Net, self).__init__()
        
        self.net_name = net_name
        self.scales   = num_scales
        self.feats    = num_features
        
        self.models   = nn.ModuleList( 
                                Corrected_MS_Net.get_SubModels( 
                                    num_scales,
                                    num_features,
                                    num_filters,
                                    f_mult,
                                    ) 
                                )

        self.bin_input = bin_input

        if summary:
            print(f'\n Here is a summary of your MS-Net ({net_name}): \n {self.models}')
        
        
    @staticmethod
    def get_SubModels(scales, features, filters, f_mult):
        
        """
        Returns an array with n-trainable models (ConvNets)
        """
        
        models   = []         # empty list to store the models
        nc_in    = features   # number of inputs on the first layer
        
        # list of number filters in each model (scale)
        num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
        for it in range( scales ): # creates a model for each scale
            if it==1: nc_in+=1     # adds an additional input to the subsecuent models 
                                   # to convolve the domain + previous(upscaled) result 
            models.append( 
                Corrected_MS_Net.Scale_SubModel( 
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
        with torch.no_grad():
            return self.forward(x)[-1]
        
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
        def __init__(self, nc_in, ncf):
            super().__init__()
            
            # default parameters
            nc_out     = 1   # number of output channels of the last layer
            ker_size   = 3   # kernel side-lenght
            padd_size  = 1   # padding size
            ncf_min    = ncf # min number of convolutional filters
            num_layers = 5   # number of conv layers
            stride     = 1
            
            self.reflec_pad = num_layers            
            self.reflector  = nn.ReflectionPad3d((0, 0, 0, 0, self.reflec_pad, self.reflec_pad))

            # first block
            self.head = Corrected_MS_Net.ConvBlock3D( 
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
                    convblock = Corrected_MS_Net.ConvBlock3D( 
                        in_channel  = max(2*new_ncf,ncf_min),
                        out_channel = max(new_ncf,ncf_min),
                        ker_size    = ker_size,
                        padd        = padd_size,
                        stride      = stride,
                        norm        = True,
                        activation  = False
                    )
                else:
                    convblock = Corrected_MS_Net.ConvBlock3D( 
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
                Corrected_MS_Net.ConvBlock3D( 
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
            #x = self.reflector(x)
            x = self.head(x)
            x = self.body(x)
            x = self.tail(x)
            #x = self.crop_3d(x)
            return x




class MS_Net(nn.Module):
    
    def __init__(
                 self, 
                 net_name     = 'test1', 
                 num_scales   =  4,
                 num_features =  1, 
                 num_filters  =  2, 
                 f_mult       =  4,  
                 summary      = False
                 ):
        
        super(MS_Net, self).__init__()
        
        self.net_name = net_name
        self.scales   = num_scales
        self.feats    = num_features
        
        self.models   = nn.ModuleList( 
                                MS_Net.get_SubModels( 
                                    num_scales,
                                    num_features,
                                    num_filters,
                                    f_mult ) 
                                )
        if summary:
            print(f'\n Here is a summary of your MS-Net ({net_name}): \n {self.models}')
        
        
    @staticmethod
    def get_SubModels(scales, features, filters, f_mult):
        
        """
        Returns an array with n-trainable models (ConvNets)
        """
        
        models   = []         # empty list to store the models
        nc_in    = features   # number of inputs on the first layer
        norm     = True       # use Norm
        last_act = None       # activation function
        
        # list of number filters in each model (scale)
        num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
        for it in range( scales ): # creates a model for each scale
            if it==1: nc_in+=1     # adds an additional input to the subsecuent models 
                                   # to convolve the domain + previous(upscaled) result 
            models.append( 
                MS_Net.Scale_SubModel( 
                nc_in    = nc_in,
                ncf      = num_filters[it],
                norm     = norm,
                last_act = last_act) 
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
        with torch.no_grad():
            return self.forward(x)[-1]
        
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
        def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm):
            super().__init__()
            
            
            self.add_module( 'conv',
                             nn.Conv3d( in_channel, 
                                        out_channel,
                                        kernel_size=ker_size,
                                        stride=stride,
                                        padding=padd ) ),
            if norm == True:
                self.add_module( 'i_norm', nn.InstanceNorm3d( out_channel ) ),
            self.add_module( 'CeLU', nn.CELU( inplace=False ) )

    class Scale_SubModel(nn.Module):
        def __init__(self, nc_in, ncf, norm, last_act):
            super().__init__()
            
            # default parameters
            nc_out     = 1   # number of output channels of the last layer
            ker_size   = 3   # kernel side-lenght
            padd_size  = 1   # padding size
            ncf_min    = ncf # min number of convolutional filters
            num_layers = 5   # number of conv layers
            stride     = 1
            
            # first block
            self.head = MS_Net.ConvBlock3D( 
                in_channel  = nc_in,
                out_channel = ncf,
                ker_size    = ker_size,
                padd        = padd_size,
                stride      = stride,
                norm        = norm)
            
            # Body of the model: stack 'num_layers' conv blocks
            self.body = nn.Sequential()
            for i in range( num_layers-1 ):
                new_ncf = int( ncf/2**(i+1) )
                if i==num_layers-2: norm=False  # no norm in the penultimate block
              
                convblock = MS_Net.ConvBlock3D( 
                    in_channel  = max(2*new_ncf,ncf_min),
                    out_channel = max(new_ncf,ncf_min),
                    ker_size    = ker_size,
                    padd        = padd_size,
                    stride      = stride,
                    norm        = norm
                )
                
                self.body.add_module( f'block{i+1}', convblock )
            
            if last_act == 'CELU':
                self.tail = nn.Sequential(
                                        nn.Conv3d( max(new_ncf,ncf_min), 
                                                  nc_out,
                                                  kernel_size   = 1,
                                                  stride        = stride,
                                                  padding       = 0),
                                        nn.CELU()
                                     )
            else:
                
                self.tail = nn.Sequential(
                                nn.Conv3d( max(new_ncf,ncf_min), nc_out, kernel_size=1,
                                           stride=stride, padding=0)) # no pad needed since 1x1x1
            
            
        def forward(self,x):
            x = self.head(x)
            x = self.body(x)
            x = self.tail(x)
            return x
        


# Danny D Ko
"""
The original code is present in :
    https://github.com/dko1217/DeepLearning-PorousMedia/tree/main
    
Assymetric padding is hadled here manually since Pytorch dont natively.
Original combination K=4, Stride=2, Padding='same' is problematic
"""

def calculate_same_padding(input_size, kernel_size, stride, dilation=1):
    """Calculate padding for 'SAME' padding mode."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    output_size = (input_size + stride - 1) // stride  # ceil division
    padding = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    return padding

def pad_for_same_conv_3d(x, kernel_size, stride, dilation=1):
    """
    Apply 'same' padding for Conv3D (padding before convolution).
    """
    
    if isinstance(kernel_size, tuple): kernel_size  = kernel_size[0]
    if isinstance(stride, tuple)     : stride       = stride[0]
    
    i_h, i_w, i_d = x.size()[-3:]
    # Calculate padding for each dimension
    pad_h = calculate_same_padding(i_h, kernel_size, stride, dilation)
    pad_w = calculate_same_padding(i_w, kernel_size, stride, dilation)
    pad_d = calculate_same_padding(i_d, kernel_size, stride, dilation)
    
    # Apply asymmetric padding
    # F.pad order: (depth_last, depth_first, width_last, width_first, height_last, height_first)
    return F.pad(x, [
        pad_d // 2, pad_d - pad_d // 2,  # depth
        pad_w // 2, pad_w - pad_w // 2,  # width
        pad_h // 2, pad_h - pad_h // 2   # height
    ])

def crop_for_same_deconv_3d(x, target_size):
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

class DannyKo_EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, activation, momentum, dropout_rate):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=0)  # Always use padding=0, we'll pad manually
        
        self.norm = nn.BatchNorm3d(out_channels, momentum=momentum)
        self.act  = nn.SELU() if activation == 'selu' else nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = kernel_size
        
    def forward(self, x):
        
        x = pad_for_same_conv_3d(x, self.kernel_size, self.stride)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class DannyKo_DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, activation, momentum, dropout_rate):
        super().__init__()
        
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, 
                                         kernel_size=kernel_size, 
                                         stride=stride, 
                                         padding=0,  # We'll handle padding manually
                                         output_padding=0)
        
        self.norm = nn.BatchNorm3d(out_channels, momentum=momentum)
        self.act = nn.SELU() if activation == 'selu' else nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = kernel_size
        
    def forward(self, x):
        input_size = x.size()[-3:]  # Save input spatial dimensions
        
        x = self.deconv(x)
        
        
        # Calculate expected output size for 'same' padding
        expected_h = input_size[0] * self.stride
        expected_w = input_size[1] * self.stride
        expected_d = input_size[2] * self.stride
            
        x = crop_for_same_deconv_3d(x, (expected_h, expected_w, expected_d))
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    
class DannyKo_Net_Original(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
        
        self.bin_input = bin_input
        
        self.x_model = self.UNetV1(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
     
        self.y_model = self.UNetV1(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.2,
            res_num=4,
            bin_input=bin_input)
        
        self.z_model = self.UNetV1(
            input_channels=1,
            output_channels=1,
            filter_num=10,
            filter_num_increase=2,
            filter_size=4,
            activation='selu',
            momentum=0.01,
            dropout=0.1,
            res_num=4,
            bin_input=bin_input)
        
        self.concat  = self.Concat_Block()
        
        self.main_model = self.UNetV1(
            input_channels=3,
            output_channels=3,
            filter_num=9,
            filter_num_increase=1,
            filter_size=3,
            activation='selu',
            momentum=0.01,
            dropout=0.001,
            res_num=3,
            bin_input=bin_input)
        
        
        
    def forward(self, x):
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            x_out = self.x_model(x) * 0.5
            y_out = self.y_model(x) * 0.5
            z_out = self.z_model(x)
                
        combined = self.concat(x_out, y_out, z_out)
        return self.main_model(combined)
    
    def predict(self,x): 
        if self.bin_input: x = (x > 0).to(torch.float32)
        with torch.no_grad():
            return self.forward(x)
         
    class Concat_Block(nn.Module):
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
    
    class UNetV1(nn.Module):
        def __init__(self, input_channels, output_channels=1, filter_num=5, filter_size=4, 
                     activation='selu', momentum=0.01, dropout=0.2, res_num=4, filter_num_increase=1, bin_input=True):
            super().__init__()
            
            # Initialize lists of modules
            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            self.skip_connection_indices = []
            self.concat = DannyKo_Net_Original.Concat_Block()
            self.bin_input = bin_input
            self.res_num = res_num
            self.filter_size = filter_size
            self.output_channels = output_channels
            self.filter_num = filter_num
            
            if filter_num_increase < 1:
                raise ValueError(
                    "filter_num_increase must be >= 1"
                )
            
            # ENCODER (res_num RESOLUTIONS, 2 BLOCKS PER RESOLUTION):            
            for i in range(res_num):
                n_filters = int(filter_num * (filter_num_increase ** i))
                
                if i == 0:          
                    # First block in first resolution
                    firstConv = DannyKo_EncBlock(
                        in_channels=input_channels,
                        out_channels=n_filters,
                        stride=1,  # Keep spatial dimensions
                        kernel_size=filter_size,
                        activation=activation, 
                        momentum=momentum, 
                        dropout_rate=dropout,
                    )
                else:
                    # Downsampling blocks
                    firstConv = DannyKo_EncBlock(
                        in_channels=self.encoder[i-1][-1].out_channels,
                        out_channels=n_filters,
                        stride=2,  # Reduce spatial dimensions by half
                        kernel_size=filter_size,
                        activation=activation, 
                        momentum=momentum, 
                        dropout_rate=dropout,
                    )
                    
                # Second block (no downsampling)
                secondConv = DannyKo_EncBlock(
                    in_channels=firstConv.out_channels,
                    out_channels=n_filters,
                    stride=1,  # Keep spatial dimensions
                    kernel_size=filter_size,
                    activation=activation, 
                    momentum=momentum, 
                    dropout_rate=dropout,
                )
                
                self.encoder.append(nn.ModuleList([firstConv, secondConv]))
            
            # DECODER (in reverse order)
            # The decoder list will be in order: [highest_res_block, ..., lowest_res_block]
            # So index 0 is the highest resolution (closest to output)
            
            for i in reversed(range(res_num)):
                # Check if this is the final layer (output layer)
                is_final_layer = (i == 0)
                
                if is_final_layer:
                    # Final output layer - use regular Conv3D instead of ConvTranspose3D
                    
                    # Determine input channels for the final layer
                    if len(self.decoder) > 0:
                        # There are previous decoder blocks, so we have a skip connection
                        in_channels_final = self.encoder[i][-1].out_channels + self.decoder[-1][-1].out_channels
                    else:
                        # No previous decoder blocks (res_num=1 case)
                        in_channels_final = self.encoder[i][-1].out_channels
                    
                    # First convolution in output layer (regular Conv3D with same padding)
                    firstConv = nn.Conv3d(
                        in_channels_final,
                        filter_num,
                        kernel_size=filter_size,
                        stride=1,
                        padding=0  # We'll handle padding manually in forward
                    )
                    
                    # Final output convolution (1x1 conv)
                    secondConv = nn.Conv3d(
                        filter_num,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                                        
                    
                else:
                    # Regular decoder blocks
                    n_filters = filter_num * (filter_num_increase ** (i - 1))
                    
                    # Determine input channels
                    if i == res_num - 1:
                        # First decoder block (lowest resolution, no previous decoder output)
                        in_channels_deconv = self.encoder[i][-1].out_channels
                    else:
                        # Middle blocks with skip connections
                        in_channels_deconv = self.encoder[i][-1].out_channels + self.decoder[-1][-1].out_channels
                    
                    # First deconvolution block (stride=1, maintains dimensions)
                    firstConv = DannyKo_DecBlock(
                        in_channels=in_channels_deconv,
                        out_channels=n_filters,
                        stride=1,  # Maintain spatial dimensions
                        kernel_size=filter_size,
                        activation=activation, 
                        momentum=momentum, 
                        dropout_rate=dropout,
                    )
                    
                    # Second block with upsampling (stride=2, doubles dimensions)
                    secondConv = DannyKo_DecBlock(
                        in_channels=n_filters,
                        out_channels=n_filters,
                        stride=2,  # Double spatial dimensions
                        kernel_size=filter_size,
                        activation=activation, 
                        momentum=momentum, 
                        dropout_rate=dropout,
                    )
                    
                firstConv.is_final_layer  = is_final_layer
                secondConv.is_final_layer = is_final_layer
                
                self.decoder.append(nn.ModuleList([firstConv, secondConv]))
        
        def predict(self,x): 
            if self.bin_input: x = (x > 0).to(torch.float32)
            with torch.no_grad():
                return self.forward(x)
        
        def forward(self, x):
            if self.bin_input: x = (x > 0).to(torch.float32)
            
            skips = []            
            # Encoder pass
            for i in range(len(self.encoder)):
                conv1, conv2 = self.encoder[i]
                x = conv1(x)
                x = conv2(x)
                skips.insert(0, x)  # Store for skip connections (reverse order)
            
            # Decoder pass
            for i in range(len(self.decoder)):
                conv1, conv2 = self.decoder[i]
                
                # Handle skip connections for all but the first decoder block
                if i == 0:  x = skips[i]
                else:       x = self.concat(x, skips[i])
                
               
                # Final layer - apply manual padding for regular Conv3D
                if conv1.is_final_layer == True:
                    x = pad_for_same_conv_3d(x, conv1.kernel_size, conv1.stride)
                
                x = conv1(x)
                
                # Final layer - apply manual padding for regular Conv3D
                if conv1.is_final_layer == True:
                    x = pad_for_same_conv_3d(x, conv2.kernel_size, conv2.stride)
                
                x = conv2(x)
            
            return x
        
    
class Inception(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, features_per_block = 16):
        super().__init__() 
        
        
        
        head = InceptionBlock(
            input_size      = input_size, 
            in_channels     = in_channels, 
            b1_out_channels = max(features_per_block//16,1), # 64
            
            b2_mid_channels = max(features_per_block//4,1), # 96
            b2_out_channels = max(features_per_block//2,1), # 128
            
            b3_mid_channels = max(features_per_block//8,1), # 16
            b3_out_channels = max(features_per_block//16,1), #32
            
            b4_out_channels = max(features_per_block//16,1)  # 32
        )
        
        body = InceptionBlock(
            input_size      = head.output_size, 
            in_channels     = head.out_channels, 
            b1_out_channels = max(features_per_block//16,1), # 64
            
            b2_mid_channels = max(features_per_block//8,1), # 16
            b2_out_channels = max(features_per_block//16,1), #32
            
            b3_mid_channels = max(features_per_block//4,1), # 96
            b3_out_channels = max(features_per_block//2,1), # 128
            
            b4_out_channels = max(features_per_block//16,1)  # 32
        )
        
        bodies = [body]
        
        for i in range(6):
        
            body = InceptionBlock(
                input_size      = bodies[-1].output_size, 
                in_channels     = bodies[-1].out_channels, 
                b1_out_channels = max(features_per_block//16,1), # 64
                
                b2_mid_channels = max(features_per_block//8,1), # 16
                b2_out_channels = max(features_per_block//16,1), #32
                
                b3_mid_channels = max(features_per_block//4,1), # 96
                b3_out_channels = max(features_per_block//2,1), # 128
                
                b4_out_channels = max(features_per_block//16,1)  # 32
            )
            
            bodies.append(body)


        tail = ConvBlock(input_size     =bodies[-1].output_size, 
                         in_channels    =bodies[-1].out_channels,
                         out_channels   =out_channels, 
                         kernel_size    =1)
                
        self.model = nn.Sequential(head, *bodies, tail)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self,x): 
        with torch.no_grad():
            return self.forward(x)
    

