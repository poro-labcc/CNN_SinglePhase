import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Functional import pad_same, crop_same, Channel_Concat
import matplotlib.pyplot as plt
# Danny D Ko
"""
The original code is present in :
    https://github.com/dko1217/DeepLearning-PorousMedia/tree/main
    
Assymetric padding is hadled here manually since Pytorch dont natively.
Original combination K=4, Stride=2, Padding='same' is problematic
"""

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
        
        x = pad_same(x, self.kernel_size, self.stride)
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
            
        x = crop_same(x, (expected_h, expected_w, expected_d))
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    



class Base_Unet(nn.Module):
    def __init__(self, input_channels, output_channels=1, filter_num=5, filter_size=4, 
                 activation='selu', momentum=0.01, dropout=0.2, res_num=4, filter_num_increase=1, bin_input=True):
        super().__init__()
        
        # Initialize lists of modules
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_connection_indices = []
        self.concat = Channel_Concat()
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
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
    
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
                x = pad_same(x, conv1.kernel_size, conv1.stride)
            
            x = conv1(x)
            
            # Final layer - apply manual padding for regular Conv3D
            if conv1.is_final_layer == True:
                x = pad_same(x, conv2.kernel_size, conv2.stride)
            
            x = conv2(x)
        
        return x
    
    
# Original structure of Danny Model: 
#   - do not include pressure
#   - weight sub-models outputs
class DannyKo_Net_Original(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
        
        self.bin_input = bin_input
        
        self.x_model = Base_Unet(
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
     
        self.y_model = Base_Unet(
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
        
        self.z_model = Base_Unet(
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
        
        self.concat = Channel_Concat()
        
        self.main_model = Base_Unet(
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
                
        combined = self.concat(z_out, y_out, x_out)
        return self.main_model(combined)
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask

        

# Danny Ko's model extended to include pressure 
class Extended_DannyKo(nn.Module):
    def __init__(self, bin_input=True):
        super().__init__() 
                
        self.bin_input = bin_input
        
        self.x_model = Base_Unet(
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
     
        self.y_model = Base_Unet(
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
        
        self.z_model = Base_Unet(
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
        
        self.p_model = Base_Unet(
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
        
        self.concat = Channel_Concat()
        
        self.main_model = Base_Unet(
            input_channels=4,
            output_channels=4,
            filter_num=9,
            filter_num_increase=2, # Originally 1
            filter_size=3,
            activation='selu',
            momentum=0.01,
            dropout=0.001,
            res_num=3,
            bin_input=bin_input)
        
        self.main_head = nn.Sequential(
            nn.Conv3d(in_channels=4,
                      out_channels=4, 
                      kernel_size=3,
                      stride=1, 
                      padding=1),

            nn.Conv3d(in_channels=4,
                      out_channels=4, 
                      kernel_size=1,
                      stride=1, 
                      padding=0),

            nn.Conv3d(in_channels=4,
                      out_channels=4, 
                      kernel_size=1,
                      stride=1, 
                      padding=0),
        )
    
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
        addition = self.main_model(combined)

        #print(f"Comb: {combined.abs().mean():<10.16f}; Add: {addition.abs().mean():<10.16f}")
        
        final_out = self.main_head(combined + addition)
        
        """
        # ==========================================
        # SEÇÃO DE PLOT / DEBUG (Agora com 4 colunas)
        # ==========================================
        num_channels = final_out.shape[1]
        x_slice_idx = final_out.shape[4] // 2
        
        # Aumentamos para 4 colunas e ajustamos a largura total (figsize de 18 para 24)
        fig, axes = plt.subplots(num_channels, 4, figsize=(24, 5 * num_channels), squeeze=False)
        
        print(f"UNET {num_channels} channels")
        
        for c in range(num_channels):
            img_z_out     = combined[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            add           = addition[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            img_final_out = final_out[0, c, :, :, x_slice_idx].detach().cpu().float().numpy()
            
            # --- COLUNA 0: Sub-modelo (Combined) ---
            im0 = axes[c, 0].imshow(img_z_out, cmap='jet')
            axes[c, 0].set_title(f"Combined Sub-models - Ch {c}")
            fig.colorbar(im0, ax=axes[c, 0], fraction=0.046, pad=0.04)
            
            # --- COLUNA 1: Adição (Main Model) com Percentis ---
            add_min, add_max = np.percentile(add, [1, 99])
            im1 = axes[c, 1].imshow(add, cmap='RdBu_r', vmin=add_min, vmax=add_max)
            axes[c, 1].set_title(f"Main Model Addition - Ch {c}\nScale: [{add_min:.2e}, {add_max:.2e}]")
            fig.colorbar(im1, ax=axes[c, 1], fraction=0.046, pad=0.04)
            
            # --- COLUNA 2: Saída Final ---
            im2 = axes[c, 2].imshow(img_final_out, cmap='jet')
            axes[c, 2].set_title(f"Final Output - Ch {c}")
            fig.colorbar(im2, ax=axes[c, 2], fraction=0.046, pad=0.04)

            # --- COLUNA 3: HISTOGRAMA DA ADIÇÃO ---
            axes[c, 3].hist(add.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7,range=(add_min, add_max))
            axes[c, 3].axvline(add_min, color='blue', linestyle='dashed', linewidth=1.5, label='1st %')
            axes[c, 3].axvline(add_max, color='red', linestyle='dashed', linewidth=1.5, label='99th %')
            axes[c, 3].set_title(f"Addition Histogram - Ch {c}")
            axes[c, 3].set_xlabel("Residual Value")
            axes[c, 3].set_ylabel("Frequency")
            axes[c, 3].legend()
            
        plt.tight_layout()
        plt.savefig('debug.png', dpi=300)
        
        plt.close(fig)
        """
        
        return final_out
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
         
    