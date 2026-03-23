import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from Utilities import loader_handler as lc
import warnings

#######################################################
#**** CUSTOMIZATION TO DEAL WITH MS-Net **************#
#######################################################

# Made by Gabriel Silveira
class MultiScaleDataset(Dataset):
    def __init__(self, base_dataset, num_scales=4):
        self.base_dataset = base_dataset
        self.num_scales   = num_scales

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        input_tensor, target_tensor = self.base_dataset[idx]
        input_scales    = self.get_coarsened_list(input_tensor, num_scales=self.num_scales)
        target_scales   = self.get_coarsened_list(target_tensor, num_scales=self.num_scales)
        return input_scales, target_scales
    
    def get_coarsened_list(self, x, num_scales):    
        ds_x = []
        ds_x.append(x)
        for i in range( num_scales-1 ): 
            ds_x.append( self.scale_tensor( ds_x[-1], scale_factor=1/2 ) )
        return ds_x[::-1] # returns the reversed list (small images first)
        
    def add_dims(self, x, num_dims):
        for dims in range(num_dims):
            x = x[np.newaxis]
        return x
    
    def scale_tensor(self, x, scale_factor=1):
        
        # Downscale image
        if scale_factor<1:
            return nn.AvgPool3d(kernel_size = int(1/scale_factor))(x)
        
        # Upscale image (Never used in the code...)
        elif scale_factor>1:
            for repeat in range (0, int(np.log2(scale_factor)) ):  # number of repeatsx2
                for ax in range(2,5): # (B,C,  H,W,D), repeat only the 3D axis, not batch and channel
                    x=x.repeat_interleave(repeats=2, axis=ax)
            return x
        
        # No alter image
        elif scale_factor==1:
            return x
        
        else: raise ValueError(f"Scale factor not understood: {scale_factor}")
        

    @staticmethod
    def get_dataloader(scaled_data, batch_size=None, verbose=False, transform_target=None, transform_input=None):
        # Divide list of pairs into array of N inputs and array of N targets
        scaled_input_tensors    = []
        scaled_output_tensors   = []
        
        # Separate scaled_data in different lists 
        for input_i_scales, target_i_scales in scaled_data: # Each Input and Target is a list of Tensors(scales)
            
            if transform_target is not None:
                new_target_i_scales = []
                for target_i_scale in target_i_scales:
                    new_target_i_scales.append( transform_target(target_i_scale) )
                scaled_output_tensors.append(new_target_i_scales)
            else:
                scaled_output_tensors.append(target_i_scales)

            if transform_input is not None:
                new_input_i_scales = []
                for input_i_scale in input_i_scales:
                    new_input_i_scales.append( transform_target(input_i_scale) )
                scaled_input_tensors.append(new_input_i_scales)
            else:
                scaled_input_tensors.append(input_i_scales)
                
                
        dataloader = lc.Data_Loader(scaled_input_tensors, scaled_output_tensors, batch_size=batch_size)
        del scaled_input_tensors
        del scaled_output_tensors
        
        # Print dimensions
        if verbose:
            inputs, targets = next(iter(dataloader.loader))    # Get first sample item: List of Tensors (scales)
            print(f" Loader with {len(dataloader.loader)} samples")
            print(f" Multi-Scale samples: {len(inputs)}")
            for i,(scale_input, scale_target) in enumerate(zip(inputs, targets)):
                print(f" -- scale {i}: in_shape: {scale_input.shape}, target_shape: {scale_target.shape}")
            print()
        
        return dataloader

        
        
from torch.utils.data import Dataset
import h5py

class LazyDatasetTorch(Dataset):
    """
    Lazy HDF5-backed dataset for PyTorch.
    Returns:
      X: (1, Z, Y, X) 
      Y: (1 or 4, Z, Y, X) 
    """

    def __init__(self, h5_path, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32):
        self.h5_path        = h5_path
        self.list_ids       = list_ids
        self.x_dtype        = x_dtype
        self.y_dtype        = y_dtype
        self.uni_directional= None
        self._validate_file()
        
    def _validate_file(self):
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Verifica existência das chaves principais
                required    = ['vel_z', 'vel_y', 'vel_x', 'edt', 'n_valid']
                missing     = [key for key in required if key not in f]
                if missing: raise KeyError(f"LazyDatasetTorch's .h5 file has a missing key: {missing}")
                
                self.total_samples = f['vel_z'].shape[0]
                
                # Verify ID range
                if self.list_ids is None or np.max(self.list_ids) >= self.total_samples:
                    if self.list_ids is None: 
                        warnings.warn(f"Number of samples not provided."
                                      f"The dataset will consider {self.total_samples} samples from {self.h5_path}.", 
                                      UserWarning, stacklevel=2)
                    elif np.max(self.list_ids) >= self.total_samples:
                        warnings.warn(f"Max listed ID {np.max(self.list_ids)} "
                                      f"exceeds the provided total. The dataset will consider only {self.total_samples} samples from {self.h5_path}.", 
                                      UserWarning, stacklevel=2)
                    
                    
                    self.list_ids = list(range(self.total_samples))
                
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.h5_path}")
            
    def __len__(self):
        return len(self.list_ids)   
    
    # Return one sample from index idx with shapes: X: (1, D, H, W), y: (3, D, H, W)
    def __getitem__(self, idx):
        # Select indexes for batch
        sample_id   = int(self.list_ids[idx])
        X, y        = self.__getbatch__([sample_id], device="cpu")

        return X.squeeze(0), y.squeeze(0)

    # Lazy interaction with .h5
    # Return samples with shapes X: (B, 1, D, H, W), y: (B, 3, D, H, W)
    # where B is the number of itens in 'sample_indices'
    def __getbatch__(self, sample_indices, device='cpu'):

        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        sample_indices = np.sort(sample_indices)
        batch_size     = len(sample_indices)

        with h5py.File(self.h5_path, "r") as f:
            # Shape original do domínio
            D, H, W     = f.attrs["raw_shape"]
            coori       = f["coorX"][sample_indices]
            coorj       = f["coorY"][sample_indices]
            coork       = f["coorZ"][sample_indices]
            edt         = f["edt"][sample_indices]
            n_valid     = f["n_valid"][sample_indices]
            
            if self.uni_directional == 0:
                # Load batch data
                vel_z       = f["vel_z"][sample_indices]
            
                # Create solid regions
                vel_z_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    vel_z_3d[b, k, j, i] = vel_z[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                vel_z_3d = torch.as_tensor(vel_z_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,        dtype=self.x_dtype, device=device)
                # Create a channel dimension of input
                X        = X.unsqueeze(1)
                Y        = vel_z_3d.unsqueeze(1)
                del vel_z_3d
                
            elif self.uni_directional == 1:
                # Load batch data
                vel_y       = f["vel_y"][sample_indices]
                # Create solid regions
                vel_y_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    vel_y_3d[b, k, j, i] = vel_y[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                vel_y_3d = torch.as_tensor(vel_y_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,   dtype=self.x_dtype, device=device)
                # Create a channel dimension of input
                X        = X.unsqueeze(1)
                Y        = vel_y_3d.unsqueeze(1)
                # Delete data after using
                del vel_y_3d
                
            elif self.uni_directional == 2:
                # Load batch data
                vel_x       = f["vel_x"][sample_indices]

                # Create solid regionS
                vel_x_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    vel_x_3d[b, k, j, i] = vel_x[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                vel_x_3d = torch.as_tensor(vel_x_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,   dtype=self.x_dtype, device=device)
                # Create a channel dimension of input
                X        = X.unsqueeze(1)
                Y        = vel_x_3d.unsqueeze(1)     
                # Delete data after using
                del vel_x_3d
                
            elif self.uni_directional == 3:
                # Load batch data
                press       = f["press"][sample_indices]

                # Create solid regions
                press_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    press_3d[b, k, j, i] = press[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                press_3d = torch.as_tensor(press_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,   dtype=self.x_dtype, device=device)
                # Create a channel dimension of input
                X       = X.unsqueeze(1)
                Y       = press_3d.unsqueeze(1)
                # Delete data after using
                del press_3d
            elif self.uni_directional == 4:
                # Load batch data
                vel_z = vel_y = vel_x = None 
                vel_z       = f["vel_z"][sample_indices]
                vel_y       = f["vel_y"][sample_indices]
                vel_x       = f["vel_x"][sample_indices] 

                # Create solid regions
                vel_z_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                vel_y_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                vel_x_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    vel_z_3d[b, k, j, i] = vel_z[b, :n]
                    vel_y_3d[b, k, j, i] = vel_y[b, :n]
                    vel_x_3d[b, k, j, i] = vel_x[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                vel_z_3d = torch.as_tensor(vel_z_3d, dtype=self.y_dtype, device=device)
                vel_y_3d = torch.as_tensor(vel_y_3d, dtype=self.y_dtype, device=device)
                vel_x_3d = torch.as_tensor(vel_x_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,   dtype=self.x_dtype, device=device)
                X        = X.unsqueeze(1)
                
                # Concat channels for output
                Y       = torch.stack([vel_z_3d, vel_y_3d, vel_x_3d], dim=1)
                del vel_z_3d, vel_y_3d, vel_x_3d
                
            else:
                
                # Load batch data
                vel_z = vel_y = vel_x = press = None 
                vel_z       = f["vel_z"][sample_indices]
                vel_y       = f["vel_y"][sample_indices]
                vel_x       = f["vel_x"][sample_indices] 
                press       = f["press"][sample_indices]

                # Create solid regions
                vel_z_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                vel_y_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                vel_x_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                press_3d = np.zeros((batch_size, D, H, W), dtype=np.float32)
                X   = np.zeros((batch_size, D, H, W), dtype=np.float32)
            
                # Fill porous regions
                for b in range(batch_size):
                    # Get coordinates to be filled
                    n = int(n_valid[b])
                    i = coori[b, :n]
                    j = coorj[b, :n]
                    k = coork[b, :n]
                    # Fill with data for each sample in the batch
                    vel_z_3d[b, k, j, i] = vel_z[b, :n]
                    vel_y_3d[b, k, j, i] = vel_y[b, :n]
                    vel_x_3d[b, k, j, i] = vel_x[b, :n]
                    press_3d[b, k, j, i] = press[b, :n]
                    X       [b, k, j, i] = edt  [b, :n]
                    
                # Turn into Pytorch tensors
                vel_z_3d = torch.as_tensor(vel_z_3d, dtype=self.y_dtype, device=device)
                vel_y_3d = torch.as_tensor(vel_y_3d, dtype=self.y_dtype, device=device)
                vel_x_3d = torch.as_tensor(vel_x_3d, dtype=self.y_dtype, device=device)
                press_3d = torch.as_tensor(press_3d, dtype=self.y_dtype, device=device)
                X        = torch.as_tensor(X,   dtype=self.x_dtype, device=device)
                X        = X.unsqueeze(1)
                
                # Concat channels for output
                Y       = torch.stack([vel_z_3d, vel_y_3d, vel_x_3d, press_3d], dim=1)
                del vel_z_3d, vel_y_3d, vel_x_3d, press_3d

        return X,Y
        

    