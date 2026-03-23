import torch
import numpy as np
import warnings
from typing import Any


from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

no_collate=lambda batch: batch


def tensor_transfomer(tensor):      return tensor.sign() * torch.log1p(tensor.abs())
def tensor_detransfomer(tensor):    return tensor.sign() * torch.expm1(tensor.abs())


def compute_loader_predictions(model, loader, N_Samples=None, shuffle=False):
    model.eval()
    if N_Samples is None: N_Samples = len(loader)
    
    #subset      = Subset(loader.dataset, range(N_Samples))    
    #new_loader  = DataLoader(subset, shuffle=shuffle)

    outputs     = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_output = model.predict(batch_inputs)

            outputs.append(batch_output)
            
            
    return outputs, loader
    
class Data_Loader(Dataset):
    
    def __init__(self, 
                 inputs_tensor, 
                 outputs_tensor, 
                 batch_size:    int         = None, 
                 shuffle:       bool        = False, 
                 deleteAfter:   bool        = True):
        
        
        self.inputs  = inputs_tensor.clone()
        if deleteAfter: del inputs_tensor

        self.outputs = outputs_tensor.clone()
        if deleteAfter: del outputs_tensor
        
        self.batch_size = batch_size if batch_size is not None else len(self.inputs)

        self.loader = DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)  # Automatically creates a DataLoader
        
    def transform_inputs(self, transformer):
        for idx in range(len(self.inputs)):
            self.inputs[idx]    = transformer(self.inputs[idx])
            
    def transform_targets(self, transformer):
        for idx in range(len(self.inputs)):
            self.outputs[idx]    = transformer(self.outputs[idx])
            
    def transform_data(self, transformer):
        self.transform_inputs(transformer)
        self.transform_targets(transformer)
        
        
    def get_splitted(self, train_ratio, val_ratio, max_samples=None, collate=None):
        train_dataset, val_dataset, test_dataset = self._split(
            train_ratio=train_ratio, val_ratio=val_ratio, max_samples=max_samples
        )
        x, y = train_dataset[0]
        
        if collate is None: collate=no_collate
        
        train_loader    = DataLoader(train_dataset, batch_size=min(max_samples, self.batch_size), collate_fn=collate, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=max_samples, collate_fn=collate, shuffle=False)    # No shuffle for validation
        test_loader     = DataLoader(test_dataset, batch_size=max_samples,  collate_fn=collate, shuffle=False)  # No shuffle for test
        
        return train_loader, val_loader, test_loader

    def print_stats(self, train_loader, val_loader, test_loader):
        
        total_samples = (
            len(train_loader.dataset) +
            len(val_loader.dataset) +
            len(test_loader.dataset)
        )
        # First batch for shape inspection
        train_first_batch = next(iter(train_loader))
        if len(train_first_batch) != 2: raise Exception(f"- Loader has {len(train_first_batch)} items and must have 2 (inputs and targets separately).")

        batch_input, batch_output = train_first_batch

        # Handle multi-scale or single-scale input
        if isinstance(batch_input, list):
            input_shape = [x.shape for x in batch_input]
            batch_size = batch_input[0].shape[0]
        elif isinstance(batch_input, torch.Tensor):
            input_shape = batch_input.shape
            batch_size = batch_input.shape[0]
        else:
            raise Exception("Batched input must be either a torch.Tensor or a list of torch.Tensors (multi-scale).")
    
        output_shape = batch_output.shape if isinstance(batch_output, torch.Tensor) else [y.shape for y in batch_output]
        
        
        
        # Print overall dataset/batch info
        print("=== Dataloader Summary ===")

        print(f"- Total samples considered: {total_samples}")
        print(f"  -- Train samples     : {len(train_loader.dataset)}")
        print(f"  -- Validation samples: {len(val_loader.dataset)}")
        print(f"  -- Test samples      : {len(test_loader.dataset)}")
        
        print(f"- Number of training batches: {len(train_loader)}")
        print("- Batch shape details: (batch_size, channels, depth, height, width)")
        print(f"  -- Train batch input shape : {input_shape}")
        print(f"  -- Train batch output shape: {output_shape}")
    
        # === Sanity checks ===
        print("=== Dataloader Sanity Check ===")
        val_first_batch = next(iter(val_loader))
        test_first_batch = next(iter(test_loader))
    
        val_input, val_output = val_first_batch
        test_input, test_output = test_first_batch
    
        if isinstance(val_input, list):
            val_input_shape = [x.shape for x in val_input]
            val_batch_size = val_input[0].shape[0]
        else:
            val_input_shape = val_input.shape
            val_batch_size = val_input.shape[0]
    
        if isinstance(test_input, list):
            test_input_shape = [x.shape for x in test_input]
            test_batch_size = test_input[0].shape[0]
        else:
            test_input_shape = test_input.shape
            test_batch_size = test_input.shape[0]
    
        # Compare batch counts
        if len(train_loader) == len(val_loader) == len(test_loader):
            print("- All loaders have the same number of batches.")
        else:
            raise Exception("Number of batches differ across loaders.")
    
        # Compare input shapes
        if input_shape == val_input_shape == test_input_shape:
            print("- Input shapes are consistent across loaders.")
        else:
            raise Exception("Input shapes differ across loaders.")
    
        # Compare output shapes
        if isinstance(output_shape, list):
            val_output_shape = [y.shape for y in val_output]
            test_output_shape = [y.shape for y in test_output]
        else:
            val_output_shape = val_output.shape
            test_output_shape = test_output.shape
    
        if output_shape == val_output_shape == test_output_shape:
            print("- Output shapes are consistent across loaders.")
        else:
            raise Exception("Output shapes differ across loaders.")
    
        # Compare batch sizes
        if batch_size == val_batch_size == test_batch_size:
            print(f"- Batch size is consistent across loaders: {batch_size}")
        else:
            raise Exception("Batch sizes differ across loaders.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def _split(self, train_ratio=0.7, val_ratio=0.15, max_samples=None):
        total_size = len(self)
        if total_size <= 3:
            raise Exception("TensorsDataset must have at least 3 samples in order to split")
    
    
        # Compute sizes ensuring at least 1 sample per dataset
        train_size = max(1, int(train_ratio * total_size))
        val_size = max(1, int(val_ratio * total_size))
        
        # Ensure remaining samples go to the test set
        test_size = max(1, total_size - (train_size + val_size))  
    
        # Adjust if rounding errors cause an overflow
        if train_size + val_size + test_size > total_size:
            train_size = total_size - (val_size + test_size)
    
        # Generate indices and split dataset    
        train_dataset = self._cut(train_size, start=0)
        val_dataset   = self._cut(val_size, start=train_size)
        test_dataset  = self._cut(test_size, start=train_size + val_size)
        
        if train_size > max_samples:
            train_dataset = self._cut(max_samples)
        if val_size > max_samples:
            val_dataset   = self._cut(max_samples)
        if test_size > max_samples:
            test_dataset  = self._cut(max_samples)

        return train_dataset, val_dataset, test_dataset
    
    
    def _cut(self, size=None, start=0):
        total_size = len(self.inputs)
    
        # handle empty dataset
        if total_size == 0:
            warnings.warn("Empty dataset; nothing to cut.")
            self.inputs  = self.inputs[0:0]
            self.outputs = self.outputs[0:0]
            return
        
        # start must be >= 0
        if start < 0:
            raise ValueError(f"Invalid start index: {start}. Must be >= 0.")
    
        # if start is beyond dataset, do nothing
        if start >= total_size:
            warnings.warn(
                f"Start index {start} is beyond dataset size ({total_size}). "
                "Dataloader will be left as is."
            )
            return
    
        # compute end
        if size is None:
            end = total_size
        else:
            if size <= 0:
                warnings.warn(
                    f"Non-positive size ({size}) passed to _cut; dataloader left as is."
                )
                return
            
            
            end = start + size
    
            # if end beyond dataset, truncate to total_size
            if end > total_size:
                warnings.warn(
                    f"Requested range [{start}:{end}) exceeds dataset size ({total_size}). "
                    f"Truncating end to {total_size}."
                )
                end = total_size
    
        # finally slice
        self.inputs  = self.inputs[start:end]
        self.outputs = self.outputs[start:end]
    
    
    