import torch
import torch.nn as nn
import numpy as np
from .Functional import pad_same, crop_same, Channel_Concat
import copy

"""
This class implements a composite architecture that generates a final output
by aggregating predictions from a suite of specialized, pre-trained sub-models.
"""
class SubModels_Composition(nn.Module):
    def __init__(self, main_model, z_name, y_name, x_name, p_name, device, is_eval=False):
        super().__init__() 
        
        # Check attributes
        for attr in ['z_model', 'y_model', 'x_model', 'p_model']:
            if not (hasattr(main_model, attr) and isinstance(getattr(main_model, attr), nn.Module)): 
                raise AttributeError(f"Provided main_model is missing required attribute: {attr}")
        
        # Deepcopy to avoid mutating the original main_model's weights
        self.z_model = copy.deepcopy(main_model.z_model)
        self.y_model = copy.deepcopy(main_model.y_model)
        self.x_model = copy.deepcopy(main_model.x_model)
        self.p_model = copy.deepcopy(main_model.p_model)
        
        # Load pre-trained sub-models safely into the copies
        self.z_model.load_state_dict(torch.load(z_name, map_location=torch.device(device), weights_only=True))
        self.y_model.load_state_dict(torch.load(y_name, map_location=torch.device(device), weights_only=True))
        self.x_model.load_state_dict(torch.load(x_name, map_location=torch.device(device), weights_only=True))
        self.p_model.load_state_dict(torch.load(p_name, map_location=torch.device(device), weights_only=True))
        
        self.concat = Channel_Concat()
        
        
    def forward(self, x):
        
        with torch.no_grad():
            x_out = self.x_model.predict(x)
            y_out = self.y_model.predict(x)
            z_out = self.z_model.predict(x)
            p_out = self.p_model.predict(x)
                
        return self.concat(z_out, y_out, x_out, p_out)
    
    def predict(self, x):
                
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
        
