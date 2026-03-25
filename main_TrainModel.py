import torch.nn as nn
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import dataset_reader as dr
from Architectures import Models

#######################################################
#************ USER INPUTS:                 ***********#
#######################################################

# Model Aspects
model_name              = "danny_z"   # The desired model name, one of: javier_z, danny_z, inception_z
binary_input            = True

# Data aspects
NN_dataset_folder       = "../NN_Datasets/"
dataset_train_name      = "PressureDriven/Train_Danny_120_120_120_Pressure.h5" 
dataset_valid_name      = "PressureDriven/Train_Danny_120_120_120_Pressure.h5" 
train_range             = (0,8)  # Not Augmented: (0,7); Augmented (0,216) 
valid_range             = (8,9)  # Not Augmented: (7,9); Augmented (216,216+26)
batch_size              = 8      # Group size of train samples that influence one update on weights
num_workers             = 4      # How many python process to load next batch's data
num_threads             = 18     # How many cpu parallel computations

# Learning aspects
N_epochs                = 10     # Not Augmented: 30000; Augmented 1000
partial_epochs          = 10     # Not Augmented: 30000; Augmented 1000
patience                = 5      # Not Augmented: 6000;  Augmented 200
learning_rate           = 0.0006    
loss_functions  = {
    # Optimization Loss Functions:          "Thresholded" = False, to evaluate the outputs 
    "MSE":              {"obj": nn.MSELoss(),                        "Thresholded": False},
    # Perfomance analysis Loss Functions:   "Thresholded" = True, to evaluate in final prediction mode
    "PRPE":             {"obj": lf.PRPE(),                           "Thresholded": True},
    "PearsonCorr":      {"obj": lf.Mask_LossFunction(lf.PearsonCorr(2000)),  "Thresholded": True},
    "MSE_inVoid":       {"obj": lf.Mask_LossFunction(nn.MSELoss()),  "Thresholded": True}, 
}

earlyStopping_loss      = "MSE"       # Which listed loss_function is used to stop trainning
backPropagation_loss    = "MSE"       # Which listed loss_function is used to calculate weights
optimizer               = 'ADAM'      # One of: 'ADAM' or 'SGD' 
weight_init             = None        # One of: 'He', 'Xavier' or None (Default)
device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed                    = 42
dtype                   = torch.float32
train_comment           = "Helpful comment" # Exemplo: "Arq javier, dada from danny Z. OBJ: Did it converge?"



# Create a new one or pass from where you want to restart
#NN_results_folder       = "/home/gabriel/Desktop/Dissertacao/NN_Results/NN_Trainning_4_March_2026_06-36PM_Joblocal/"
NN_results_folder       = nnt.create_training_data_folder(base_dir="../NN_Results")



#######################################################
#************ METADATA REGISTER **********************#
#######################################################    
NN_model_weights_folder = NN_results_folder+"NN_Model_Weights/"
model_full_name         = NN_model_weights_folder+model_name
dataset_train_full_name = NN_dataset_folder+dataset_train_name
dataset_valid_full_name = NN_dataset_folder+dataset_valid_name
print(f"Optimizing with: {backPropagation_loss} for {optimizer}")
print()
print("Folder created for results: ",NN_results_folder)
print("Saving model weights in:    ",NN_model_weights_folder)
print("Model base name:            ",model_name)
print("Binary input:               ",binary_input)
print("Weights initialization:     ",weight_init)
print(f"Trainning with dataset ({train_range}):     ",dataset_train_full_name)
print(f"Validating with dataset({valid_range}):     ",dataset_valid_full_name)
print()
print("Batch_size:                 ",batch_size)
print("N_epochs:                   ",N_epochs)
print("patience:                   ",patience)
print("learning_rate:              ",learning_rate)
print("optimizer:                  ",optimizer)
print("device:                     ",device)
print("seed:                       ",seed)
print("earlyStopping_loss:         ",earlyStopping_loss)
print("backPropagation_loss:       ",backPropagation_loss)
print()
metadata_file = nnt.save_metadata(model_name, 
                      NN_dataset_folder,
                      dataset_train_name,
                      dataset_valid_name,
                      batch_size,
                      N_epochs,
                      patience,
                      learning_rate,
                      optimizer,
                      weight_init,
                      binary_input,
                      earlyStopping_loss,
                      backPropagation_loss,
                      loss_functions,
                      NN_results_folder,
                      NN_model_weights_folder,
                      model_full_name,
                      dataset_train_full_name,
                      dataset_valid_full_name,
                      train_comment)
print(f"Metadata saved at: {metadata_file}")



#######################################################
#************ LOADING DATA          ******************#
#######################################################

# Set seed to random initializations
nnt.set_global_seed(seed) 

print("Loading Trainning Data ... ")
train_ds = dr.LazyDatasetTorch(h5_path=dataset_train_full_name, 
                               list_ids=np.arange(train_range[0],train_range[1]), 
                               x_dtype=torch.float32,
                               y_dtype=torch.float32)

valid_ds = dr.LazyDatasetTorch(h5_path=dataset_valid_full_name, 
                               list_ids=np.arange(valid_range[0],valid_range[1]), 
                               x_dtype=torch.float32,
                               y_dtype=torch.float32)



#######################################################
#******************** MODEL **************************#
#######################################################

print("Loading Model ... ")

if model_name=="javier_z":
    model       = Models.Corrected_MS_Net()
    # Restrict
    train_ds.uni_directional = 0
    valid_ds.uni_directional = 0
    # Make loss function multiscale 
    for loss_name, items in loss_functions.items():
        # If it is not prediction mode
        if not items["Thresholded"]: 
            loss_functions[loss_name]["obj"] = lf.MultiScaleLoss(loss_functions[loss_name]["obj"], norm_mode='var')
    
elif model_name=="danny_z":
    #model_aux   = Models.DannyKo_Net()
    model_aux   = Models.DannyKo_Net_Original()
    model       = model_aux.z_model
    # Restrict
    train_ds.uni_directional = 0
    valid_ds.uni_directional = 0
    
    model.bin_input =binary_input
    
elif model_name=="danny_y":
    model_aux   = Models.DannyKo_Net_Original()
    model       = model_aux.y_model
    # Restrict
    train_ds.uni_directional = 1
    valid_ds.uni_directional = 1
    
    model.bin_input =binary_input
    
elif model_name=="danny_x":
    model_aux   = Models.DannyKo_Net_Original()
    model       = model_aux.x_model
    # Restrict
    train_ds.uni_directional = 2
    valid_ds.uni_directional = 2
    
    model.bin_input =binary_input
    
elif model_name=="danny_zyx":
    model = Models.DannyKo_Net_Original()
    
    model.bin_input =binary_input
    
else:
    raise Exception(f"Specified model {model_name} is not defined.")

# Weights initialization
if   weight_init in ('Xavier','xavier','XAVIER'):  model.apply(nnt.init_weights_xavier)
elif weight_init in ('He','he','HE'):              model.apply(nnt.init_weights_he)
elif weight_init is None or weight_init in ('None', 'none', 'NONE'): pass
elif weight_init in ('Zero', 'Zeros', 'zero', 'zeros', 'ZERO', 'ZEROS'): model.apply(nnt.init_weights_zeros)
else: raise(f"Weights initialization mode {weight_init} not implemented.")
        
print('Model size: {:.3f}MB'.format(mh.get_MB_storage_size(model)))
print('Model size: {} parameters'.format(mh.get_n_trainable_params(model)))




#######################################################
#************ OPTIMIZER    ***************************#
####################################################### 
if      optimizer == 'ADAM':    optimizer = torch.optim.Adam (model.parameters(), lr=learning_rate)
elif    optimizer == 'ADAMW':   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif    optimizer == 'SGD':     optimizer = torch.optim.SGD  (model.parameters(), lr=learning_rate)
else:   raise Exception(f"Optimizer {optimizer} is not implemented.")
    

#######################################################
#************ COMPUTATIONS ***************************#
#######################################################

# Create dataloader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


print(f"Starting Train on {device}... \n")

torch.set_num_threads(num_threads)

nnt.partial_train(
    model, 
    train_loader,
    valid_loader,
    loss_functions,
    earlyStopping_loss,
    backPropagation_loss,
    optimizer,
    partial_epochs       = partial_epochs,
    N_epochs             = N_epochs,
    scheduler            = None,
    results_folder       = NN_results_folder,
    device               = device,
    patience             = patience,
    dtype                = torch.float32
    )
print("Ending Train ... ")



### DELETE MODEL AFTER USING IT
mh.delete_model(model)
del train_loader
del valid_loader

