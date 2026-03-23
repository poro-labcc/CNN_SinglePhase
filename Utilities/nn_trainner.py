import torch
import numpy             as np
from   datetime          import datetime
import matplotlib.pyplot as plt
import pandas            as pd
import os.path
import json
import time
import random
import copy
import torch.nn          as nn
import os
from   Utilities           import usage_metrics as um


def full_train(model, 
               train_batch_loader,
               valid_loader,
               loss_functions,
               earlyStopping_loss,
               backPropagation_loss,
               optimizer,
               N_epochs             = 1000,
               scheduler            = None,
               weights_file_name    = "model_weights",
               results_folder       = "",
               device               = "cpu",
               dtype                = torch.float32):
    
    # Check the presence of 'predict' method
    if not hasattr(model, 'predict'):
        raise NotImplementedError("Model must implement 'predict' method for thresholded loss.")
    
    # Make sure to use same device
    model.to(device, dtype=dtype)
    
    # Initialize tracking
    average_computation = 0
    train_costs_h = []
    val_costs_h   = []
    best_valid_loss     = np.inf   
    train_avg_loss      = {}  
    valid_avg_loss      = {}
    best_model_path     = weights_file_name+"_LowerValidationLoss.pth"
    progress_points     = set(int(N_epochs * i / 100) for i in range(0, 101, 1))  
    best_model          = None

    
    print("\n\nTrainning session has started.\n")
    # Trainning process
    for epoch_index in range(N_epochs):
        
        epoch_timestamp_start = time.perf_counter()
        
        # Learn updating model
        print("Trainning")
        model = train_one_epoch( 
            model               = model,
            train_batch_loader  = train_batch_loader,
            loss_function       = loss_functions[backPropagation_loss], 
            optimizer           = optimizer,
            scheduler           = scheduler,
            device              = device,
            dtype               = dtype
        )
        
        # Get learning metrics
        print("Validating")
        train_avg_loss, valid_avg_loss = validate_one_epoch(
                                              model         = model, 
                                              train_loader  = train_batch_loader,
                                              valid_loader  = valid_loader,
                                              loss_functions= loss_functions,
                                              device        = device,
                                              dtype         = dtype
                                        )
        
        
        # Tracking performance
        train_costs_h.append(train_avg_loss)
        val_costs_h.append(valid_avg_loss)
        epoch_timestamp_stop    = time.perf_counter()
        diff                    = (epoch_timestamp_stop - epoch_timestamp_start)
        average_computation     += (diff - average_computation) / (epoch_index + 1)
        

        # Save tracking based on best performance
        if valid_avg_loss[earlyStopping_loss] < best_valid_loss:
            percent         = round((1-(valid_avg_loss[earlyStopping_loss] / best_valid_loss)) * 100, 2)
            best_valid_loss = valid_avg_loss[earlyStopping_loss]
            best_model      = copy.deepcopy(model.state_dict())
            print(f"--> New best solution for Validation dataset achieved at {epoch_index} / {N_epochs}: {best_valid_loss} ({percent:.6f}% better)")
            torch.save(best_model, best_model_path)
        
        # Save tracking based on % of epochs
        if epoch_index in progress_points:
            # Make tracking prints
            percent = round((epoch_index / N_epochs) * 100, 2)  # Calcula o percentual relativo
            print(f"\nExecuting epoch {epoch_index} / {N_epochs} ({percent:.1f}%)")
            print("--> Allocated memory {} (MB) ".format(round(um.get_memory_usage())))
            print("--> Average epoch processing time (seconds): ", round(average_computation,6))
            print(f"--> Back-Propagated Loss for trainning vs validation data: {train_avg_loss[backPropagation_loss]:.4f}, {valid_avg_loss[backPropagation_loss]:.4f}")
            # Create a directory and file name for model
            model_path = weights_file_name+"_ProgressTracking_{}.pth".format(round((epoch_index / N_epochs) * 100))
            model_dir = os.path.dirname(weights_file_name)
            if model_dir: os.makedirs(model_dir, exist_ok=True)
            # Save model state
            torch.save(model.state_dict(), model_path)                              # Save current model
            # Plot loss 
            Plot_LossHistory(train_costs_h, val_costs_h, output_path=f"{results_folder}LossHistory")
            

#### PARTIAL TRAINNING

        
def partial_train(model, 
               train_batch_loader,
               valid_loader,
               loss_functions,
               earlyStopping_loss,
               backPropagation_loss,
               optimizer,
               partial_epochs       = 100,
               N_epochs             = 1000,
               scheduler            = None,
               results_folder       = "",
               device               = "cpu",
               patience             = None,
               dtype                = torch.float32):
    
    # Check the presence of 'predict' method
    if not hasattr(model, 'predict'):
        raise NotImplementedError("Model must implement 'predict' method for thresholded loss.")
    # Check the folder 
    if results_folder:
        results_folder = results_folder.rstrip("/") + "/"
        os.makedirs(results_folder, exist_ok=True)
    
    # Make sure to use same device
    model.to(device, dtype=dtype)
        
    # RESTORE CHECK POINT
    last_update, start_epoch, train_costs_h, val_costs_h, best_valid_loss = resume_checkpoint(results_folder, 
                                                                                              model, optimizer, 
                                                                                              scheduler, device)
        
    average_computation = 0
    train_avg_loss      = {}  
    valid_avg_loss      = {}
    best_model          = None
    if patience is None: patience = N_epochs

    # Create a directory and file name for model
    weights_file_name   = results_folder+"model"
    best_model_path     = weights_file_name+"_LowerValidationLoss.pth"
    
    # Trainning process
    end_epoch = min(start_epoch+partial_epochs, N_epochs)
    if start_epoch >= end_epoch: raise Exception("Partial trainning not feasible, the trainning has ended already.")
    print("\n\nTrainning session has started.\n")
    progress_points     = set(int(N_epochs * i / 100) for i in range(0, 101, 1))  
    for epoch_index in range(start_epoch, end_epoch):
        
        epoch_timestamp_start = time.perf_counter()
        
        # Learn updating model
        model = train_one_epoch( 
            model               = model,
            train_batch_loader  = train_batch_loader,
            loss_function       = loss_functions[backPropagation_loss], 
            optimizer           = optimizer,
            scheduler           = scheduler,
            device              = device,
            dtype               = dtype
        )
        
        # Get learning metrics
        train_avg_loss, valid_avg_loss = validate_one_epoch(
                                              model         = model, 
                                              train_loader  = train_batch_loader,
                                              valid_loader  = valid_loader,
                                              loss_functions= loss_functions,
                                              device        = device,
                                              dtype         = dtype
                                        )
        
        
        # Tracking performance
        train_costs_h.append(train_avg_loss)
        val_costs_h.append(valid_avg_loss)
        epoch_timestamp_stop    = time.perf_counter()
        diff                    = (epoch_timestamp_stop - epoch_timestamp_start)
        average_computation     += (diff - average_computation) / (epoch_index + 1)
        

        # Save tracking based on best performance
        if valid_avg_loss[earlyStopping_loss] < best_valid_loss:
            percent         = round((1-(valid_avg_loss[earlyStopping_loss] / best_valid_loss)) * 100, 2)
            last_update     = epoch_index
            best_valid_loss = valid_avg_loss[earlyStopping_loss]
            best_model      = copy.deepcopy(model.state_dict())
            print(f"--> New best solution for Validation dataset achieved at {epoch_index} / {N_epochs}: {best_valid_loss} ({percent:.6f}% better)")
            atomic_torch_save(best_model, best_model_path)
            
        
        # Save tracking based on % of epochs
        if epoch_index in progress_points:
            # Make tracking prints
            percent     = round((epoch_index / N_epochs) * 100, 2)  # Calcula o percentual relativo
            print(f"\nExecuting epoch {epoch_index} / {N_epochs} ({percent:.1f}%)")
            print("--> Average epoch processing time (seconds): ", round(average_computation,6))
            print("--> Back-Propagated Loss for trainning vs validation data:")
            max_len = max([len(k) for k in train_avg_loss.keys()]) + 2
            print(f"{'Loss Name':<{max_len}} {'Train':<10} | {'Valid':<10}")
            print("-" * (max_len + 25))
            for loss_name in train_avg_loss.keys():
                t_loss = train_avg_loss[loss_name]
                v_loss = valid_avg_loss[loss_name]
                print(f"{loss_name:<{max_len}} {t_loss:<10.6f} | {v_loss:<10.6f}")
            
            # Plot loss 
            Plot_LossHistory(train_costs_h, val_costs_h, output_path=f"{results_folder}LossHistory.png")
            # Save trainning state on checkpoint
            checkpoint_data = {
                'epoch':                epoch_index,
                'last_update':          last_update,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_costs_h':  train_costs_h,
                'val_costs_h':    val_costs_h,
                'best_valid_loss':      best_valid_loss,
                'rng_state':            torch.get_rng_state().cpu().byte(),
                'cuda_rng_state': (
                                        [t.cpu().byte() for t in torch.cuda.get_rng_state_all()] 
                                        if torch.cuda.is_available() else None
                                  )
            }
            atomic_torch_save(checkpoint_data, results_folder+f"train_checkpoint_{epoch_index}")
            
        if epoch_index - last_update > patience: 
             print(f"Patience of {patience} achieved (last update in epoch {last_update}). Resuming training...")
             break
    # Save trainning state on ending
    checkpoint_data = {
        'epoch':                epoch_index,
        'last_update':          last_update,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_costs_h':  train_costs_h,
        'val_costs_h':    val_costs_h,
        'best_valid_loss':      best_valid_loss,
        'rng_state':            torch.get_rng_state().cpu().byte(),
        'cuda_rng_state': (
                                [t.cpu().byte() for t in torch.cuda.get_rng_state_all()] 
                                if torch.cuda.is_available() else None
                          )
    }
    atomic_torch_save(checkpoint_data, results_folder+f"train_checkpoint_{epoch_index}")
    


    
def train_one_epoch(model, train_batch_loader, loss_function, optimizer, scheduler, dtype, device='cpu'):
    
    
    # Set model to train mode
    model.train()
    
    # For each batch
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_batch_loader):
        if not torch.isfinite(batch_inputs).all():  raise Exception("NaN/Inf in inputs.")
        if not torch.isfinite(batch_targets).all(): raise Exception("NaN/Inf in targets.")
    
        # Make the batch
        if isinstance(batch_inputs, (list, torch.Tensor)):
            # Load batch data on device
            batch_inputs    = move_to_device(batch_inputs,  device, dtype)
            batch_targets   = move_to_device(batch_targets, device, dtype)
            
            optimizer.zero_grad()                   # Reset gradients 
            
            batch_outputs   = model(batch_inputs)   # Get model outputs
            
            loss            = loss_function["obj"](batch_outputs, batch_targets) # Get output loss
            
            loss.backward()                         # Get gradient of the loss with respect to the model
            
            optimizer.step()                        # Update the model weights with respect to the gradient
            
        
        
    if scheduler is not None: scheduler.step() # Realiza um passo no learning rate

    return  model

def validate_one_epoch(model, train_loader, valid_loader, loss_functions, device, dtype):
    return get_loader_loss(model, train_loader, loss_functions, device, dtype), get_loader_loss(model, valid_loader, loss_functions, device, dtype)
    

def get_loader_loss(model, loader, loss_functions, device, dtype):
    
    results = {loss_name: 0.0 for loss_name in loss_functions}
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
        
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      
      batch_count = 0
      # For each batch listed in loader
      for batch_inputs, batch_targets in loader:
        
        # If batch's samples are listed: iterate the list
        if not (isinstance(batch_inputs, list) or isinstance(batch_inputs, torch.Tensor)): 
            raise Exception(f"The train batch loader must be one of torch.Tensor of list but got type {type(batch_inputs)}.")
            
        # Load batch data on device
        batch_inputs    = move_to_device(batch_inputs, device, dtype)
        batch_targets   = move_to_device(batch_targets, device, dtype)
        
        # For each loss function listed
        for loss_name, loss_function in loss_functions.items():
            
            # If the loss function is in thresholded mode: use predict mode to get output
            if loss_function["Thresholded"]: batch_outputs = model.predict(batch_inputs)
            
            # If the loss function is NOT in thresholded mode: use forward mode to get output
            else: batch_outputs = model(batch_inputs)
            
            # Compute loss based on output
            aux  = loss_function["obj"](batch_outputs, batch_targets).item()
            results[loss_name] += aux
        batch_count        += 1
        
      # Mean loss over batchs
      if batch_count > 0: 
          for loss_name in results: results[loss_name] /= batch_count
    return results



#######################################################
#********************* INITIALIZERS ******************#
#######################################################

def init_weights_last_conv_ones(model):
    """
    Zero initialize most layers, but set last conv layer's bias to ones
    This makes the last conv layer output start at ~1.0 before activation
    """
    # Find the last conv layer
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            conv_layers.append((name, module))
    
    last_conv_name = None
    last_conv_module = None
    if conv_layers:
        last_conv_name, last_conv_module = conv_layers[-1]
        print(f"Last conv layer found: {last_conv_name}")
    
    # Apply initialization
    for name, module in model.named_modules():
        # For conv layers
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            
            # Zero initialize weights for ALL conv layers
            nn.init.zeros_(module.weight)
            
            # Initialize bias
            if module.bias is not None:
                if module is last_conv_module:
                    # Last conv: set bias to ones
                    nn.init.ones_(module.bias)
                    print(f"Initialized {name}.bias to ones")
                else:
                    # Other conv layers: zero bias
                    nn.init.zeros_(module.bias)
        
        # For linear layers (if you have them)
        elif isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # For normalization layers
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                               nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
def init_weights_zeros(m):

    # Conv layers: zero weights and bias
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                     nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.ones_(m.bias)
            
    # Linear layers: zero weights and bias  
    elif isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.ones_(m.bias)
            
    # Normalization layers: keep defaults (ones for weight, zeros for bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                       nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                       nn.InstanceNorm2d, nn.InstanceNorm3d)):
        # These will use PyTorch defaults, but we're explicit for clarity
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    # Embedding layers: zero initialize
    elif isinstance(m, nn.Embedding):
        nn.init.zeros_(m.weight)
        
def init_weights_xavier(m):

    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        # Keras default: glorot_uniform
        nn.init.xavier_uniform_(m.weight)
        # Keras default: zeros
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                       nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                       nn.InstanceNorm2d, nn.InstanceNorm3d)):
        # Keras and PyTorch both default to gamma=1, beta=0, 
        # but it is good practice to be explicit if enforcing strict consistency.
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_he(m):

    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # Keras default: glorot_uniform
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        # Keras default: zeros
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
       
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                       nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                       nn.InstanceNorm2d, nn.InstanceNorm3d)):
        # Keras and PyTorch both default to gamma=1, beta=0, 
        # but it is good practice to be explicit if enforcing strict consistency.
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


    
#######################################################
#****************** AUXILIARY FUNCTIONS **************#
#######################################################

# Move tensors to device
# If obj is a list, move each element to device
def move_to_device(obj, device, dtype):
    if isinstance(obj, (list, tuple)):
        return [move_to_device(item, device, dtype) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, dtype) for k, v in obj.items()}
    else:
        return obj.to(device, dtype=dtype)
    
def atomic_torch_save(model, path):
    temp_path = path + ".tmp"    
    torch.save(model, temp_path)
    if os.path.exists(path):
        os.remove(path) 
    os.rename(temp_path, path)
    
def resume_checkpoint(folder, model, optimizer, scheduler, device):
    latest_checkpoint_path  = get_latest_checkpoint (folder,    "train_checkpoint_")
    if latest_checkpoint_path is not None:
        print(f"Loading Checkpoint from {latest_checkpoint_path}...")
        checkpoint          = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch         = checkpoint['epoch'] + 1 # Start from the NEXT epoch
        last_update         = checkpoint['last_update']
        train_costs_h       = checkpoint['train_costs_h']
        val_costs_h         = checkpoint['val_costs_h']
        best_valid_loss     = checkpoint['best_valid_loss']
            
        rng_state           = checkpoint['rng_state']
        rng_state           = rng_state.cpu().to(torch.uint8)
        torch.set_rng_state(rng_state)
       
        if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            cuda_rng_state = checkpoint['cuda_rng_state']
            try:
                clean_cuda_rng = [t.cpu().to(torch.uint8) for t in cuda_rng_state]
                torch.cuda.set_rng_state_all(clean_cuda_rng)
            except RuntimeError as e:
                    print(f"Warning: Could not load CUDA RNG state (likely GPU count mismatch): {e}")
                    
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print(f"No previous checkpoints found in {folder}")
        start_epoch         = 0
        train_costs_h = []
        val_costs_h   = []
        best_valid_loss     = np.inf  
        last_update         = 0
        
    return last_update, start_epoch, train_costs_h, val_costs_h, best_valid_loss

def get_latest_checkpoint(folder_path, prefix="train_checkpoint_"):
    """
    Scans a folder for files matching 'train_checkpoint_{NUMBER}' 
    (with or without .pth extension) and returns the path to the 
    one with the highest NUMBER.
    """
    
    folder_path = folder_path if folder_path else "."
    
    if not os.path.exists(folder_path):
        return None

    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return None

    latest_file = None
    max_value   = -1
    for filename in files:
        # Check if file matches the specific prefix
        if filename.startswith(prefix):
            
            try:
                # 1. Remove file extension (e.g., .pth) if it exists
                name_no_ext = os.path.splitext(filename)[0]
                # 2. Extract the number part
                number_part = name_no_ext.split(prefix)[-1]
                # 3. Convert to integer
                val = int(number_part)
                # 4. Update if this is the highest number seen so far
                if val > max_value:
                    max_value = val
                    latest_file = filename
                    
            except ValueError:
                # Skip files that match the prefix but don't end in a valid number
                # e.g., "train_checkpoint_temp"
                continue 

    if latest_file:
        return os.path.join(folder_path, latest_file)
    
    return None

def load_model_from_checkpoint(model, folder_path, epoch, device='cpu'):

    checkpoint_path = os.path.join(folder_path, f"train_checkpoint_{epoch}")
    
    # Check if file exists with or without .pth extension
    if not os.path.exists(checkpoint_path):
        if os.path.exists(checkpoint_path + ".pth"):
            checkpoint_path += ".pth"
        else:
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {folder_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint

def set_global_seed(seed: int, deterministic_strict: bool = False):
    """
    Sets seeds for reproducibility across PyTorch, NumPy, and Python.
    """
    # 1. Python environment
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    
    # 4. CuDNN backend options
    if deterministic_strict:
        torch.backends.cudnn.deterministic  = True
        torch.backends.cudnn.benchmark      = False
    
    print(f"Global seed set to {seed}")


def create_training_data_folder(base_dir: str = None):
    # Use current working directory if base_dir not provided
    if base_dir is None:
        base_dir = os.getcwd()

    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Get the current date and time
    now = datetime.now()
    
    # Format the date string
    date_str = f"{now.day}_" + now.strftime("%B_%Y_%I-%M%p")
    base_folder_name = f"NN_Trainning_{date_str}"
    
    # Initial path attempt
    new_folder_path = os.path.join(base_dir, base_folder_name)
    
    slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID') or os.environ.get('SLURM_JOB_ID', 'local')
    new_folder_path = new_folder_path+f"_Job{slurm_id}"

    # --- Collision Handling Logic ---
    counter = 1
    # Keep looping as long as the directory exists
    while os.path.exists(new_folder_path):
        # Update the name to include (1), (2), etc.
        unique_name = f"{base_folder_name}({counter})"
        new_folder_path = os.path.join(base_dir, unique_name)
        counter += 1
    # --------------------------------

    try:
        os.makedirs(new_folder_path, exist_ok=True)
        # Using os.sep ensures the trailing slash works on Windows and Linux
        return new_folder_path + os.sep
    except OSError as error:
        print(f"Error creating folder: {error}")
        return None
    
    
    
    
def save_metadata(model_name, 
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
                  train_comment
                  ):
    
    # Defina o caminho onde o arquivo será salvo
    metadata_file = os.path.join(NN_results_folder, "metadata.txt")
    
    # Monte o conteúdo do metadata
    metadata_content = f"""
    ================= TRAINING METADATA =================
    
    Model Aspects:
    - model_name:           {model_name}
    - model weigth init:    {weight_init}
    - binary input:         {binary_input}
    
    Data Aspects:
    - NN_dataset_folder:    {NN_dataset_folder}
    - dataset_train_name:   {dataset_train_name}
    - dataset_valid_name:   {dataset_valid_name}
    - batch_size:           {batch_size}
    
    Learning Aspects:
    - N_epochs:             {N_epochs}
    - patience:             {patience}
    - learning_rate:        {learning_rate}
    - optimizer:            {optimizer}
    - earlyStopping_loss:   {earlyStopping_loss}
    - backPropagation_loss: {backPropagation_loss}
    
    Loss Functions:
    {json.dumps({k: {"Thresholded": v["Thresholded"], "obj": str(v["obj"])} for k,v in loss_functions.items()}, indent=4)}
    
    Paths:
    - NN_results_folder:        {NN_results_folder}
    - NN_model_weights_folder:  {NN_model_weights_folder}
    - model_full_name:          {model_full_name}
    - dataset_train_full_name:  {dataset_train_full_name}
    - dataset_valid_full_name:  {dataset_valid_full_name}
    
    ======================================================
    
    """+train_comment
    
    # Escreve no txt
    with open(metadata_file, "w") as f:
        f.write(metadata_content)
        
    return metadata_file

#######################################################
#********************* PLOTTERS **********************#
#######################################################

def Plot_LossHistory(train_cost_h, val_cost_h, normalize=False, output_path='loss_history.svg'):
    
    
    # 1. Configure Matplotlib for a Professional/Scientific Look
    plt.rcParams.update({
        'font.family': 'serif',       
        'font.size': 12,              
        'axes.labelsize': 14,         
        'axes.titlesize': 16,         
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.2,        
        'lines.linewidth': 2.0        
    })
    # Updated Color Palette (Solid Blue, Black, Grey)
    c_train = '#005b96'  # A rich, professional solid blue
    c_valid = '#000000'  # Solid black
    c_diff  = '#757575'  # Solid medium grey
    
    # Extract training history
    train_h_dicts = {loss_name: [] for loss_name in train_cost_h[0]}
    for epoch_dict in train_cost_h:
        for loss_name in epoch_dict:
            train_h_dicts[loss_name].append(epoch_dict[loss_name])
            
    # Extract validation history
    valid_h_dicts = {loss_name: [] for loss_name in val_cost_h[0]}
    for epoch_dict in val_cost_h:
        for loss_name in epoch_dict:
            valid_h_dicts[loss_name].append(epoch_dict[loss_name])
            
    num_plots = len(train_h_dicts)
    fig, axes = plt.subplots(num_plots, 3, figsize=(16, 4.5 * num_plots))
    if num_plots == 1:
        axes = np.array([axes])
        
    def scale(data):
        max_val = np.max(data)
        return data / max_val if max_val != 0 else data

    total_epochs = len(train_cost_h)
    n_recent = max(1, int(0.2 * total_epochs))
    
    for idx, loss_name in enumerate(train_h_dicts.keys()):
        train_loss_h = train_h_dicts[loss_name]
        valid_loss_h = valid_h_dicts[loss_name]
        
        if normalize:
            train_loss_h = scale(train_loss_h)
            valid_loss_h = scale(valid_loss_h)
            
        loss_difference = np.array(valid_loss_h) - np.array(train_loss_h)

        # Pre-calculate ranges
        x_full = range(len(train_loss_h))
        n_r = min(n_recent, len(train_loss_h))
        x_recent = x_full[-n_r:]
        
        # Plot 1: Full cost history
        ax1 = axes[idx, 0]
        ax1.plot(x_full, train_loss_h, label='Train', color=c_train)
        ax1.plot(x_full, valid_loss_h, label='Validation', color=c_valid)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Cost History\n({loss_name})')
        ax1.set_ylim(bottom= min(0.0, np.min(train_loss_h), np.min(valid_loss_h)))

        # Plot 2: Recent history
        ax2 = axes[idx, 1]
        ax2.plot(x_recent, train_loss_h[-n_r:], label='Train', color=c_train)
        ax2.plot(x_recent, valid_loss_h[-n_r:], label='Validation', color=c_valid)
        ax2.set_xlabel('Epochs')
        ax2.set_title(f'Recent History (Last {n_r})\n({loss_name})')
        ax2.set_ylim(bottom= min(0.0, np.min(train_loss_h[-n_r:]), np.min(valid_loss_h[-n_r:])))

        # Plot 3: Loss Difference
        ax3 = axes[idx, 2]
        ax3.plot(x_recent, loss_difference[-n_r:], label='Val - Train', color=c_diff)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.5) 
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Difference')
        ax3.set_title(f'Loss Difference\n({loss_name})')

        # Apply clean aesthetic to all subplots
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, loc='best')

    plt.tight_layout(pad=2.0)
    
    # Enforce SVG saving
    if output_path is not None:
        if not output_path.lower().endswith('.svg'):
            output_path += '.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    