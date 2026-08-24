import gc
import torch
import os
import datetime

def clear_gpu_memory():
    """Limpa o cache de memória da GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def clear_cpu_memory():
    """Libera a memória utilizada por tensores da CPU."""
    for _ in range(10):
        gc.collect()

def delete_model(model):
    """Deleta o modelo PyTorch e libera a memória associada."""

    # Remover referências a submódulos e parâmetros
    for name, module in model.named_modules():
        del module
    for param in model.parameters():
        del param

    # Deletar a referência ao modelo
    del model

    # Limpar memória da GPU e CPU
    clear_gpu_memory()
    clear_cpu_memory()

    print("Modelo PyTorch e memória associada liberados.")
    
# Get the memory used for a model instance
def get_MB_storage_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return size_all_mb

# Get the number of parameters for a model instance
def get_n_trainable_params(model):
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
    return trainable_count
            
def get_n_non_trainable_params(model):
    nontrainable_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            nontrainable_count += param.numel()
    return nontrainable_count

def get_total_params(model):
    count = 0
    for name, param in model.named_parameters():
        count += param.numel()
    return count


def print_cuda_mem(tag=""):
    """Print current CUDA allocated and reserved memory in MB (no-op if no CUDA)."""
    if not torch.cuda.is_available():
        return
    device    = torch.device("cuda")
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved  = torch.cuda.memory_reserved(device)  / 1024**2
    print(f"[{tag}] allocated: {allocated:8.2f} MB | reserved: {reserved:8.2f} MB")


def profile_model_memory(model, example_input, tag="model"):

    # No CUDA: just run things on CPU if requested and return
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available; running on CPU (no GPU memory profile).")
        with torch.no_grad():
            _ = model(example_input)

    device = torch.device("cuda")

    # Clean slate for clearer numbers
    torch.cuda.empty_cache()

    print("=" * 60)
    print(f"Profiling CUDA memory for {tag}")
    print("=" * 60)

    print_cuda_mem("before model.to()")

    # Move model to GPU
    model = model.to(device)
    torch.cuda.synchronize()
    print_cuda_mem("after model.to()")

    # Forward pass
    example_input = example_input.to(device)

    with torch.no_grad():
        _ = model(example_input)
    torch.cuda.synchronize()
    print_cuda_mem("after one forward()")

    print("=" * 60)
