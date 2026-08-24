import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps  # Necessário para o local_thickness
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter

# Importe os seus leitores e utilitários
from Utilities import dataset_reader as dr
from Utilities import visu_utils as vu # Assumindo que sua função tensor_denorm está aqui


def plot_reynolds_distribution(datapath: str, 
                               property_label: str = r"Mean Local Reynolds $Re$", 
                               batch_size: int = 4, 
                               bins: int = 15, 
                               save_path: str = None,
                               lim = None):
    """
    Itera sobre o dataset HDF5, calcula a espessura local, desnormaliza a 
    velocidade e calcula o Número de Reynolds local 3D na região dos poros.
    """
    
    print(f"Loading dataset from: {datapath}")
    dataset = dr.LazyDatasetTorch(h5_path=datapath, 
                                  list_ids=None, 
                                  x_dtype=torch.float32,
                                  y_dtype=torch.float32)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    sample_means = []
    sample_maxs = [] # Útil para LBM: verificar se o Re máximo quebra o regime de Stokes
    
    print("Extracting local Reynolds from samples (this may take a while due to local_thickness)...")
    with torch.no_grad():
        for batch_inp, batch_tar in loader:
            B = batch_inp.shape[0]
            
            for b in range(B):
                # 1. Extrair a máscara Booleana do poro
                # batch_inp shape: (B, C_in, Z, Y, X). C_in = 0 é assumido como geometria/EDT
                mask_tensor = batch_inp[b, 0] > 0
                mask_np = mask_tensor.cpu().numpy()
                
                # 2. Calcular o Local Thickness (L) usando porespy
                # O porespy retorna o raio da esfera inscrita máxima
                local_thickness = ps.filters.local_thickness(mask_np)
                
                # 3. Desnormalizar o campo alvo (mantendo a dimensão do batch = 1 para a função vu)
                # target shape passa a ser (1, C, Z, Y, X)
                denorm_tar = vu.tensor_denorm(batch_tar[b:b+1], batch_inp[b:b+1])
                
                # 4. Calcular a Magnitude da Velocidade 3D
                # Assumindo canais 0, 1 e 2 como v_z, v_y, v_x
                v_z = denorm_tar[0, 0]
                v_y = denorm_tar[0, 1]
                v_x = denorm_tar[0, 2]
                vel_mag = torch.sqrt(v_z**2 + v_y**2 + v_x**2).cpu().numpy()
                
                # 5. Calcular o Reynolds 3D
                mu = 1.0 / 3.0
                re_local = vel_mag * local_thickness / mu
                
                # 6. Extrair os valores apenas da região fluida
                valid_re = re_local[mask_np]
                
                if valid_re.size > 0:
                    sample_means.append(valid_re.mean())
                    sample_maxs.append(valid_re.max())
                else:
                    print(f"Warning: Sample in batch has no fluid voxels.")
                    
    sample_means = np.array(sample_means)
    sample_maxs = np.array(sample_maxs)
    
    if len(sample_means) == 0:
        raise ValueError("No valid data collected. Check your dataset and mask logic.")

    mu_val = np.mean(sample_means)
    sigma_val = np.std(sample_means)
    max_global = np.max(sample_maxs)
    n_samples = len(sample_means)
    
    print(f"Collected {n_samples} samples. Mean Re: {mu_val:.4e}, Global Max Re: {max_global:.4e}")

    # Configuração Acadêmica do Plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    counts, bins_edges, patches = ax.hist(sample_means, bins=bins, density=True, 
                                          color='darkorange', edgecolor='black', 
                                          alpha=0.7, linewidth=1.2, label='Histogram (Mean Re)')

    ax.axvline(mu_val, color='black', linestyle='--', linewidth=1.5, label=r'Average ($\mu$)')

    # Estilização do Eixo
    ax.set_xlabel(property_label)
    ax.set_ylabel('Probability Density')
    ax.set_title('Local Reynolds Number Distribution')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Formatação Científica para o eixo X
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)
    
    if lim is None:
        p1, p99 = np.percentile(sample_means, [1, 99])
        margin = (p99 - p1) * 0.1
        ax.set_xlim(p1 - margin, p99 + margin)
    else:
        ax.set_xlim(lim[0], lim[1])
    
    # Caixa de Estatísticas
    stats_text = '\n'.join((
        fr'$N = {n_samples}$',
        fr'$\mu = {mu_val:.2e}$',
        fr'$\sigma = {sigma_val:.2e}$',
        fr'$Re_{{max}} = {max_global:.2e}$'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper left')

    # Salvar ou Mostrar
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir: 
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_9_random_samples_reynolds(datapath: str, 
                                   property_label: str = r"Local Reynolds $Re$", 
                                   save_path: str = None,
                                   seed: int = 42):
    """
    Plota um painel 3x3 com o histograma de todos os Reynolds locais dos voxels fluidos
    para 9 amostras aleatórias do dataset.
    """
    print(f"Loading dataset from: {datapath}")
    dataset = dr.LazyDatasetTorch(h5_path=datapath, 
                                  list_ids=None, 
                                  x_dtype=torch.float32,
                                  y_dtype=torch.float32)
    
    total_samples = len(dataset)
    if total_samples < 9:
        raise ValueError(f"Dataset only has {total_samples} samples. Need at least 9.")
    
    np.random.seed(seed)
    random_indices = np.random.choice(total_samples, size=9, replace=False)
    print(f"Selected random samples: {random_indices}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Intra-Sample Local Reynolds Distribution', fontsize=18, fontweight='bold')

    for i, (ax, idx) in enumerate(zip(axes.flat, random_indices)):
        # Extrair a amostra diretamente do dataset
        inp, tar = dataset[idx]
        
        # O modelo tensor_denorm exige dimensão de batch, então usamos unsqueeze(0)
        inp_batch = inp.unsqueeze(0)
        tar_batch = tar.unsqueeze(0)
        
        mask_tensor = inp[0] > 0
        mask_np = mask_tensor.cpu().numpy()
        
        local_thickness = ps.filters.local_thickness(mask_np)
        
        denorm_tar = vu.tensor_denorm(tar_batch, inp_batch)
        
        v_z = denorm_tar[0, 0]
        v_y = denorm_tar[0, 1]
        v_x = denorm_tar[0, 2]
        vel_mag = torch.sqrt(v_z**2 + v_y**2 + v_x**2).cpu().numpy()
        
        mu = 1.0 / 3.0
        re_local = vel_mag * local_thickness / mu
        
        # Puxar todos os voxels do fluido dessa amostra
        valid_re = re_local[mask_np]
            
        mu_val = np.mean(valid_re)
        sigma_val = np.std(valid_re)
        max_val = np.max(valid_re)
        
        bins = min(int(len(valid_re) * 0.05), 200) # Otimizado para não gerar mil bins pesados
        ax.hist(valid_re, bins=bins, density=True, 
                color='darkred', edgecolor='none', alpha=0.7)
        
        p1, p99 = np.percentile(valid_re, [0.1, 99.9])
        margin = (p99 - p1) * 0.1
        ax.set_xlim(max(0, p1 - margin), p99 + margin) # Garantir que x não seja negativo
        ax.set_title(f"Sample Index: {idx}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Sci format local
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)

        stats_text = '\n'.join((
            fr'$\mu = {mu_val:.2e}$',
            fr'$Re_{{max}} = {max_val:.2e}$'
        ))
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.95, 0.90, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        if i >= 6:
            ax.set_xlabel(property_label, fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel("Density", fontsize=11)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir: 
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"9-Panel Plot saved to {save_path}")
    else:
        plt.show()

# ==========================================
# Exemplo de Execução
# ==========================================

datasets = {
    "Parker (Rmax^2)": "./NN_Datasets/Test_Oliveira_Parker_120_120_120_RotAug_Rmax.h5"
}

for dataset_name, datapath in datasets.items():
    
    # 1. Distribuição Global do Reynolds (Média das Amostras)
    plot_reynolds_distribution(datapath=datapath, 
                               property_label=r"Mean Local Reynolds $Re$",
                               batch_size=2, # Batch size reduzido para evitar OOM no local_thickness
                               save_path=dataset_name+"_Distribution_Reynolds_Dataset.png")
    
    # 2. Reynolds Voxels Distribution em 9 amostras
    plot_9_random_samples_reynolds(datapath=datapath, 
                                   property_label=r"Local Reynolds $Re$",
                                   save_path=dataset_name+"_Distribution_Reynolds_Samples.png")