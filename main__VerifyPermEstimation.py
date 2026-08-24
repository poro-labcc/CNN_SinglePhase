import os
import numpy              as np
import torch
import matplotlib.pyplot  as plt
import pandas             as pd
import porespy            as ps
from scipy.stats          import pearsonr
from torch.utils.data     import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import distance_transform_edt

from Architectures.Unet   import Extended_DannyKo
from Utilities            import dataset_reader as dr
from Utilities            import velocity_usage as vu

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']


#######################################################
#************ UTILITY FUNCTIONS:           ***********#
#######################################################

def find_directional_local_maxima(dist_transform):
    """
    Finds local maxima in a 3D distance transform array.
    A voxel is considered a local maximum if it is strictly greater than 
    both of its neighbors along at least 2 out of the 3 axes (X, Y, Z).
    
    Parameters:
        dist_transform (numpy.ndarray): 3D array of the distance transform.
        
    Returns:
        numpy.ndarray: A 3D boolean array where True indicates a local maximum.
    """
    dt = np.asarray(dist_transform)
    
    # Initialize boolean arrays for each direction (padded with False at the borders)
    max_z = np.zeros_like(dt, dtype=bool)  # Axis 0
    max_y = np.zeros_like(dt, dtype=bool)  # Axis 1
    max_x = np.zeros_like(dt, dtype=bool)  # Axis 2
    
    # 1. Check Z direction (Axis 0)
    # Center is strictly greater than the slice before it AND the slice after it
    max_z[1:-1, :, :] = ((dt[1:-1, :, :] > dt[:-2, :, :]) & (dt[1:-1, :, :] > dt[2:, :, :]))
    
    # 2. Check Y direction (Axis 1)
    max_y[:, 1:-1, :] = (dt[:, 1:-1, :] > dt[:, :-2, :]) & (dt[:, 1:-1, :] > dt[:, 2:, :])
    
    # 3. Check X direction (Axis 2)
    max_x[:, :, 1:-1] = (dt[:, :, 1:-1] > dt[:, :, :-2]) & (dt[:, :, 1:-1] > dt[:, :, 2:])
    
    # Count how many axes flag the voxel as a maximum (converts True to 1, False to 0)
    max_count = (max_z & max_y).astype(np.int8) + (max_y & max_x).astype(np.int8) + (max_z & max_x).astype(np.int8)
    
    # Define global maximum: true in at least 2 directions AND must be inside the pore (dt > 0)
    local_maxima_mask = (max_count >= 2) & (dt > 0)
    
    return local_maxima_mask    

def analyze_thickness_correlation(dataloader, model, component, datasetname, device='cpu'):
    
    
    returns = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            current_bs = batch_inputs.shape[0]
            print(f"Processing Batch {batch_idx+1}...")

            batch_inputs_dev  = batch_inputs.to(dtype=torch.float32, device=device)
            
            #inlets              = np.zeros_like(vol, dtype=bool)
            #inlets[0, :, :]     = 1  
            #outlets             = np.zeros_like(vol, dtype=bool)    
            #outlets[-1, :, :]   = 1 
            #filt_vol            = ps.filters.trim_nonpercolating_paths(vol, inlets=inlets, outlets=outlets)
            
            
            if hasattr(model, 'predict'):
                batch_outputs = model.predict(batch_inputs_dev)
            else:
                batch_outputs = model(batch_inputs_dev)
            
            # Move outputs to CPU for NumPy correlation functions
            batch_outputs = batch_outputs.cpu()
            
            for samp in range(current_bs):
                void_mask = batch_inputs[samp, 0, ...] > 0
                out_samp = batch_outputs[samp]
                tar_samp = batch_targets[samp]
                
                # Analyse error
                if out_samp.shape[0] >= 3: 
                    out_mag = torch.sqrt(out_samp[0]**2 + out_samp[1]**2 + out_samp[2]**2)
                    tar_mag = torch.sqrt(tar_samp[0]**2 + tar_samp[1]**2 + tar_samp[2]**2)
                else:
                    out_mag = out_samp[0]
                    tar_mag = tar_samp[0]
                x_flat      = out_mag[void_mask].numpy().flatten()
                y_flat      = tar_mag[void_mask].numpy().flatten()
                corr_matrix = np.corrcoef(x_flat, y_flat)
                r_coeff     = corr_matrix[0, 1]
                
                # Analyse thickness accordance
                void_mask = void_mask.numpy().astype(bool)
                
                
                denorm_targets = vu.tensor_denorm(batch_targets[samp:samp+1], batch_inputs[samp:samp+1])
                
                perm = vu.permeability_calculation( denorm_targets, 
                                                    batch_inputs [samp:samp+1], 
                                                    tau=1.5,   
                                                    Re=0.1, 
                                                    dens=1.0,
                                                    denorm=False)
                                
                
                edt     = distance_transform_edt(void_mask).astype("float32")
                por         = np.mean(void_mask)
                
                # Danny Ko
                R_est_1     = 0.5*np.max(edt)
                perm_est_1  = 1013* 0.2 *(R_est_1)**2
                
                # Danny - Rmaximas
                maxima_mask = find_directional_local_maxima(edt)
                maximas     = edt[maxima_mask]
                R_est_2     = np.mean(maximas)
                perm_est_2  = 1013* 0.2 *(R_est_2)**2
                
                # Danny Ko adaptado (razao 0.125 corrigida, termo de porosidade adicionado)
                perm_est_3  = 1013* 0.125 *(R_est_1)**2 * por
                
                # Danny Ko adaptado - Rmaximas (razao 0.125 corrigida, termo de porosidade adicionado)
                perm_est_4  = 1013* 0.125 *(R_est_2)**2 * por
                
                # Kozeny-Carman (adaptacao de Danny Ko + tortuosidade)
                tort        = 1 + 0.8*(1-por)
                perm_est_5  = 1013* (0.125 * R_est_1**2 * por / tort**2)
                
                # Kozeny-Carman - Rmaximas (adaptacao de Danny Ko + tortuosidade)
                perm_est_6  = 1013* (0.125 * R_est_2**2 * por / tort**2)
                returns.append(  (perm, (perm_est_1, perm_est_2, perm_est_3, perm_est_4, perm_est_5, perm_est_6) ) )
                
                print(f"{perm.item():.4f}: {perm_est_1.item():.4f};  {perm_est_2.item():.4f};  {perm_est_3.item():.4f};  {perm_est_4.item():.4f}")
    
    return returns

#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################

component  = 0
batch_size = 10  
N_samples  = None
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path defined as requested
model_folder = "/home/gabriel/remote/hal/dissertacao/NN_Results/NN_Trainning_13_March_2026_02-16PM_Job16074/"
model_name   = "model_LowerValidationLoss.pth"
model_path   = os.path.join(model_folder, model_name)

# Ensure you have multiple datasets active here to see the combined plot
datasets = {
    "Training Data":     "../NN_Datasets/OUTDATED_18_05_2026/PressureDriven/Train_Danny_120_120_120_Pressure.h5",
    "Cylindrical Grain": "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_CylinGrain_120_120_120.h5",
    "Cylindrical Pore":  "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_CylinPore_120_120_120.h5",
    "Spherical Grain":   "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_SphGrain_120_120_120.h5",
    "Spherical Pore":    "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_SphPore_120_120_120.h5",
    #"Parker":       "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Parker_120_120_120.h5",
    #"Leopard":      "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Leopard_120_120_120.h5",
    #"Kirby":        "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Kirby_120_120_120.h5",
    #"Castle Gate":  "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_CastleGate_120_120_120.h5",
    #"Brown":        "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Brown_120_120_120.h5",
    #"Upper Gray":   "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_BereaUpperGray_120_120_120.h5",
    #"Sinter Gray":  "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_BereaSinterGray_120_120_120.h5",
    #"Berea Buff":   "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_BereaBuff_120_120_120.h5",
    #"Berea":        "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Berea_120_120_120.h5",
    #"Bentheimer":   "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Bentheimer_120_120_120.h5",
    #"Bandera":      "../NN_Datasets/OUTDATED_18_05_2026/ForceDriven/Test_Oliveira_Bandera_120_120_120.h5",
}

# Load Model
print("\nLoading Model...")
model_aux = Extended_DannyKo()
danny_model = model_aux.z_model
danny_model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
danny_model.eval()
danny_model.bin_input = True
danny_model.to(device)


#######################################################
#************ RUN ANALYSIS:                ***********#
#######################################################

all_results = {}

for dataname, datapath in datasets.items():
    print(f"\nAnalyzing Dataset: {dataname}")
    
    # Load Dataloader
    ids_to_load = np.arange(N_samples) if N_samples is not None else None
    dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=ids_to_load, x_dtype=torch.float32, y_dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run analysis and store in dictionary
    res = analyze_thickness_correlation(dataloader, danny_model, component, dataname, device)
    all_results[dataname] = res
    

# =====================================================
# PREPARANDO OS DADOS
# =====================================================
print("\nGenerating plots...")

plot_out_dir = "../NN_Results/Plots/Thickness_Correlation/"
os.makedirs(plot_out_dir, exist_ok=True)
color_palette = plt.cm.tab10.colors

x_labels = [
    "Ko et al. (Max R)", 
    "Ko et al. (Mean Max R)", 
    "Ko Corrected (Max R)", 
    "Ko Corrected (Mean Max)",
    "Kozeny-Carman (Max R)", 
    "Kozeny-Carman (Mean Max)"
]

# =====================================================
# FIGURA 1: SCATTER PLOTS (True vs Est)
# =====================================================
fig1, axes1 = plt.subplots(1, 6, figsize=(24, 5))
fig1.suptitle("Comparação de Permeabilidade: Real vs Estimada", fontsize=16, y=1.05)

for idx, (dataname, res_list) in enumerate(all_results.items()):
    c = color_palette[idx % len(color_palette)]
    
    # Extrair os dados da rocha atual
    y_true_perm = np.array([float(item[0]) for item in res_list])
    
    x_ests = [
        np.array([float(item[1][0]) for item in res_list]),
        np.array([float(item[1][1]) for item in res_list]),
        np.array([float(item[1][2]) for item in res_list]),
        np.array([float(item[1][3]) for item in res_list]),
        np.array([float(item[1][4]) for item in res_list]),
        np.array([float(item[1][5]) for item in res_list])
    ]
    
    # Plotar os pontos no Scatter Plot
    for i in range(6):
        axes1[i].scatter(x_ests[i], y_true_perm, alpha=0.4, s=60, color=c, 
                         edgecolors='black', linewidth=0.5, label=dataname if i==0 else "", zorder=3)

# Formatação dos Scatter Plots
for i, ax in enumerate(axes1):
    ax.set_xlabel(x_labels[i], fontsize=13)
    if i == 0:
        ax.set_ylabel("Calculated Permeability (True)", fontsize=13)
    
    # Pegar limites automáticos e forçar intervalo quadrado
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    lower_bound = min(0, x_min, y_min) 
    upper_bound = max(x_max, y_max)
    
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], color='grey', linestyle='--', alpha=0.7, zorder=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
    ax.set_box_aspect(1) # Forçar formato quadrado

# Legenda do scatter plot
axes1[5].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True, fontsize=10)
fig1.tight_layout()


# =====================================================
# FIGURA 2: TABELA DE ERRO PERCENTUAL MÉDIO (MAPE)
# =====================================================
# Calcular altura da figura dinamicamente com base na quantidade de datasets
fig2, ax2 = plt.subplots(figsize=(14, 0.6 * len(datasets) + 2))
fig2.suptitle("Erro Percentual Absoluto Médio (MAPE) por Grupo", fontsize=16, y=0.95)
ax2.axis('off')
ax2.axis('tight')

table_data = []
row_labels = []

# Calcular o erro (MAPE) para cada dataset e cada estimador
for dataname, res_list in all_results.items():
    row_labels.append(dataname)
    y_true = np.array([float(item[0]) for item in res_list])
    valid = y_true > 0
    
    row_errors = []
    for i in range(6):
        x_est = np.array([float(item[1][i]) for item in res_list])
        if np.any(valid):
            mape = np.mean(np.abs((x_est[valid] - y_true[valid]) / y_true[valid])) * 100
            row_errors.append(f"{mape:.1f}%")
        else:
            row_errors.append("N/A")
            
    table_data.append(row_errors)

# Calcular a MÉDIA GERAL (todas as rochas juntas) para a última linha
overall_errors = []
for i in range(6):
    x_est_all = np.concatenate([np.array([float(item[1][i]) for item in res_list]) for res_list in all_results.values()])
    y_true_all = np.concatenate([np.array([float(item[0]) for item in res_list]) for res_list in all_results.values()])
    valid_all = y_true_all > 0
    
    if np.any(valid_all):
        mape_overall = np.mean(np.abs((x_est_all[valid_all] - y_true_all[valid_all]) / y_true_all[valid_all])) * 100
        overall_errors.append(f"{mape_overall:.1f}%")
    else:
        overall_errors.append("N/A")

# Adicionar linha global à tabela
table_data.append(overall_errors)
row_labels.append("MÉDIA GERAL")

# Headers mais curtos para caber bem na tabela
col_labels = [
    "Ko et al. (Max R)", 
    "Ko et al. (Mean Max R)", 
    "Ko Corr. (Max R)", 
    "Ko Corr. (Mean Max)",
    "Koz-Car (Max R)", 
    "Koz-Car (Mean Max)"
]


# Renderizar a tabela no Matplotlib
table = ax2.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')

# Estilização visual da tabela
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2) # Aumentar a altura das linhas para melhor leitura

# Destacar a linha "MÉDIA GERAL" em negrito (opcional)
for (row, col), cell in table.get_celld().items():
    if row == len(row_labels): # Última linha
        cell.set_text_props(weight='bold')

# Imprimir também no console para acesso rápido via texto usando o Pandas
print("\n--- Tabela de Erro (MAPE) ---")
df_table = pd.DataFrame(table_data, index=row_labels, columns=[c.replace('\n', ' ') for c in col_labels])
print(df_table)
print("-----------------------------\n")

plt.tight_layout()
plt.show()