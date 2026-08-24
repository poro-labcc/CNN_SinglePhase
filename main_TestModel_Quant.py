import numpy              as np
import torch
import tensorflow         as tf
import matplotlib.pyplot  as plt
import pandas             as pd
from torch.utils.data     import DataLoader, Subset

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition

from Utilities            import dataset_reader as dr
from Utilities            import error_metrics as em 
from Utilities            import model_handler as mh


#######################################################
#************ GETTER COMPARISONS:          ***********#
#######################################################

def Test_Model_on_Dataset(dataloader, model, component, model_name, datasetname):
        
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']
    
    # Initialize lists to store metrics for EVERY sample in the dataset
    all_metrics = {
        "b_metrics": [], # Bias
        "m_metrics": [], # Magnitude
        "a_metrics": [], # Angular
        "c_metrics": [], # Correlation
        "f_metrics": [], # Flux
        "t_metrics": [], # Tortuosity
        "d_metrics": []  # Divergent
    }

    # Iterate over the entire dataset
    for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        
        current_bs = batch_inputs.shape[0]
        print(f"Processing Batch {batch_idx+1} (Size: {current_bs})...")

        # 1. Prediction
        batch_inputs  = batch_inputs.clone().detach().to(dtype=torch.float32)
        batch_outputs = model.predict(batch_inputs)
        print("Out: ", batch_outputs.mean(), "; Tar: ", batch_targets.mean())
        
        print("output shape: ", batch_outputs.shape)
        print("target shape: ", batch_targets.shape)
            
        # 2. Casting and transformation
        batch_targets = batch_targets.clone().detach().to(dtype=torch.float32)
        batch_outputs = batch_outputs.clone().detach().to(dtype=torch.float32)
        
    
        # Calculate & Collect Metrics
        if component is None or component==5: # If component is Z,Y,X, P
            
            # 3D Metrics
            all_metrics["b_metrics"].extend(em.Bias_Comparison        (batch_inputs, batch_outputs, batch_targets))
            all_metrics["m_metrics"].extend(em.Magnitude_Comparison   (batch_inputs, batch_outputs, batch_targets))
            all_metrics["a_metrics"].extend(em.Angular_Comparison     (batch_inputs, batch_outputs, batch_targets))
            all_metrics["c_metrics"].extend(em.Correlation_Comparison (batch_inputs, batch_outputs, batch_targets))
            all_metrics["f_metrics"].extend(em.Flux_Comparison        (batch_inputs, batch_outputs, batch_targets))
            all_metrics["t_metrics"].extend(em.Tortuosity_Comparison  (batch_inputs, batch_outputs, batch_targets))
            all_metrics["d_metrics"].extend(em.Divergent_Residual     (batch_inputs, batch_outputs))
            
        # If analyzing a sub-model, restrict the tensors to 1 channel
        else: 
            batch_outputs = batch_outputs[:, component:component+1, :,:,:]
            batch_targets = batch_targets[:, component:component+1, :,:,:]
            # 1D Metrics (Z-direction only)
            all_metrics["b_metrics"].extend(em.Bias_Comparison        (batch_inputs, batch_outputs, batch_targets))
            all_metrics["m_metrics"].extend(em.Magnitude_Comparison   (batch_inputs, batch_outputs, batch_targets))
            all_metrics["c_metrics"].extend(em.Correlation_Comparison (batch_inputs, batch_outputs, batch_targets))
            
                
    # --- Final Global Aggregation ---
    final_results = {}
    
    final_results["Mean Bias Error      [%]"]    = np.mean(all_metrics["b_metrics"])
    final_results["Mean Magnitude Error [%]"]    = np.mean(all_metrics["m_metrics"])
    final_results["Mean Correlation        "]    = np.mean(all_metrics["c_metrics"])
    
    if component is None or component==5: # If component is Z,Y,X
        final_results["Mean Angular Error    [Deg]"]    = np.mean(all_metrics["a_metrics"])
        final_results["Mean Tortuosity Error   [%]"]    = np.mean(all_metrics["t_metrics"])
        final_results["Mean Divergent Residual [%]"]    = np.mean(all_metrics["d_metrics"])
        final_results["Mean Flux Error            "]    = np.mean(all_metrics["f_metrics"])


    return final_results



#######################################################
#************ MAIN SETUP:                  ***********#
#######################################################
# Choose component
# 0 - z models
# 1 - y models
# 2 - x models
# 3 - p models
# 4 - zyx models
# None - zyx-p models
component        = 0

batch_size       = 9
N_samples        = None # 'None' to consider all available samples
device           = 'cpu'

# DEFINE DATASETS
datasets        = {
    
    "Ko et. al":            "../NN_Datasets_Grad/Test_Danny_SphPore_DAug_DNorm.h5",
    
    "Spherical Pores":      "../NN_Datasets_Grad/Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":     "../NN_Datasets_Grad/Test_Silveira_SphGrain_SAug_DNorm.h5",
    "Cylindrical Pores":    "../NN_Datasets_Grad/Test_Silveira_CylinPore_SAug_DNorm.h5",
    "Cylindrical Grains":   "../NN_Datasets_Grad/Test_Silveira_CylinGrain_SAug_DNorm.h5",
     
    "Bentheimer":           "../NN_Datasets_Grad/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    "Berea Buff":           "../NN_Datasets_Grad/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Leopard":              "../NN_Datasets_Grad/Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":          "../NN_Datasets_Grad/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":     "../NN_Datasets_Grad/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray":    "../NN_Datasets_Grad/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea":                "../NN_Datasets_Grad/Test_Oliveira_Berea_SAug_DNorm.h5",
    
    
    }


# DEFINE MODELS
models          = {}

    
 
danny_model         = Extended_DannyKo()
danny_model_z       = danny_model.z_model
model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-12PM_Job26188/model_LowerValidationLoss.pth"
danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
danny_model_z.bin_input = True
danny_model_z.eval()
models["Ko et. al (Etapa 0)"]     = danny_model_z


danny_model         = Extended_DannyKo()
danny_model_z       = danny_model.z_model
model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-31PM_Job26190/model_LowerValidationLoss.pth"
danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
danny_model_z.bin_input = True
danny_model_z.eval()
models["Ko et. al (Etapa 1)"]     = danny_model_z


danny_model         = Extended_DannyKo()
danny_model_z       = danny_model.z_model
model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-28PM_Job26189/model_LowerValidationLoss.pth"
danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
danny_model_z.bin_input = True
danny_model_z.eval()
models["Ko et. al (Etapa 2)"]     = danny_model_z

print(" Model's trainable parameters")
for model_name, model in models.items():
    print(" Model {model_name}: {mh.get_n_trainable_params(model)}")

#######################################################
#************ RUN ANALYSIS:                ***********#
#######################################################

results = {}
def stash_metrics(dataname: str, model_id, metrics: dict):
    for metric_name, value in metrics.items():
        results.setdefault(metric_name, {})
        results[metric_name].setdefault(model_id, {})
        results[metric_name][model_id][dataname] = float(value)
        
        


            
print("Starting test routine...")
for dataname, datapath in datasets.items():
    print("="*120)
    print("\n\n\n ", dataname, " results:\n\n")
    
    # Load data
    print("Loading data...")
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    if N_samples is not None:
        N = min(N_samples, len(dataset))
        dataset = Subset(dataset, range(N))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Compute metrics for each model
    for model_id, model in models.items():
        print("Testing model:  ", model_id)
        # Compute metrics
        metrics = Test_Model_on_Dataset(dataloader, model, component=component, model_name=model_id, datasetname=dataname)
        print("\n-----------------------------------------------------\n")
        # Register metrics achieved
        stash_metrics(dataname, model_id, metrics)
    print()
    
    del dataloader
    
#######################################################
#************ SAVE RESULTS IN DATAFRAME:   ***********#
#######################################################

dfs = {}
for metric_name, metric_cube in results.items():
    df = pd.DataFrame.from_dict(metric_cube, orient="index")
    df = df.reindex(columns=list(datasets.keys()))  # garante ordem das colunas
    df = df.T
    df['Mediana'] = df.median(axis=1)
    df.loc['Mediana'] = df.median(axis=0)
    dfs[metric_name] = df
    print()
    print(metric_name)
    print(df)
    print()

if      component==0:   path = "../NN_Results/Z_"
elif    component==1:   path = "../NN_Results/Y_"
elif    component==2:   path = "../NN_Results/X_"
elif    component==3:   path = "../NN_Results/P_"
else:                   path = "../NN_Results/"
   

GREEN_CELL = "green!25"
RED_CELL   = "red!25"

for metric_name, df in dfs.items():
    latex_path  = path + f"{metric_name.replace(' ', '_')}.tex"
    print("Saving dataframe in: ", latex_path)
    n_models    = df.shape[1] - 1
    col_fmt     = "P{3cm} " + " ".join(["P{2cm}"] *n_models ) + " | P{2cm}"
    
    # which columns to color (e.g. skip the first one = baseline)
    columns_to_color = list(df.columns)

    # ------------- choose rule based on metric_name -------------
    # defaults: no coloring
    mode        = "none" # "lower_better", "higher_better" or "none"
    green_thr   = None
    red_thr     = None
    
    # EXAMPLES – you will adapt these:
    if "Mean Bias Error [%]" in metric_name: # ok
        mode = "lower_better"
        green_thr = 20
        red_thr   = 50  

    elif "Mean Magnitude Error [%]" in metric_name: # ok
        mode = "lower_better"
        green_thr = 20  
        red_thr   = 50 

    elif "Mean Correlation" in metric_name: # ok
        mode = "higher_better"
        green_thr = 0.8
        red_thr   = 0.5
        
    elif "Mean Flux Error" in metric_name: # ok
        mode = "lower_better"
        green_thr = 0.2
        red_thr   = 0.5 
        
    elif "Mean Angular Error [Deg]" in metric_name:# ok
        mode = "lower_better"
        green_thr = 15
        red_thr   = 75
        
    elif "Mean Tortuosity Error [%]" in metric_name:# ok
        mode = "lower_better"
        green_thr = 20
        red_thr   = 50
        
    elif "Mean Divergent Residual [%]" in metric_name:
        mode = "lower_better"
        green_thr = 5
        red_thr   = 20
        
        
            

    # ------------- formatter using the chosen rule -------------
    def make_formatter(col_name):
        def formatter(v):
            if pd.isna(v):
                return ""

            color_prefix = ""
            if col_name in columns_to_color and mode != "none":
                if mode == "lower_better" and green_thr is not None and red_thr is not None:
                    if v < green_thr:
                        color_prefix = f"\\cellcolor{{{GREEN_CELL}}}"
                    elif v > red_thr:
                        color_prefix = f"\\cellcolor{{{RED_CELL}}}"
                elif mode == "higher_better" and green_thr is not None and red_thr is not None:
                    if v > green_thr:
                        color_prefix = f"\\cellcolor{{{GREEN_CELL}}}"
                    elif v < red_thr:
                        color_prefix = f"\\cellcolor{{{RED_CELL}}}"

            return f"{color_prefix}{v:.4f}"
        return formatter

    formatters = {col: make_formatter(col) for col in df.columns}

    latex_body = df.to_latex(
        column_format=col_fmt,
        escape=False,
        formatters=formatters
    )
    
    latex_body = latex_body.replace("\nMediana", "\n\\hline\nMediana")
    metric_name = metric_name.replace("%", "\\%")

    wrapped_latex = (
        "\\begin{table}[h!]\n"
        "    \\centering\n"
        "    \\footnotesize\n"
        f"    \\caption{{{metric_name}}}\n"
        
        
        "    \\label{tab:results}\n"
        + latex_body +
        "\\end{table}\n"
    )

    with open(latex_path, "w") as f:
        f.write(wrapped_latex)
