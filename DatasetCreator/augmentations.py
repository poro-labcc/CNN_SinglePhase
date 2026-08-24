import numpy as np



def flip_x_augmentation(solid, uz, uy, ux, pr):
    
    aux_so = np.flip(solid, axis=2)
    aux_ux = np.flip(-1*ux, axis=2)
    aux_uy = np.flip(uy,    axis=2)
    aux_uz = np.flip(uz,    axis=2)
    aux_pr = np.flip(pr,    axis=2)
    
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr


def flip_y_augmentation(solid, uz, uy, ux, pr):
    
    aux_so = np.flip(solid, axis=1)
    aux_ux = np.flip(ux,    axis=1)
    aux_uy = np.flip(-1*uy, axis=1)
    aux_uz = np.flip(uz,    axis=1)
    aux_pr = np.flip(pr,    axis=1)
    
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr

def rotate_z_augmentation(solid, uz, uy, ux, pr, direc):
    # Change signals 
    if direc > 0:
        k_val   = 1
        base_ux = -1 * uy
        base_uy = ux
        
    elif direc < 0:
        k_val   = -1
        base_ux = uy
        base_uy = -1 * ux
        
    else:
        return solid, uz, uy, ux, pr
    
    # Attributes which the signal are not influencied by rotation
    aux_so = np.rot90(solid,    k=k_val, axes=(1, 2))
    aux_pr = np.rot90(pr,       k=k_val, axes=(1, 2))
    aux_uz = np.rot90(uz,       k=k_val, axes=(1, 2)) 
    
    # Attributes which the signal are influencied by rotation
    aux_ux = np.rot90(base_ux,  k=k_val, axes=(1, 2))
    aux_uy = np.rot90(base_uy,  k=k_val, axes=(1, 2))

    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr



def mirror_y_augmentation(solid, uz, uy, ux, pr, shift):
   
    # Aux data
    aux_so        = np.zeros_like(solid)
    aux_uz        = np.zeros_like(uz)
    aux_uy        = np.zeros_like(uy)
    aux_ux        = np.zeros_like(ux)
    aux_pr        = np.zeros_like(pr)

    # --- Shift True Y (axis=1) ---
    if shift < 0:
    
        aux_so[:, :shift, :]          = solid[:, -1*shift:, :]    
        aux_uz[:, :shift, :]          = uz[:, -1*shift:, :]
        aux_uy[:, :shift, :]          = uy[:, -1*shift:, :]
        aux_ux[:, :shift, :]          = ux[:, -1*shift:, :]
        aux_pr[:, :shift, :]          = pr[:, -1*shift:, :]
        
        aux_so[:, shift:, :]          = np.flip(solid, axis=1) [:,:-1*shift,:]
        aux_ux[:, shift:, :]          = np.flip(ux,    axis=1 )[:,:-1*shift,:] # Ux
        aux_uy[:, shift:, :]          = np.flip(-1*uy, axis=1 )[:,:-1*shift,:] # Uy
        aux_uz[:, shift:, :]          = np.flip(uz,    axis=1 )[:,:-1*shift,:] # Uz
        aux_pr[:, shift:, :]          = np.flip(pr,    axis=1 )[:,:-1*shift,:] # Pr
        
    elif shift > 0:
        
        aux_so[:, :shift, :]          = np.flip(solid, axis=1) [:,-1*shift:,:] # 
        aux_ux[:, :shift, :]          = np.flip(ux,    axis=1 )[:,-1*shift:,:] # Ux
        aux_uy[:, :shift, :]          = np.flip(-1*uy, axis=1 )[:,-1*shift:,:] # Uy
        aux_uz[:, :shift, :]          = np.flip(uz,    axis=1 )[:,-1*shift:,:] # Uz
        aux_pr[:, :shift, :]          = np.flip(pr,    axis=1 )[:,-1*shift:,:] # Pr
        
        aux_so[:, shift:, :]          = solid[:, :-1*shift, :]    
        aux_uz[:, shift:, :]          = uz   [:, :-1*shift, :]
        aux_uy[:, shift:, :]          = uy   [:, :-1*shift, :]
        aux_ux[:, shift:, :]          = ux   [:, :-1*shift, :]
        aux_pr[:, shift:, :]          = pr   [:, :-1*shift, :]
        
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr
    
def mirror_x_augmentation(solid, uz, uy, ux, pr, shift):
   
    # Aux data
    aux_so        = np.zeros_like(solid)
    aux_uz        = np.zeros_like(uz)
    aux_uy        = np.zeros_like(uy)
    aux_ux        = np.zeros_like(ux)
    aux_pr        = np.zeros_like(pr)
    
    if shift < 0:
    
        aux_so[:, :, :shift]          = solid[:, :, -1*shift:]    
        aux_uz[:, :, :shift]          = uz[:, :, -1*shift:]
        aux_uy[:, :, :shift]          = uy[:, :, -1*shift:]
        aux_ux[:, :, :shift]          = ux[:, :, -1*shift:]
        aux_pr[:, :, :shift]          = pr[:, :, -1*shift:]
        
        aux_so[:, :, shift:]          = np.flip(solid, axis=2) [:,:-1*shift]
        aux_ux[:, :, shift:]          = np.flip(-1*ux, axis=2) [:,:-1*shift] # Ux
        aux_uy[:, :, shift:]          = np.flip(uy,    axis=2) [:,:-1*shift] # Uy
        aux_uz[:, :, shift:]          = np.flip(uz,    axis=2) [:,:-1*shift] # Uz
        aux_pr[:, :, shift:]          = np.flip(pr,    axis=2) [:,:-1*shift] # Pr
        
    elif shift > 0:
        
        aux_so[:, :, :shift]          = np.flip(solid, axis=2) [:,-1*shift:]
        aux_ux[:, :, :shift]          = np.flip(-1*ux, axis=2) [:,-1*shift:] # Ux
        aux_uy[:, :, :shift]          = np.flip(uy,    axis=2) [:,-1*shift:] # Uy
        aux_uz[:, :, :shift]          = np.flip(uz,    axis=2) [:,-1*shift:] # Uz
        aux_pr[:, :, :shift]          = np.flip(pr,    axis=2) [:,-1*shift:] # Pr
        
        aux_so[:, :, shift:]          = solid[:, :, :-1*shift]   
        aux_uz[:, :, shift:]          = uz   [:, :, :-1*shift]
        aux_uy[:, :, shift:]          = uy   [:, :, :-1*shift]
        aux_ux[:, :, shift:]          = ux   [:, :, :-1*shift]
        aux_pr[:, :, shift:]          = pr   [:, :, :-1*shift]

    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr
