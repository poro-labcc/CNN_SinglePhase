import os
import numpy         as np
import math
from   pathlib       import Path
from   typing        import Tuple,  List
from   scipy.ndimage import distance_transform_edt
from   typing        import Union
from   numpy.typing  import NDArray
import cc3d
import matplotlib.pyplot as plt
import porespy as ps

def pressure_calculation(
    matriz_binaria: np.ndarray,
    tau:        Union[float, int],
    Re:         float = 0.1,
    Dens:       float = 1.0,
    )->float:
    
    L               = matriz_binaria.shape[0]
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    dP              = (Re * 8.0 * (Visc ** 2) * L) / (Dens * (R ** 3))

    return dP

def force_calculation(
    matriz_binaria: np.ndarray,
    tau:            Union[float, int],
    Re:             float = 0.1,
    Dens:           float = 1.0,
) -> float:
    
    dist_transform  = distance_transform_edt(matriz_binaria)
    if dist_transform.size == 0 or np.max(dist_transform) == 0: return 0.0
    R               = np.max(dist_transform)
    Visc            = (tau - 0.5) / 3.0
    Fx              = (Re * 8.0 * (Visc ** 2)) / (Dens * (R ** 3))
    
    return Fx

def calculate_permeability(
        matriz_binaria: np.ndarray,
        velocity_field: np.ndarray,
        ):
        
    tau     = 1.5
    Dens    = 1.0
    Re      = 0.1
    porosity  = np.count_nonzero(matriz_binaria)/matriz_binaria.size
    visc      = (tau-0.5)/3
    force_z   = force_calculation(matriz_binaria, tau=tau, Re=Re, Dens=Dens)
    mask      = matriz_binaria[0]
    u_z       = velocity_field[0]
    flow_rate = np.mean(u_z[mask])
    k_star    = flow_rate *visc*porosity  / (Dens*force_z)
    k         = k_star*1013.0
    return k