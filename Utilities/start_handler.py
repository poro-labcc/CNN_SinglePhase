import numpy as np
from pathlib import Path
from typing import Tuple, Union
from scipy.ndimage import distance_transform_edt
import pyvista as pv
import matplotlib.pyplot as plt
import re
import subprocess
from Utilities import velocity_usage as vu
import os

def write_start_raw(
    filename: str,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    pr: np.ndarray,
):
    """
    Writes Start.00000.raw as a DENSE 3D buffer.
    Input arrays must be (Nz, Ny, Nx) including Halos.
    Layout: [Ux(0,0,0), Uy(0,0,0), Uz(0,0,0), Pr(0,0,0), Ux(0,0,1)...]
    """
    Nz, Ny, Nx = ux.shape
    N = Nz * Ny * Nx
        
    full_path = filename if filename.endswith(".raw") else f"{filename}.raw"
    output_dir = os.path.dirname(full_path)
    
    if output_dir: os.makedirs(output_dir, exist_ok=True)
        
    dense_grid = np.stack((ux, uy, uz, pr), axis=-1) 

    buffer = dense_grid.astype(np.float64) 

    with open(filename+".raw", "wb") as f:
        buffer.tofile(f)

def write_lbpm_db(
    path: str,
    *,
    db_name:    str = "simulation.db",   # used if `path` is a directory
    bc:         int = 0,
    din:        float = 1.0,
    dout:       float = 1.0,
    fz:         float = 0.0,
    fx:         float = 0.0,
    fy:         float = 0.0,
    tau:        float = 1.5,
    timestep_max: int = 100000000,
    tolerance: float = 1e-6,
    Start: bool = True,
    # Domain
    domain_filename:str = "domain.raw",
    read_type:      str = "8bit",
    nproc:          Tuple[int, int, int] = (1, 1, 4),
    n:              Tuple[int, int, int] = (256, 256, 128),
    N:              Tuple[int, int, int] = (256, 256, 512),
    offset:         Tuple[int, int, int] = (0, 0, 0),
    voxel_length:   float = 1.0,
    read_values:    Tuple[int, int] = (0, 1),
    write_values:   Tuple[int, int] = (0, 1),
    inlet_layers:   Tuple[int, int, int] = (0, 0, 0),
    outlet_layers:  Tuple[int, int, int] = (0, 0, 0),
    # Visualization
    write_silo:     bool = True,
    save_8bit_raw:  bool = True,
    save_phase_field: bool = True,
    save_pressure:  bool = True,
    save_velocity:  bool = True,
    # Analysis
    analysis_interval:          int = 1000,
    subphase_analysis_interval: int = 5000,
    n_threads:                  int = 0,
    visualization_interval:     int = 5000,
    restart_interval:           int = 100_000_000,
    restart_file:               str = "Restart",
    out_format:                 str = "vtk"
) -> str:
    def tsv3(v): return f"{v[0]}, {v[1]}, {v[2]}"
    def tsv2(v): return f"{v[0]}, {v[1]}"
    def b(v):    return "true" if v else "false"
    def ffmt(x): return f"{x:.6g}"

    text = f"""MRT {{
   tau         = {ffmt(tau)}
   din         = {din}   // inlet density (controls pressure)
   dout        = {dout}  // outlet density (controls pressure)
   F           = {ffmt(fx)}, {ffmt(fy)}, {ffmt(fz)}   // Fx, Fy, Fz
   timestepMax = {timestep_max}
   tolerance   = {ffmt(tolerance)}
   Start       = {b(Start)}
}}
Domain {{
   Filename = "{domain_filename}"
   ReadType = "{read_type}"      // data type

   nproc = {tsv3(nproc)}
   n     = {tsv3(n)}
   N     = {tsv3(N)}

   offset         = {tsv3(offset)} // offset to read sub-domain
   voxel_length   = {ffmt(voxel_length)}     // voxel length (in microns)
   ReadValues     = {tsv2(read_values)}    // labels within the original image
   WriteValues    = {tsv2(write_values)}    // associated labels to be used by LBPM (0:solid, 1..N:fluids)
   BC             = {bc}       // boundary condition type (0 for periodic)
   InletLayers    = {tsv3(inlet_layers)}   // specify layers along the inlet
   OutletLayers   = {tsv3(outlet_layers)}  // specify layers along the outlet
}}
Visualization {{
   format            = "{out_format}"
   write_silo        = {b(write_silo)}     // SILO databases with assigned variables
   save_8bit_raw     = {b(save_8bit_raw)}  // labeled 8-bit binary files with phase assignments
   save_phase_field  = {b(save_phase_field)}  // phase field within SILO database
   save_pressure     = {b(save_pressure)}    // pressure field within SILO database
   save_velocity     = {b(save_velocity)}    // velocity field within SILO database
}}
Analysis {{
   analysis_interval             = {analysis_interval}        // logging interval for timelog.csv
   subphase_analysis_interval    = {subphase_analysis_interval}  // logging interval for subphase.csv
   N_threads                     = {n_threads}                // number of analysis threads (GPU version only)
   visualization_interval        = {visualization_interval}   // interval to write visualization files
   restart_interval              = {restart_interval}         // interval to write restart file
   restart_file                  = "{restart_file}"           // base name of restart file
}}
"""
    p = Path(path)
    # If `path` is a directory or lacks a suffix, write inside it
    if p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        p = p / db_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(text, encoding="utf-8")
    return text
