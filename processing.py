"""
Processing / math utilities for CCD measurement notebook.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


#########################
#       Functions       #
#########################
def linear(x, a, b):
    return a * x + b

def linear_odr(B, x):
    """ODR-compatible model"""
    return B[0] * x + B[1]

def normal_odr(p, x):
    """ODR-compatible model"""
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))

def dark_current_func(p, T):
    Tk = T + 273.15  # degC -> K
    return p[0] * Tk ** (3 / 2) * np.exp(-p[1] / Tk)

def total_noise(S, G, RON, P_FPN):
    return np.sqrt(RON ** 2 + S / G + (P_FPN * S) ** 2)

#########################
#       Fitting         #
#########################
def fitting(func,x,y,p0,bounds=None):
    popt, pcov = curve_fit(func, x, y, p0=p0, 
                           bounds=bounds if bounds is not None else (-np.inf, np.inf))
    perr = np.sqrt(np.diag(pcov)) # fitting error
    return popt, perr


##########################
#        Binning         #
##########################
def binning(arr,binsize=5):
    Ny, Nx = arr.shape
    Nx2, Ny2 = Nx//binsize, Ny//binsize
    binned = np.zeros((Ny2,Nx2))
    for i in range(Ny2):
        y_lb, y_ub = i*binsize, min((i+1)*binsize,Ny)
        for j in range(Nx2):
            x_lb,x_ub = j*binsize, min((j+1)*binsize,Nx)
            binned[i][j] = np.median(arr[y_lb:y_ub,x_lb:x_ub])
    return binned


####################################
# Pairwise frame differences (RDN) #
####################################
def calc_difference(frames, device = 0):
    """
    Pairwise differences between frames (i vs j>i), returning arrays of:
      - mean(|diff|) over pixels
      - std(diff)/sqrt(2) over pixels
      - uses CuPy if available, otherwise NumPy

    Parameters
    ----------
    frames : array-like, shape (N, Ny, Nx)
    device : int
        GPU device index when using CuPy.

    Returns
    -------
    mean : np.ndarray
    std  : np.ndarray
    """
    try:
        import cupy as cp 
        xp = cp
        use_gpu = True
    except Exception:
        cp = None
        xp = np
        use_gpu = False

    mean, std = [], []
    num_frames = len(frames)

    if use_gpu:
        assert cp is not None
        with cp.cuda.Device(device):
            frames_xp = cp.asarray(frames)
            for i in tqdm(range(num_frames - 1)):
                diffs = frames_xp[i] - frames_xp[i + 1 : num_frames]
                diffs_mean = cp.abs(cp.mean(diffs, axis=(1, 2)))
                diffs_std = cp.std(diffs, axis=(1, 2)) / cp.sqrt(2)
                mean.extend(cp.asnumpy(diffs_mean).tolist())
                std.extend(cp.asnumpy(diffs_std).tolist())
        return np.array(mean), np.array(std)

    frames_np = np.asarray(frames)
    for i in tqdm(range(num_frames - 1)):
        diffs = frames_np[i] - frames_np[i + 1 : num_frames]
        diffs_mean = np.abs(np.mean(diffs, axis=(1, 2)))
        diffs_std = np.std(diffs, axis=(1, 2)) / np.sqrt(2)
        mean.extend(diffs_mean.tolist())
        std.extend(diffs_std.tolist())
    return np.array(mean), np.array(std)


###########################
# Temporal Noise & Signal #
###########################
def calc_temporal_noise(batch_flat, master_ubias, master_grad, fixed_GN = None,
                       boxsize = 256, centroid = None):
    """
    Compute temporal noise from a stack of flat frames.

    Parameters
    ----------
    batch_flat : (N, Ny, Nx)
    master_ubias, master_grad : (Ny, Nx)
    fixed_GN : float | None
        If provided, subtracts fixed_GN^2 inside the RMS noise calculation (GN-corrected).
        (keeps original notebook behavior)
    boxsize : int
        Crop size (default 256, matches the notebook cell defining boxsize=256).
    centroid = (xcen, ycen): int | None
        Crop center. Defaults to image center.

    Returns
    -------
    C : float
        Average counts of the raw cropped frames.
    S : float
        Average signal of the corrected cropped frames.
    noise : float
        RMS temporal noise (optionally GN corrected).
    """
    corr = batch_flat - master_ubias - master_grad

    _, Ny, Nx = batch_flat.shape
    if centroid is None: xcen = Nx // 2; ycen = Ny // 2
    else: xcen, ycen = centroid
    bh = boxsize // 2

    flat_clipped = batch_flat[:, ycen - bh : ycen + bh, xcen - bh : xcen + bh]
    corr_clipped = corr[:, ycen - bh : ycen + bh, xcen - bh : xcen + bh]

    C = float(np.mean(flat_clipped))
    E = np.mean(corr_clipped, axis=0)
    S = float(np.mean(E))
    N = corr_clipped - S

    temporal_noise = np.std(N, axis=(1, 2))

    if fixed_GN is None:
        noise = float(np.sqrt(np.mean(temporal_noise ** 2)))
    else:
        val = np.mean(temporal_noise ** 2 - fixed_GN ** 2)
        noise = float(np.sqrt(max(val, 0.0)))

    return C, S, noise