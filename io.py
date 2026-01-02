"""
I/O helpers for CCD measurement notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd
import concurrent.futures #Threading
from tqdm import tqdm


PathLike = Union[str, Path]


def load_frame(path, header = False):
    """
    Load a single frame from a file.
    
    Parameters
    ----------
    path : str | Path
        File path.
    header : bool
        If True, returns (hdr, data) where hdr is a list:
        [Quality, Gain, Exposure Time, Sensor Temperature]
        Otherwise returns data only.

    Returns
    -------
    np.ndarray | Tuple[list, np.ndarray]
    """
    path = Path(path)

    # Image data: skip first 4 lines, comma-separated
    tmp = pd.read_csv(path, sep=",", header=None, skiprows=4)
    tmp.dropna(axis=1, inplace=True)
    data = tmp.values

    if not header:
        return data

    # Header: read the first 4 lines, colon-separated
    df = pd.read_csv(path, sep=":", header=None, nrows=4)

    # Keep original positional mapping (matches the user's existing code)
    # Quality, Gain, Exposure Time, Sensor Temperature
    # Cast what is known numeric; leave others as-is.
    q = df.iloc[1, 1]
    g = df.iloc[2, 1]
    exptime = float(df.iloc[0, 1])
    temp = float(df.iloc[3, 1])
    hdr = [q, g, exptime, temp]
    return hdr, data


def frame_bulk_loader(frame_list, label = None, header = False):
    """
    Load multiple frames with a ThreadPoolExecutor.

    Parameters
    ----------
    frame_list : list-like
        Paths to frame files.
    label : str | None
        tqdm label.
    header : bool
        If True, also loads the 4-line header for each frame and returns a header DataFrame.

    Returns
    -------
    np.ndarray | Tuple[pd.DataFrame, np.ndarray]
    """
    frame_list = [str(Path(p)) for p in frame_list]  # keep executor picklable
    if header:
        hdrs = pd.DataFrame(columns=["Quality", "Gain", "Exposure", "Temperature"])
        data = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(lambda f: load_frame(f, header=True), frame_list),
                     total=len(frame_list),
                     desc=label))
        for hdr, arr in results:
            hdrs.loc[len(hdrs)] = hdr
            data.append(arr)
        return hdrs, np.array(data)
    
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(tqdm(executor.map(load_frame, frame_list), total=len(frame_list), desc=label))
        return np.array(data)
