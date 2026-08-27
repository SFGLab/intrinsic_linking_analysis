import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

import tqdm
from dataclasses import dataclass, field
from typing import List

from datasources import DataSources
import topoly

from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time
from collections import defaultdict


RESULTS_DIR = './results/loopsage/experiment_2'

def extract_graphlet(cohesins: np.ndarray):
    positions = {pos: i for i, pos in enumerate(np.unique(cohesins))}
    assert len(positions) == cohesins.size, "Cohesin positions must be unique"
    graphlet = np.empty_like(cohesins)
    for i in range(cohesins.shape[0]):
        graphlet[i, 0] = positions[cohesins[i, 0]]
        graphlet[i, 1] = positions[cohesins[i, 1]]
    return graphlet

def get_big_loops(cohesins: np.ndarray, min_size: int = 5) -> np.ndarray:
    """
    Extract loops from cohesin indices that are larger than a specified size.
    """
    big_loops = []
    n = cohesins.shape[0]
    for i in range(n):
        s0, e0 = cohesins[i, 0], cohesins[i, 1]
        if e0 - s0 >= min_size:
            big_loops.append(cohesins[i])
    return np.array(big_loops)

def nonoverlapping_indices(bead_indices):
    non_overlapping = []
    n = bead_indices.shape[0]

    for i in range(n):
        s0, e0 = bead_indices[i, 0], bead_indices[i, 1]
        for j in range(i + 1, n):
            s1, e1 = bead_indices[j, 0], bead_indices[j, 1]
            assert s0 < e0 and s1 < e1, "Start must be less than end for both loops"
            assert s0 < s1, "Start of first loop must be less than start of second loop"
            if s1 > e0:
                non_overlapping.append((i, j))

    return non_overlapping

def get_curve(coords, start, end, closed=True):
    n = end - start + 1
    m = n + 1 if closed else n
    curve_coords = np.empty((m, 3), dtype=coords.dtype)
    curve_coords[:n, :] = coords[start:end + 1, :]
    if closed:
        curve_coords[-1, :] = curve_coords[0, :]
    return curve_coords

def topoly_fmt(curve):
    return curve.tolist()

def calculate_linking_number(coords: np.ndarray, cohesins: np.ndarray, loop_pairs: np.ndarray, threshold=0.1):
    lns = []
    for i, j in loop_pairs:
        s0, e0 = cohesins[i, 0], cohesins[i, 1]
        s1, e1 = cohesins[j, 0], cohesins[j, 1]
        curve_a = get_curve(coords, s0, e0)
        curve_b = get_curve(coords, s1, e1)
        ln = topoly.gln(topoly_fmt(curve_a), topoly_fmt(curve_b))
        if abs(ln) < threshold:
            continue
        lns.append((i, j, s0, e0, s1, e1, ln))
    return pd.DataFrame(lns, columns=['idx_loop_A', 'idx_loop_B', 'idx_start_A', 'idx_end_A', 'idx_start_B', 'idx_end_B', 'linking_number'])    

def sort_cohesins(X: np.ndarray) -> np.ndarray:
    return X[np.lexsort(X[:, ::-1].T)]

def get_linking_numbers_from_frame(coords: np.ndarray, cohesins: np.ndarray, min_size: int, threshold: float):
    """
    Calculate linking numbers for a given frame of coordinates and cohesins.
    """
    cohesins = sort_cohesins(cohesins)
    big_loops = get_big_loops(cohesins, min_size=min_size)
    if big_loops.size == 0:
        return pd.DataFrame()
    non_overlapping = nonoverlapping_indices(big_loops)
    if not non_overlapping:
        return pd.DataFrame()
    linking_numbers = calculate_linking_number(coords, big_loops, non_overlapping, threshold=threshold)
    return linking_numbers

def _process_frame(frame, min_size, threshold):
    """
    Process a single frame to extract linking numbers.
    """
    if frame is None:
        return None
    coords, cohesins = frame
    try:
        return get_linking_numbers_from_frame(coords, cohesins, min_size=min_size, threshold=threshold)
    except Exception as e:
        return e

def _process_rep(args):
    idx, frames_file, min_size, threshold, start_frame, frame_interval = args
    try:
        
        with open(frames_file, 'rb') as f:
            obj = pickle.load(f)
        idx_loaded = obj['index']
        frames = obj['frames']
        assert idx_loaded == idx, f"Index mismatch: {idx_loaded} != {idx}"
        result = pd.concat({
            i: _process_frame(frames[i], min_size, threshold)
            for i in range(start_frame, len(frames), frame_interval)
        }, names=['frame_idx']).sort_index()
    except Exception as e:
        return idx, e
    return idx, result

def process_experiment(frames_files, max_workers, min_size, threshold, start_frame=0, frame_interval=1):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, frames_file in frames_files.items():
            if os.path.exists(frames_file):
                futures.append(
                    executor.submit(
                        _process_rep,
                        (idx, frames_file, min_size, threshold, start_frame, frame_interval)
                    )
                )
            else: print(frames_file)
        print(f"Running {len(futures)} tasks")
        _results = {}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                idx, df = future.result()
                if isinstance(df, Exception):
                    exc = df
                    print(f'Error processing {idx}: {exc}')
                if df is not None and not df.empty:
                    _results[idx] = df
            except Exception as e:
                print(f'Uncaught exception during processing: {e}')
    print(f"Got {len(futures)} results")
    res_df = pd.concat(_results, names=['cell_line', 'chromosome', 'start', 'end', 'ccd_id', 'evp', 'rep', 'frame', 'pair_idx'])
    res_df['abs_linking_number'] = res_df['linking_number'].abs()
    return res_df.sort_index()

def main():
    ds = DataSources(
        RESULTS_DIR,
        ['cell_line', 'chromosome', 'start', 'end', 'ccd_id', 'evp', 'rep']
    ).add(
        'init_struct_file', r'(?P<cell_line>GM12878)_(?P<chromosome>chr_?\w+)_(?P<start>\d+)-(?P<end>\d+)_ccd(?P<ccd_id>\d+)/reps/evp_(?P<evp>.+)_rep_(?P<rep>\d+)/LE_init_struct.cif$',
        parsers={'start': int, 'end': int, 'ccd_id': int, 'rep': int, 'evp': float}
    )
    loopsage_outputs = ds.get_paths_as_dataframe()
    loopsage_outputs['ensemble_dir'] = loopsage_outputs.init_struct_file.apply(os.path.dirname)
    loopsage_outputs['frames_file'] = loopsage_outputs.ensemble_dir.apply(lambda x: Path(x) / 'frames.pkl')
    print(loopsage_outputs)

    res_df = process_experiment(
        loopsage_outputs.frames_file,
        max_workers=100, min_size=3, threshold=0.4, start_frame=0, frame_interval=1
    )

    print(f"Processed {len(res_df)} linking numbers, saving")
    res_df.to_csv(Path(RESULTS_DIR) / 'all_linking_numbers.csv')
    print("Done")


if __name__ == '__main__':
    main()
