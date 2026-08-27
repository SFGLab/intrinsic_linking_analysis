import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

import tqdm
import gemmi

from datasources import DataSources

from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


RESULTS_DIR = './results/loopsage/experiment_2'

ds = DataSources(
    RESULTS_DIR,
    ['cell_line', 'chromosome', 'start', 'end', 'ccd_id', 'evp', 'rep']
).add(
    'init_struct_file', r'(?P<cell_line>GM12878)_(?P<chromosome>chr_?\w+)_(?P<start>\d+)-(?P<end>\d+)_ccd(?P<ccd_id>\d+)/reps/evp_(?P<evp>.+)_rep_(?P<rep>\d+)/LE_init_struct.cif$',
    parsers={'start': int, 'end': int, 'ccd_id': int, 'rep': int, 'evp': float}
)
loopsage_outputs = ds.get_paths_as_dataframe()
loopsage_outputs['ensemble_dir'] = loopsage_outputs.init_struct_file.apply(os.path.dirname)
loopsage_outputs
print(f"Found {len(loopsage_outputs)} loopsage output dirs.")


def read_coords_from_cif(cif_path):
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()

    # Extract the columns
    x = block.find_values('_atom_site.Cartn_x')
    y = block.find_values('_atom_site.Cartn_y')
    z = block.find_values('_atom_site.Cartn_z')

    # Convert to float and stack into a NumPy array
    coords = np.array(list(zip(map(float, x), map(float, y), map(float, z))))
    return coords

def sort_cohesins(X: np.ndarray) -> np.ndarray:
    return X[np.lexsort(X[:, ::-1].T)]

def read_rep(rep_dir: str):
    rep_dir = Path(rep_dir)
    Ms = np.load(rep_dir / 'other' / 'Ms.npy')
    Ns = np.load(rep_dir / 'other' / 'Ns.npy')
    n_cohesins = Ms.shape[0]
    n_frames = Ms.shape[1]
    frames = [None for _ in range(n_frames)]
    for structure_file in os.listdir(rep_dir / 'ensemble'):
        m = re.match(r'MDLE_(?P<frame_idx>\d+)\.cif$', structure_file)
        if not m:
            continue
        i = int(m.group('frame_idx')) - 1
        try:
            coords = read_coords_from_cif(rep_dir / 'ensemble' / structure_file)
        except Exception as e:
            print(f"Error reading {structure_file}: {e}")
            continue
        assert frames[i] is None, f"Duplicate frame index {i - 1} in {rep_dir}"
        cohesins = np.stack([
            Ms[:, i], Ns[:, i]
        ]).T
        cohesins = sort_cohesins(cohesins)
        frames[i] = coords, cohesins
    return frames

# read_rep(loopsage_outputs.iloc[0].ensemble_dir)

def _make_pickle(args):
    idx, rep_dir, cache_file = args    
    try:        
        frames = read_rep(rep_dir)
        with open(cache_file, 'wb') as f:
            obj = {'index': idx, 'frames': frames}
            pickle.dump(obj, f)
        return idx, None
    except Exception as e:
        return idx, e    

def make_pickles(rep_dirs: pd.Series, update=True, max_workers=10):
    cached = 0
    updated = 0
    errors = 0
    total = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []            
        for idx, rep_dir in rep_dirs.items():            
            cache_file = Path(rep_dir) / 'frames.pkl'
            if cache_file.exists() and not update:         
                cached += 1
            else:
                futures.append(executor.submit(_make_pickle, (idx, rep_dir, cache_file)))                
            total += 1
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                idx, res = future.result()
                if isinstance(res, Exception):
                    exc = res
                    errors += 1
                    print(f"Error in future: {exc} at index {idx}")
                else:
                    updated += 1
            except Exception as e:
                print(f'Uncaught exception during processing: {e}')
    print(f"Processed {total} directories: {cached} cached, {updated} updated, {errors} errors.")    

make_pickles(loopsage_outputs.ensemble_dir, max_workers=60, update=True)