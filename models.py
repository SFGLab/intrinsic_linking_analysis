import os
import pickle
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import networkx as nx  # For drawing capabilities
import igraph as ig
from tqdm import tqdm
from collections import defaultdict

import datasources
import knots_tools
import gaussian_linking

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv


def gen_restraints(graph, resolution, pad = 0, bounds = None):
    assert pad >= 0
    assert bounds is None or bounds[0] < bounds[1]
    node_coords = np.array(list(nx.get_node_attributes(graph, 'coord').values()))
    if bounds is not None:
        min_coord = bounds[0] - pad
        max_coord = bounds[1] + pad
    else:
        min_coord = node_coords[0] - pad - resolution // 2
        max_coord = node_coords[-1] + pad + resolution // 2
    bead_coords = np.arange(min_coord, max_coord, resolution)
    bead_index = (node_coords - min_coord) // resolution

    groups = []
    current_group = -1
    start = -1
    end = -1
    for bi, (_, gi) in zip(bead_index, graph.nodes(data='group_idx')):
        if gi != current_group:
            if current_group != -1:
                groups.append((start, end, current_group))
            current_group = gi
            start = bi
        end = bi

    restraints = pd.DataFrame.from_records(
        (bead_index[u], bead_index[v])
        for u, v, data in graph.edges(data=True)
        if (
            data['is_contact']\
            and data['idx_in_minor']\
            and node_coords[u] >= min_coord and node_coords[u] <= max_coord\
            and node_coords[v] >= min_coord and node_coords[v] <= max_coord
        )
    )
    restraints.columns = ['ibead1', 'ibead2']
    restraints = restraints.drop_duplicates()
    restraints = restraints.sort_values(['ibead1', 'ibead2'])
    restraints = restraints.to_numpy()
    return bead_coords, restraints, groups


def save_chimera_restraints(path, restraints):
    np.savetxt(path, restraints, fmt=':%d\t:%d')


def save_coloring(groups, path, colors, min_str_file='min_str.pdb'):
    with open(path, 'w') as f:
        f.write(f'open {min_str_file}\n')
        f.write(f'background solid white\n')
        f.write(f'color #878787 :.A\n')
        for start, end, gi in groups:
            color = matplotlib.colors.to_hex(colors[gi])
            f.write(f'color {color} :{start}-{end}\n')


def load_structure_from_pdb(path):
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                fields = line.split()
                coords.append(tuple(map(float, fields[6:9])))
    return np.array(coords)


def normalize_coords(coords, scale=1.05):
    coords = coords.copy()
    for i in range(3):
        coords[:, i] -= (coords[:, i].max() + coords[:, i].min()) / 2
    coords /= (np.abs(coords).max() * scale)
    return coords


def plot_model(X, R, color=None, label=None, radius=0.01):        
    n_points = X.shape[0]
    spline = pv.Spline(X, n_points).tube(radius=radius)
    spline.plot(
        scalars='arc_length',
        show_scalar_bar=False
    )


class SpringModelAPI(object):
    def __init__(self, config_file_path, modeling_command, initial_structure_command, working_directory_root):
        self.config_file_path = config_file_path        
        self.working_directory_root = working_directory_root
        self.modeling_command = modeling_command.split()
        self.initial_structure_command = initial_structure_command.split()

    def _run(self, command, *args):
        full_command = command + [str(arg) for arg in args]      
        cmd_str = ' '.join(full_command)
        result = subprocess.run(
            full_command,
            capture_output=True,
            check=True
        )
        output = result.stdout.decode().strip()
        error = result.stderr.decode().strip()        
        return output, error

    def prepare_modeling_files(self, nx_graph, name, resolution = 100, padding = 10_000, make_chimera_files = True):
        bead_coords, restraints, groups = gen_restraints(nx_graph, resolution, padding)

        models_dir = os.path.join(self.working_directory_root, name)        
        os.makedirs(models_dir, exist_ok=True)

        if make_chimera_files:
            restraints_file = os.path.join(models_dir, f'restraints.rst')
            save_chimera_restraints(restraints_file, restraints)
            palette = sns.color_palette('tab10', 6)
            coloring_file = os.path.join(models_dir, f'prepare_model.cmd')
            save_coloring(groups, coloring_file, palette)
        else:
            restraints_file = None 
            coloring_file = None
                    
        init_str_file = os.path.join(models_dir, f'init_str.pdb')

        config_file = os.path.join(models_dir, f'config.ini')
        shutil.copy(self.config_file_path, config_file)

        return bead_coords, restraints, groups, init_str_file, models_dir, config_file, restraints_file, coloring_file

    def run_modeling(self, n_beads, init_str_file, models_dir):
        try:
            out1, err1 = self._run(self.initial_structure_command, n_beads, '-o', init_str_file)        
            out2, err2 = self._run(self.modeling_command, models_dir)    
        except subprocess.CalledProcessError as e:
            raise Exception(f'Error during modeling: {e.stderr.decode()}')
        return out1 + '\n' + out2, err1 + '\n' + err2

    def get_modeling_results(self, models_dir):
        raw_coords = load_structure_from_pdb(os.path.join(models_dir, 'min_str.pdb'))
        return raw_coords
