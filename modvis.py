import os
import pickle
import contextlib
import tempfile
import shutil
import subprocess

import matplotlib.patches
import numpy as np
import pandas as pd
import networkx as nx  # For drawing capabilities
import igraph as ig
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv

from structuregenerator.generator import self_avoiding_random_walk
from structuregenerator.generator import save_points_as_pdb


def beads_and_restraints_from_graph(graph, bead_spacing_bp, padding_bp=0, bounds=None, all_restraints=False):
    assert padding_bp >= 0
    assert bounds is None or bounds[0] < bounds[1]
    sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1]['coord'])
    if bounds is not None:
        min_coord = bounds[0] - padding_bp
        max_coord = bounds[1] + padding_bp
    else:
        min_coord = sorted_nodes[0][1]['coord'] - padding_bp - bead_spacing_bp // 2
        max_coord = sorted_nodes[-1][1]['coord'] + padding_bp + bead_spacing_bp // 2
    bead_coords = np.arange(min_coord, max_coord, bead_spacing_bp)

    def coord_to_bead_idx(coord):
        return int(round((coord - min_coord) / bead_spacing_bp))

    restraints = []
    for u, v, data in graph.edges(data=True):
        if data['is_contact'] and (all_restraints or data['end_segments']):
            if v < u:
                u, v = v, u
            x1 = graph.nodes[u]['coord']
            x2 = graph.nodes[v]['coord']
            if x1 >= min_coord and x1 <= max_coord and x2 >= min_coord and x2 <= max_coord:
                restraints.append((coord_to_bead_idx(x1), coord_to_bead_idx(x2)))
    restraints = np.array(sorted(restraints))

    bead_groups = []
    u0 = sorted_nodes[0]
    prev_seg = u0[1]['segment']
    start_i = coord_to_bead_idx(u0[1]['coord'])
    for u in sorted_nodes[1:]:
        seg = u[1]['segment']
        if seg != prev_seg:
            i_bead = coord_to_bead_idx(u[1]['coord'])
            bead_groups.append((start_i, i_bead - 1, prev_seg))
            prev_seg = seg
            start_i = i_bead
    bead_groups.append((start_i, bead_coords.shape[0] - 1, prev_seg))

    return bead_coords, restraints, bead_groups


def make_initial_structure(bead_coords, restraints=None, bead_groups=None):
    if isinstance(bead_coords, int):
        n_beads = bead_coords
    else:        
        n_beads = bead_coords.shape[0]
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                raw_coords = self_avoiding_random_walk(n_beads)
    return raw_coords


def save_restraints_to_rst(path, restraints):
    np.savetxt(path, restraints, fmt=':%d\t:%d')


def load_restraints_from_rst(path):
    return np.loadtxt(path, converters=lambda x: int(x[1:]), dtype=int)


def save_chimera_coloring(bead_groups, path, palette, min_str_file='min_str.pdb'):
    with open(path, 'w') as f:
        f.write(f'open {min_str_file}\n')
        f.write(f'background solid white\n')
        f.write(f'color #878787 :.A\n')
        for i, (start, end, _) in enumerate(bead_groups):
            color = matplotlib.colors.to_hex(palette[i])
            f.write(f'color {color} :{start}-{end}\n')


def load_structure_from_pdb(path):
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                fields = line.split()
                coords.append(tuple(map(float, fields[6:9])))
    return np.array(coords)


def scale_coords(coords, scale=1.05):
    coords = coords.copy()
    for i in range(3):
        coords[:, i] -= (coords[:, i].max() + coords[:, i].min()) / 2
    coords /= np.abs(coords).max() * scale
    return coords


def save_model(path, coords, restraints, bead_groups=None):
    with open(path, 'wb') as f:
        pickle.dump((coords, restraints, bead_groups), f)
    

def load_model(path):
    with open(path, 'rb') as f:
        coords, restraints, bead_groups = pickle.load(f)
    return coords, restraints, bead_groups


def bead_groups_vector(bead_groups):
    n_beads = bead_groups[-1][1] + 1
    groups = np.full(n_beads, -1, dtype=int)
    for s, e, i in bead_groups:
        groups[s:e + 1] = i
    return groups


def plot_model(
    coords, restraints,
    bead_groups=None,
    palette=None, interaction_color='fuchsia', strand_radius=0.01, interaction_radius=0.01,
    selected_groups=None, nonselected_opacity=None,
    restraints_only_in_selected_groups=True,    
    restraints_only_within_groups=False,
    continous=True,
    plotter=None
):
    n_points = coords.shape[0]
    _plt = pv.Plotter() if plotter is None else plotter
    if palette is None:
        palette = model_colors_palette()    
    if selected_groups is not None:
        assert bead_groups is not None
    elif bead_groups is not None:
        selected_groups = set(i for _, _, i in bead_groups)
    if bead_groups is None:
        spline = pv.Spline(coords, n_points).tube(radius=strand_radius)
        _plt.add_mesh(spline, scalars='arc_length', show_scalar_bar=False)    
    else:                        
        
        def _segment(s, e, color, op):
            spline = pv.Spline(coords[s:e + 1], e - s + 1).tube(radius=strand_radius)
            _plt.add_mesh(spline, color=color, opacity=op)

        prev_selected = False
        for s, e, i in bead_groups:            
            if i in selected_groups or nonselected_opacity is not None:                
                if continous or i not in selected_groups:
                    e = min(e + 1, n_points - 1)
                if not continous and prev_selected:
                    if i in selected_groups:
                        if nonselected_opacity is not None:                            
                            _segment(s - 1, s, palette[0], nonselected_opacity)
                    else:
                        s -= 1                
                op = 1.0 if i in selected_groups else nonselected_opacity
                _segment(s, e, palette[i], op)
            prev_selected = i in selected_groups
    g = bead_groups_vector(bead_groups)
    for i in range(restraints.shape[0]):
        u, v = restraints[i]
        if restraints_only_in_selected_groups and not (  # only restraints within selected groups
            g[u] in selected_groups and g[v] in selected_groups
        ):
            continue
        if restraints_only_within_groups and g[u] != g[v]:
            continue
        tube = pv.Tube(coords[u], coords[v], radius=interaction_radius)
        _plt.add_mesh(tube, color=interaction_color)
    if plotter is None:
        _plt.show()


class SpringModelAPI(object):
    def __init__(self, modeling_command, config_file_path='', model_working_dir=None):        
        self.spring_model_run_command = modeling_command.split()
        self.config_file_path = config_file_path
        self.model_working_dir=model_working_dir

    def _run(self, command, *args):
        full_command = command + [str(arg) for arg in args]
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        error = result.stderr.strip()        
        return output, error

    def prepare_modeling_files(self, working_dir, init_str_points, restraints, bead_groups=None, config=None, overwrite=False):
        os.makedirs(working_dir, exist_ok=True)
        if os.path.exists(init_str_points) and not overwrite:
            raise FileExistsError(f'File {init_str_points} already exists, use overwrite=True to overwrite.')

        init_str_file = os.path.join(working_dir, f'init_str.pdb')
        if os.path.exists(init_str_points) and not overwrite:
            raise FileExistsError(f'File {init_str_points} already exists, use overwrite=True to overwrite.')
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    save_points_as_pdb(init_str_points, init_str_file)

        target_config_file = os.path.join(working_dir, f'config.ini')
        if os.path.exists(target_config_file) and not overwrite:
            raise FileExistsError(f'File {target_config_file} already exists, use overwrite=True to overwrite.')
        if config is None:        
            config = self.config_file_path        
        shutil.copy(config, target_config_file)

        restraints_file = os.path.join(working_dir, f'restraints.rst')
        if os.path.exists(restraints_file) and not overwrite:
            raise FileExistsError(f'File {restraints_file} already exists, use overwrite=True to overwrite.')
        save_restraints_to_rst(restraints_file, restraints)

        if bead_groups is not None:
            palette = model_colors_palette()
            coloring_file = os.path.join(working_dir, f'prepare_model.cmd')
            if os.path.exists(coloring_file) and not overwrite:
                raise FileExistsError(f'File {coloring_file} already exists, use overwrite=True to overwrite.')
            save_chimera_coloring(bead_groups, coloring_file, palette)

    def run_modeling(self, bead_coords, restraints, groups, overwrite=False, container_name=None, init_str_points=None):    
        if init_str_points is None:
            init_str_points = make_initial_structure(bead_coords)
        dir_context = contextlib.nullcontext(self.model_working_dir) if self.model_working_dir is not None else tempfile.TemporaryDirectory()
        with dir_context as model_working_dir:
            self.prepare_modeling_files(model_working_dir, init_str_points, restraints, bead_groups=groups, overwrite=overwrite)
            self.execute_spring_model(model_working_dir, container_name)
            raw_coords = self.get_modeling_results(model_working_dir)
        return raw_coords, init_str_points
    
    def run_modeling_multistep(self, bead_coords, restraints_steps, config_steps):
        init_str_points = make_initial_structure(bead_coords)
        dir_context = contextlib.nullcontext(self.model_working_dir) if self.model_working_dir is not None else tempfile.TemporaryDirectory()        
        with dir_context as model_working_dir:
            step_str_points = [init_str_points]
            for i_step, (restraints, config) in enumerate(zip(restraints_steps, config_steps)):
                self.prepare_modeling_files(model_working_dir, step_str_points[-1], restraints, config=config, overwrite=True)
                self.execute_spring_model(model_working_dir)
                step_str_points.append(self.get_modeling_results(model_working_dir))
        return step_str_points

    def execute_spring_model(self, models_dir, container_name='md-soft'):
        try:
            out, err = self._run(self.spring_model_run_command, models_dir, container_name)
        except subprocess.CalledProcessError as e:
            raise Exception(f'Error during modeling: {e.stderr}')
        return out, err

    def get_modeling_results(self, models_dir):
        min_str_points = load_structure_from_pdb(os.path.join(models_dir, 'min_str.pdb'))
        return min_str_points


def _get_edges(g: ig.Graph, minor_only: bool):    
    for e in g.es():
        if not e['is_contact']:
            continue
        if (len(e['minors']) == 0) == minor_only:
            continue
        u, v = e.source, e.target
        yield u, v, g.vs[u]['coord'], g.vs[v]['coord'], float(e['petcount'])


def get_minor_edges(g, minor_only=True):
    _tmp = pd.DataFrame.from_records(_get_edges(g, minor_only))
    _tmp.columns = ['node_id_A', 'node_id_B', 'coord_A', 'coord_B', 'weight']
    return _tmp


def clear_ax(ax):
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)


def model_colors_palette(pal="tab10", n=6, pad_l=0, pad_r=0):
    _nogroup_color = np.array([.5, .5, .5]).reshape(3)
    _pal = sns.color_palette(pal, n + pad_l + pad_r)
    colors = [_nogroup_color] + [
        np.array(col).reshape(3) for col in _pal[pad_l:(n + pad_l - pad_r)]
    ] + [_nogroup_color]
    return colors


def make_drawable_graph(graph: ig.Graph, minor_idx: int | None = None):
    minor = None
    for u in graph.vs():
        for m in u['minors']:
            if minor_idx is None or m.idx == minor_idx:
                minor_idx = m.idx  # in case it was None
                minor = m
                break
        if minor is not None:
            break
    else:
        if minor_idx is not None:
            raise ValueError(f'Not found {minor_idx}')
    g = nx.MultiDiGraph()
    for u in graph.vs():
        d = dict(u.attributes())
        if minor in d['minors']:
            d['segment'] = d['idx_in_minor'][d['minors'].index(minor)] + 1
        else:
            d['segment'] = 0
        g.add_node(u.index, **d)
    for e in graph.es():
        d = dict(e.attributes())
        if minor in d['minors']:
            _ends = d['idx_in_minor'][d['minors'].index(minor)]            
            d['end_segments'] = (_ends[0] + 1, _ends[1] + 1)
        else:
            d['end_segments'] = None
        g.add_edge(e.source, e.target, **d)
    return g, minor


def plot_graph_with_minor(
    g: nx.MultiDiGraph | ig.Graph, minor_idx: int | None = None,
    arc_color='gray', node_size=20, node_spacing=1.0,
    arc_linewidth=1.5, strand_linewidth=1.0, rad=-0.5, arc_alpha = 0.3,
    selected_arc_linewidth=None, selected_arc_alpha = None,
    layout=None, draw_strands=False,
    highlighted_loops=None, segments_palette=None,
    visible_segments=None,
    tick_y_coords=(-0.001, -0.003, -0.005), tick_linewidth=1.0, tick_fontsize=14,
    ax=None
):
    if not isinstance(g, nx.MultiDiGraph):
        g, minor = make_drawable_graph(g, minor_idx=minor_idx)
    else:
        g = g.copy()
        minor = None    
    n = g.number_of_nodes()

    if layout is None:
        pos = {i: (float(i * node_spacing), 0.0) for i in range(n)}
    else:
        pos = getattr(nx.layout, f'{layout}_layout')(g)

    if visible_segments is not None:
        to_remove = [
            u for u in g.nodes()
            if g.nodes[u]['segment'] not in visible_segments
        ]
        for u in to_remove:
            g.remove_node(u)

    strand_edges = []    
    normal_arcs = []
    selected_arcs = []
    for u, v, d in g.edges(data=True):
        e = (u, v)
        if d['is_contact']:
            if d['end_segments']:
                selected_arcs.append(e)                
            else:
                normal_arcs.append(e)
        if d['is_strand']:
            strand_edges.append(e)    

    if highlighted_loops is None:
        highlighted_loops = {}
    highlighted_arcs = {}
    for i_loop, color in highlighted_loops.items():                
        highlighted_arcs[selected_arcs[i_loop]] = color
    for e in highlighted_arcs.keys():
        selected_arcs.remove(e)

    normal_nodes = []
    highlighted_nodes = defaultdict(list)
    for u in g.nodes():
        for e in highlighted_arcs.keys():
            if u >= e[0] and u <= e[1]:
                highlighted_nodes[highlighted_arcs[e]].append(u)
                break
        else:
            normal_nodes.append(u)

    if ax is None:
        ax = plt.gca()
    
    nx.draw_networkx_nodes(
        g, pos, normal_nodes, node_color='gray', node_size=node_size, ax=ax
    )
    for color, nodes in highlighted_nodes.items():
        nx.draw_networkx_nodes(
            g, pos, nodes, node_color=color, node_size=node_size, ax=ax
        )

    connectionstyle = f'arc3,rad={rad:.3f}' if rad is not None else 'arc3'
    strand_connectionstyle = 'arc3'
    selected_arc_linewidth = 2 * arc_linewidth if selected_arc_linewidth is None else selected_arc_linewidth
    selected_arc_alpha = 2 * arc_alpha if selected_arc_alpha is None else selected_arc_alpha
    if draw_strands:        
        nx.draw_networkx_edges(
            g, pos, strand_edges, edge_color=arc_color, width=strand_linewidth, ax=ax,
            arrowstyle = '-', connectionstyle=strand_connectionstyle, alpha = arc_alpha
        )
    nx.draw_networkx_edges(
        g, pos, normal_arcs, edge_color=arc_color, width=arc_linewidth, ax=ax,
        arrowstyle = '-', connectionstyle=connectionstyle, alpha = arc_alpha,
        node_size=0
    )    
    nx.draw_networkx_edges(
        g, pos, selected_arcs, edge_color='black', width=arc_linewidth, ax=ax,
        arrowstyle = '-', connectionstyle=connectionstyle, alpha = selected_arc_alpha,
        node_size=0
    )
    for arc, color in highlighted_arcs.items():
        nx.draw_networkx_edges(
            g, pos, [arc], edge_color=color, width=selected_arc_linewidth, ax=ax,
            arrowstyle = '-', connectionstyle=connectionstyle, alpha = selected_arc_alpha,
            node_size=0
        )

    if segments_palette is not None:
        nodes_by_segment = defaultdict(list)
        for u in g.nodes():
            nodes_by_segment[g.nodes[u]['segment']].append(u)
        for segment, nodes in nodes_by_segment.items():
            if segment == 0:
                continue
            x_min = min(pos[u][0] for u in nodes)
            x_max = max(pos[u][0] for u in nodes)            
            x = (x_min + x_max) / 2
            y0, y1, y2 = tick_y_coords            
            ax.plot([x_min, x_min, x_max, x_max], [y0, y1, y1, y0], color=segments_palette[segment], linewidth=tick_linewidth)
            ax.text(x, y2, f'{segment}', fontsize=tick_fontsize, ha='center', va='center')

    clear_ax(ax)
    return minor
