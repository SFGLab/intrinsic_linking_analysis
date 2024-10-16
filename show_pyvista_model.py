#!/usr/bin/env python

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv

import modvis


def groups_loop_pair(restraints, total_beads, loop_A, loop_B):
    g = [
        (restraints[loop_A, 0], restraints[loop_A, 1], 1),
        (restraints[loop_B, 0], restraints[loop_B, 1], 2)
    ]
    if g[0][0] > 0:
        g = [(0, g[0][0] - 1, 0)] + g
    if g[-1][1] < total_beads - 1:
        g = g + [(g[-1][1] + 1, total_beads - 1, 0)]
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('loop_pair', type=int, nargs=2)
    parser.add_argument('--windowsize', type=int, nargs=2, default=[800, 600])
    args = parser.parse_args()
    
    model_file = args.model_file
    example_loop_pair = tuple(args.loop_pair)
    windowsize = list(args.windowsize)

    if os.path.exists(model_file):
        coords, restraints, bead_groups = modvis.load_model(model_file)
        print(f'Loaded model from {model_file}')
    else:
        print(f'Model not loaded, path "{model_file}" does not exist.')

    minor_palette = modvis.model_colors_palette("light:#1953c8", pad_l=3)
    link_palette = [minor_palette[0], '#bd2828', '#2abb30', minor_palette[0]]

    norm_coords = modvis.scale_coords(coords)
    plotter = pv.Plotter(shape=(1, 2))
    _strand_radius = 0.015
    _interaction_radius = 0.015
    _interaction_color = 'orange'
    plotter.subplot(0, 0)
    modvis.plot_model(
        norm_coords, restraints, bead_groups=bead_groups,
        strand_radius=_strand_radius, interaction_radius=_interaction_radius,
        interaction_color=_interaction_color,
        palette=minor_palette,
        plotter=plotter
    )
    plotter.subplot(0, 1)
    modvis.plot_model(
        norm_coords, restraints,
        bead_groups=groups_loop_pair(restraints, len(norm_coords), *example_loop_pair),
        interaction_color=_interaction_color,
        selected_groups=[1, 2], nonselected_opacity=0.1,
        strand_radius=_strand_radius, interaction_radius=_interaction_radius,
        restraints_only_within_groups=True,
        continous=False,
        palette=link_palette,
        plotter=plotter
    )
    plotter.link_views()
    plotter.show(window_size=windowsize)


if __name__ == '__main__':
    main()
