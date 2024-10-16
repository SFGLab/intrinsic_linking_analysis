#!/usr/bin/env python

import os
import argparse
import time
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

import modvis


def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


GLOBALS = {}


def prepare_global_data(path, bead_spacing_bp=500, padding_bp=20_000):
    graph = load_graph(path)
    drawable_graph, minor = modvis.make_drawable_graph(graph)    
    if minor is not None:
        info(f'Minor components detected in the graph: {minor}')
    else:
        info('No minor components detected in the graph')
    bead_coords, restraints, _ = modvis.beads_and_restraints_from_graph(
        drawable_graph, bead_spacing_bp=bead_spacing_bp, padding_bp=padding_bp
    )        
    GLOBALS['graph'] = graph
    GLOBALS['minor'] = minor
    GLOBALS['drawable_graph'] = drawable_graph
    GLOBALS['bead_coords'] = bead_coords
    GLOBALS['minor_restraints'] = restraints
    _, possible_restraints, _ = modvis.beads_and_restraints_from_graph(
        drawable_graph, bead_spacing_bp=bead_spacing_bp, padding_bp=padding_bp, all_restraints=True
    )
    GLOBALS['possible_restraints'] = possible_restraints
    return len(bead_coords)


def _make_initial_structure(n_beads):
    return modvis.make_initial_structure(n_beads, None, None)


def prepare_initial_structures(n_beads, n_structures, n_workers=None):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for _ in range(n_structures):
            futures.append(executor.submit(_make_initial_structure, n_beads))
        results = []
        for res in tqdm(as_completed(futures), total=len(futures)):
            results.append(res.result())
    GLOBALS['initial_structures'] = results


def _run_dropout(path, i_rep, to_keep, restraint_set, max_attempts, modeling_script):    
    try:
        args = (i_rep, to_keep, restraint_set, path)
        bead_coords = GLOBALS['bead_coords']
        restraints = GLOBALS[restraint_set]
        used_restraints = restraints[to_keep]        
        runner = modvis.SpringModelAPI(
            config_file_path='./spring_model_config.ini',
            modeling_command=modeling_script,
            model_working_dir=None
        )
        init_str_points = GLOBALS['initial_structures'][i_rep]
        raw_coords = None
        n_restarts = 0        
        while True:
            try:      
                debug(f'Running "{path}", rep={i_rep}, attempt {n_restarts + 1}/{max_attempts}')          
                raw_coords, _ = runner.run_modeling(
                    bead_coords, used_restraints, groups=None, overwrite=True,
                    container_name=f'md-soft-{os.path.basename(path)}',
                    init_str_points=init_str_points
                )
                break
            except Exception as e:
                n_restarts += 1
                debug(f'Will restart "{path}" for the {n_restarts} time because: {e}')
                if n_restarts >= max_attempts:
                    raise e
                continue
        debug(f'Finished "{path}", rep={i_rep}, restarts={n_restarts}')
        result = args, (bead_coords, used_restraints, raw_coords, init_str_points), None, n_restarts
        retval = path, None, n_restarts
    except Exception as e:
        debug(f'Failed "{path}": {e}, n_restarts={n_restarts}/{max_attempts}')
        result = args, None, e, n_restarts
        retval = path, e, n_restarts
    finally:
        try:
            with open(f'{path}.pkl', 'wb') as f:
                pickle.dump(result, f)
            debug(f'Saved result for "{path}".')
        except Exception as e_io:
            warn(f'Failed to save result for "{path}": {e_io}')
            retval = path, e_io, n_restarts
    return retval


def run_dropout_batch_modeling(
        name, output_dir,        
        reps_per_dropout,
        max_attempts,
        modeling_script,
        n_workers=None,
        reps_idx_start=0
    ):        
    has_minor = GLOBALS['minor'] is not None
    n_possible_restraints = GLOBALS['possible_restraints'].shape[0]    
    n_minor_restraints = 10  # this is a constant derived from the number of edges in a K6 graph
    if has_minor:
        assert GLOBALS['minor_restraints'].shape[0] == n_minor_restraints    
    info(f'Number of minor restraints: {n_minor_restraints}')
    info(f'Number of possible restraints: {n_possible_restraints}')
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i_rep in range(reps_idx_start, reps_idx_start + reps_per_dropout):
            for k in range(1, n_minor_restraints + 1):
                if has_minor:
                    to_keep = np.random.choice(n_minor_restraints, k, replace=False)
                    to_keep.sort()
                    futures.append(executor.submit(
                        _run_dropout,
                        os.path.join(output_dir, f'{name}_minor_k{k:02d}_{i_rep:05d}'),
                        i_rep, to_keep, 'minor_restraints',
                        max_attempts, modeling_script
                    ))
                to_keep_control = np.random.choice(n_possible_restraints, k, replace=False)
                to_keep_control.sort()
                group = 'control' if has_minor else 'nolink'
                futures.append(executor.submit(
                    _run_dropout,
                    os.path.join(output_dir, f'{name}_{group}_k{k:02d}_{i_rep:05d}'),
                    i_rep, to_keep_control, 'possible_restraints',
                    max_attempts, modeling_script
                ))
        info(f'Running {len(futures)} tasks.')
        results = {}
        for res in tqdm(as_completed(futures), total=len(futures)):
            path, e, n_restarts = res.result()
            results[path] = (e, n_restarts)
    return results


def info(s):
    logging.info(s)
    print(s)


def warn(s):
    logging.warning(s)
    print(s)


def debug(s):
    logging.debug(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_file')
    parser.add_argument('--batch_dir', '-o', default='./batch')    
    parser.add_argument('--n_reps', '-n', default=10, type=int)
    parser.add_argument('--n_jobs', '-j', default=8, type=int)
    parser.add_argument('--max_attempts', '-a', default=5, type=int)
    parser.add_argument('--modeling_script', '-s', default='./run_sm_docker_GPU.sh')
    parser.add_argument('--bead_spacing', default=500, type=int)
    parser.add_argument('--padding', default=20_000, type=int)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()    
    
    graph_file = args.graph_file
    if not os.path.exists(graph_file):
        print(f'Graph file {graph_file} does not exist')
        return
    model_name = os.path.splitext(os.path.basename(graph_file))[0]
    output_dir = os.path.join(args.batch_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)        
    print(f'Output will be saved in "{output_dir}".')
        
    logfile = os.path.join(output_dir, f'{model_name}.log')
    logging.basicConfig(
        filename=logfile,
        filemode='w',
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(asctime)s:%(levelname)s]%(message)s',
        datefmt='%H:%M:%S'
    )
    info(f'Logging to {logfile}.')
    info(f'Output will be saved in "{output_dir}".')
    assert os.path.exists(logfile)

    info(f'Reading in data from "{graph_file}"')
    n_beads = prepare_global_data(graph_file, args.bead_spacing, args.padding)
    info(f'Data loaded for {model_name}')

    info(f'Preparing {args.n_reps} initial structures with {n_beads} beads.')
    prepare_initial_structures(n_beads, args.n_reps, args.n_jobs)
    info(f'Initial structures prepared: {len(GLOBALS["initial_structures"])}')

    info(f'Running batch dropout modeling for model {model_name} for {args.n_reps} repetitions.')
    info(f'Using {args.n_jobs} workers')
    info(f'Using bead spacing of {args.bead_spacing} and padding of {args.padding}')    
    t = time.time()
    batch_results = run_dropout_batch_modeling(
        model_name, output_dir,
        reps_per_dropout=args.n_reps,
        n_workers=args.n_jobs,
        max_attempts=args.max_attempts,
        modeling_script=args.modeling_script
    )
    info(f'Done in {time.time() - t:.1f} seconds')
    tot_restarts = sum(n for _, n in batch_results.values())
    n_ok = len([e for e, _ in batch_results.values() if e is None])
    info(f'Successful runs: {n_ok}/{len(batch_results)}. Total restarts: {tot_restarts}')


if __name__ == '__main__':
    main()
