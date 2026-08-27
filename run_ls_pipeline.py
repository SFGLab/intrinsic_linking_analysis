#!/usr/bin/env python3

import argparse
from collections import namedtuple
import concurrent.futures
import pickle
import json
import os
import re
import gemmi
import shutil
import subprocess
import sys
import time
import traceback
import topoly
import numpy as np
import pandas as pd
import numba
from dataclasses import dataclass
from pathlib import Path


RunIndexTuple = namedtuple(
    'RunIndex',
    ['dataset', 'chromosome', 'ccd_id', 'variant', 'evp', 'rep']
)

@dataclass(frozen=True)
class RunID:
    dataset: str
    chromosome: str
    ccd_id: int
    variant: int
    evp: str
    rep: int

    def __str__(self) -> str:
        return f"{self.data_id}__{self.repl_id}"

    @property
    def ccd_full_id(self):
        return f"{self.chromosome}-ccd{self.ccd_id}"

    @property
    def data_id(self) -> str:
        return f"{self.dataset}_{self.ccd_full_id}_v{self.variant}"

    @property
    def repl_id(self) -> str:
        return f"evp_{self.evp}_rep_{self.rep}"

    @property
    def index(self) -> dict:
        return {
            "dataset": self.dataset,
            "chromosome": self.chromosome,
            "ccd_id": self.ccd_id,
            "variant": self.variant,
            "evp": self.evp,
            "rep": self.rep,
        }


@dataclass(frozen=True)
class RunInfo:
    root: Path
    args: argparse.Namespace
    id: RunID

    @property
    def workdir(self) -> Path:
        return self.root / Path(self.id.data_id)
    
    @property
    def repldir(self) -> Path:
        return self.workdir / Path(self.id.repl_id)

    @property
    def input_loops_path(self) -> Path:
        return self.workdir / f"loopsage_input.bedpe"

    @property
    def frames_pickle_path(self) -> Path:
        return self.repldir / f"frames.pkl"

    @property
    def linking_numbers_path(self) -> Path:
        return self.repldir / f"linking_numbers.csv"

    @property
    def loopsage_log(self) -> Path:
        return self.repldir / f"ls_{self.id}.log"


class PipelineError(RuntimeError):
    """Error raised for an expected pipeline failure."""


def print_message(run_id: RunID, message: str, *, stream=sys.stdout) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{run_id}] {message}", file=stream, flush=True)


def ensure_positive_workers(value: int) -> None:
    if value < 1:
        raise PipelineError("--workers must be at least 1.")


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


class PipelineStage:
    name = 'NOT_SET'

    def __init__(self, number):
        self.number = number        
        self.args = None
        self.tempdir = None

    def is_complete(self, run: RunInfo) -> bool:
        return self.get_marker_path(run).exists()

    def cleanup(self, run: RunInfo) -> None:
        pass

    def action(self, run: RunInfo) -> None:
        pass

    def get_marker_path(self, run: RunInfo) -> Path:
        return run.repldir / f"stage_{self.number}.completed"

    def place_completion_marker(self, run: RunInfo) -> None:
        marker_path = self.get_marker_path(run)
        marker_path.touch()

    def clear_completion_marker(self, run: RunInfo) -> None:
        marker_path = self.get_marker_path(run)
        if marker_path.exists():
            marker_path.unlink()

    def execute(self, args: argparse.Namespace, run: RunInfo, force: bool) -> None:
        self.args = args
        self.tempdir = Path(args.tempdir) if hasattr(args, 'tempdir') else None
        run_id = run.id
        if self.is_complete(run):
            if force:
                self.clear_completion_marker(run)
                print_message(run_id, f"Stage {self.number} ({self.name}) is already complete; forcing re-run.")                
                self.cleanup(run)
            else:
                print_message(run_id, f"Stage {self.number} ({self.name}) is already complete; skipping.")
                return

        started_at = time.monotonic()
        print_message(run_id, f"Entering stage {self.number}: {self.name}")

        try:
            self.action(run)

        except Exception:
            elapsed = time.monotonic() - started_at
            print_message(run_id, f"Stage {self.number} ({self.name}) failed after {elapsed:.1f} seconds.", stream=sys.stderr)
            raise

        self.place_completion_marker(run)
        elapsed = time.monotonic() - started_at
        print_message(run_id, f"Stage {self.number} ({self.name}) completed after {elapsed:.1f} seconds.")


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class StagePrepareData(PipelineStage):
    name = 'prepare_data'

    def __init__(self, number):
        super().__init__(number)

    def load_bedpe_loops(self, path):
        return pd.read_csv(path, sep='\t', header=None)

    def select_ccd_loops(self, df, chrom, start, end):
        df = df[
            (df.iloc[:, 0] == chrom) & (df.iloc[:, 1] >= start) & (df.iloc[:, 2] <= end) & \
            (df.iloc[:, 3] == chrom) & (df.iloc[:, 4] >= start) & (df.iloc[:, 5] <= end)
        ].sort_values(list(range(6))).reset_index(drop=True)
        return df

    def action(self, run: RunInfo) -> None:
        loops_source_file = Path(self.args.source_data)
        loops_output_file = run.input_loops_path  # input as in "to loopsage"

        if not loops_source_file.is_file():
            raise PipelineError(f"Source data file does not exist: {loops_source_file}")
        try:
            os.makedirs(run.repldir, exist_ok=True)
            if loops_output_file.is_file():
                return
            
            loops_pickle_file = self.tempdir / loops_source_file.with_suffix('.pkl').name
            if loops_pickle_file.is_file():            
                with open(loops_pickle_file, 'rb') as f:
                    loops_df = pickle.load(f)
            else:
                loops_df = self.load_bedpe_loops(loops_source_file)
                with open(loops_pickle_file, 'wb') as f:
                    pickle.dump(loops_df, f)
            
            sel_df = self.select_ccd_loops(
                loops_df,
                run.id.chromosome,
                int(run.args['region_start']),
                int(run.args['region_end'])
            )            
            sel_df.to_csv(loops_output_file, index=False, header=False, sep='\t')

        except (OSError, UnicodeError) as exc:
            remove_path(loops_output_file)
            raise PipelineError(
                f'Could not create LoopSage input file "{loops_output_file}": {exc}'
            ) from exc


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


class StageRunLoopSage(PipelineStage):
    name = 'loopsage'

    def __init__(self, number):
        super().__init__(number)

    def build_loopsage_command(self, run: RunInfo) -> str:
        command = [self.args.loopsage_executable]
        command += [f'--bedpe_path', run.input_loops_path]
        command += [f'--out_path', run.repldir]
        for key, value in run.args.items():
            command += [f'--{key}', value]
        return command

    def action(self, run: RunInfo) -> None:
        if not os.path.isfile(run.input_loops_path):
            raise PipelineError(f'LoopSage input is missing. Expected: "{run.input_loops_path}"')

        command = self.build_loopsage_command(run)
        printable_command = subprocess.list2cmdline(command)

        print_message(run.id, f"Executing command: {printable_command}")

        if self.args.dry_run:
            with open(run.loopsage_log, "w", encoding="utf-8") as log_handle:
                log_handle.write("DRY RUN\n")
                log_handle.write(printable_command + "\n")
            return

        try:
            with open(run.loopsage_log, "w", encoding="utf-8") as log_handle:
                log_handle.write(f"Command: {printable_command}\n\n")
                log_handle.flush()

                completed_process = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

        except FileNotFoundError as exc:
            raise PipelineError(
                f"LoopSage ({self.args.loopsage_executable}) error: file not found:"
                f"{exc}"
            ) from exc

        except OSError as exc:
            raise PipelineError(
                f"Could not execute LoopSage: {exc}"
            ) from exc

        if completed_process.returncode != 0:
            raise PipelineError(
                "LoopSage exited with a non-zero status.\n"
                f"  Return code: {completed_process.returncode}\n"
                f"  Log file: {run.loopsage_log}"
            )


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

# def sort_cohesins(X: np.ndarray) -> np.ndarray:
#     return X[np.lexsort(X[:, ::-1].T)]


class StagePreprocessOutputs(PipelineStage):
    name = 'preprocess_outputs'

    def __init__(self, number):
        super().__init__(number)

    @staticmethod
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

    def read_frames(self, ls_dir: str):
        ls_dir = Path(ls_dir)
        Ms = np.load(ls_dir / 'other' / 'Ms.npy')
        Ns = np.load(ls_dir / 'other' / 'Ns.npy')
        n_cohesins = Ms.shape[0]
        n_frames = Ms.shape[1]
        frames = [None for _ in range(n_frames)]
        for structure_file in os.listdir(ls_dir / 'ensemble'):
            m = re.match(r'MDLE_(?P<frame_idx>\d+)\.cif$', structure_file)
            if not m:
                continue
            i = int(m.group('frame_idx')) - 1
            try:
                coords = self.read_coords_from_cif(ls_dir / 'ensemble' / structure_file)
            except Exception as e:
                print(f'Error reading structure file "{structure_file}": {e}')
                continue
            assert frames[i] is None, f"Duplicate frame index {i - 1} in {ls_dir}"
            cohesins = np.stack([
                Ms[:, i], Ns[:, i]
            ]).T
            frames[i] = coords, cohesins
        return frames

    def action(self, run: RunInfo) -> None:
        frames = self.read_frames(run.repldir)
        with open(run.frames_pickle_path, 'wb') as f:
            obj = {'index': run.id.index, 'frames': frames}
            pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------

@numba.njit
def get_big_loops(cohesins: np.ndarray, min_size: int = 5) -> np.ndarray:
    """
    Extract loops from cohesin indices that are at least as large as a specified size.
    """
    is_big_loops = cohesins[:, 1] - cohesins[:, 0] >= min_size    
    return cohesins[is_big_loops]


@numba.njit
def get_nonoverlapping_loops(bead_indices: np.ndarray):
    iord = np.lexsort(bead_indices[:, ::-1].T)
    bi = bead_indices[iord]
    n = bi.shape[0]
    k = 0    
    for i in range(n):
        j = i + 1
        while j < n and bi[j, 0] <= bi[i, 1]:
            j += 1
        k += n - j
    non_overlapping_indices = np.empty((k, 2), dtype=np.int64)
    for i in range(n):
        j = i + 1
        while j < n and bi[j, 0] <= bi[i, 1]:
            j += 1
        non_overlapping_indices[k - (n - j):k, 0] = iord[i]
        non_overlapping_indices[k - (n - j):k, 1] = iord[j:n]
        k -= n - j
    return non_overlapping_indices


@numba.njit
def get_bounding_box(coords: np.ndarray, start: int, end: int):
    """
    Get the bounding box of a curve defined by start and end indices.
    """
    curve = coords[start:end + 1]
    min_coords = np.min(curve, axis=0)
    max_coords = np.max(curve, axis=0)
    return min_coords, max_coords


@numba.njit
def is_bounding_box_overlapping(min0, max0, min1, max1):
    """
    Check if two bounding boxes overlap.
    """
    return np.all(max0 >= min1) and np.all(max1 >= min0)


@numba.jit
def filter_by_bbox(loop_pairs_idx: np.ndarray, cohesins: np.ndarray, coords: np.ndarray):
    bboxes = {}
    for i in range(loop_pairs_idx.shape[0]):
        idx0 = loop_pairs_idx[i][0]
        if idx0 not in bboxes:
            bboxes[idx0] = get_bounding_box(coords, *cohesins[loop_pairs_idx[i][0]])
        idx1 = loop_pairs_idx[i][1]
        if idx1 not in bboxes:
            bboxes[idx1] = get_bounding_box(coords, *cohesins[loop_pairs_idx[i][1]])

    is_candidate = np.empty(loop_pairs_idx.shape[0], dtype=np.bool_)
    for i in range(loop_pairs_idx.shape[0]):
        idx0, idx1 = loop_pairs_idx[i]
        is_candidate[i] = is_bounding_box_overlapping(*bboxes[idx0], *bboxes[idx1])

    return loop_pairs_idx[is_candidate]
    

@numba.njit
def make_topoly_curve(coords, start, end, closed=True):
    n = end - start + 1
    m = n + 1 if closed else n
    curve_coords = np.empty((m, 3), dtype=coords.dtype)
    curve_coords[:n, :] = coords[start:end + 1, :]
    if closed:
        curve_coords[-1, :] = curve_coords[0, :]
    return curve_coords


def topoly_fmt(curve):
    return curve.tolist()


LINKING_NUMBER_COLUMNS = ['idx_loop_A', 'idx_loop_B', 'idx_start_A', 'idx_end_A', 'idx_start_B', 'idx_end_B', 'linking_number']


def calculate_linking_number(coords: np.ndarray, cohesins: np.ndarray, loop_pairs: np.ndarray, threshold=0.1):
    lns = []
    for i, j in loop_pairs:
        s0, e0 = cohesins[i, 0], cohesins[i, 1]
        s1, e1 = cohesins[j, 0], cohesins[j, 1]
        curve_a = make_topoly_curve(coords, s0, e0)
        curve_b = make_topoly_curve(coords, s1, e1)
        ln = topoly.gln(topoly_fmt(curve_a), topoly_fmt(curve_b))
        if abs(ln) < threshold:
            continue
        lns.append((i, j, s0, e0, s1, e1, ln))
    return pd.DataFrame(lns, columns=LINKING_NUMBER_COLUMNS)    


def get_linking_numbers_from_frame(coords: np.ndarray, cohesins: np.ndarray, min_size: int, threshold: float):
    """
    Calculate linking numbers for a given frame of coordinates and cohesins.
    """
    big_loops = get_big_loops(cohesins, min_size=min_size)
    if big_loops.size == 0:
        return None
    non_overlapping = get_nonoverlapping_loops(big_loops)
    if non_overlapping.size == 0:
        return None
    filtered = filter_by_bbox(non_overlapping, cohesins, coords)
    if filtered.size == 0:
        return None
    linking_numbers = calculate_linking_number(coords, big_loops, filtered, threshold=threshold)
    return linking_numbers


_FRAMES_GLOBAL_DATA = {}  # Global cache for frames data to avoid reloading in each worker process


def process_frame(run_idx, iframe, min_size, threshold):
    """
    Process a single frame to extract linking numbers.
    """    
    try:
        idx_tuple = tuple(run_idx.values())
        frame = _FRAMES_GLOBAL_DATA[idx_tuple][iframe]
        if frame is None:
            return None
        coords, cohesins = frame
        return get_linking_numbers_from_frame(coords, cohesins, min_size=min_size, threshold=threshold)
    except Exception as e:
        return e


class StageCalculateLinkingNumbers(PipelineStage):
    name = 'calc_linking_numbers'

    def __init__(self, number):
        super().__init__(number)

    def action(self, run: RunInfo) -> None:                    
        with open(run.frames_pickle_path, 'rb') as f:
            obj = pickle.load(f)
            idx_tuple = tuple(run.id.index.values())
            _FRAMES_GLOBAL_DATA[idx_tuple] = obj['frames']  # Store frames in global cache
            n_frames = len(obj['frames'])

        dummy_record = pd.DataFrame(
            [(-1, -1, -1, -1, -1, -1, np.nan)],
            columns=LINKING_NUMBER_COLUMNS
        )
        results = {-1: dummy_record}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.args.linking_workers
        ) as executor:
            future_to_iframe = {
                executor.submit(
                    process_frame,
                    run.id.index,
                    iframe,
                    self.args.topoly_min_size,
                    self.args.topoly_threshold
                ): iframe
                for iframe in range(self.args.topoly_start_frame, n_frames, self.args.topoly_frame_interval)
            }            
            for future in concurrent.futures.as_completed(future_to_iframe):
                iframe = future_to_iframe[future]
                try:
                    res = future.result()
                    if res is None:
                        continue
                    if isinstance(res, Exception):
                        raise res                    
                    results[iframe] = res
                except Exception as exc:
                    # Cancel tasks that have not yet started.
                    for other_future in future_to_iframe:
                        other_future.cancel()

                    raise PipelineError(
                        f"A parallel linking-number task failed for {run.id}"
                        f"  Error: {exc}"
                    ) from exc                

            res_df = pd.concat(results, names=['frame_idx']).sort_index()
            res_df.reset_index().to_csv(run.linking_numbers_path, index=False)


STAGES = {
    i: stage_cls(i)
    for i, stage_cls in enumerate([
        StagePrepareData,
        StageRunLoopSage,
        StagePreprocessOutputs,
        StageCalculateLinkingNumbers,
    ], start=1)
}


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LoopSage and linking-number pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run_group = parser.add_argument_group("run identification")
    run_group.add_argument("--dataset", required=True)
    run_group.add_argument("--variant", default=0, type=int)
    run_group.add_argument("--rep", required=True, type=int)
    run_group.add_argument("--ccd_id", required=True, type=int)

    general_group = parser.add_argument_group("general pipeline options")
    general_group.add_argument(
        "--output_root",
        required=True,
        help="Root directory under which the run directory is created.",
    )
    general_group.add_argument(
        "--start_from",
        type=int,
        choices=sorted(STAGES.keys()),
        default=min(STAGES.keys()),
        metavar="{1,2,3,4}",
        help="Start from this stage.",
    )
    general_group.add_argument(
        "--stop_after",
        type=int,
        choices=sorted(STAGES.keys()),
        default=max(STAGES.keys()),
        metavar="{1,2,3,4}",
        help="Stop after this stage.",
    )
    general_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Display external commands without executing them.",
    )
    general_group.add_argument(
        "--tempdir",
        default='/tmp/loopsage_pipeline_temp',
        help="Directory for temporary files.",
    )
    for STAGE_NUMBER, STAGE in STAGES.items():
        general_group.add_argument(
            f"--force_{STAGE.number}",
            action="store_true",
            help=f"Overwrite outputs of stage {STAGE_NUMBER} ({STAGE.name}).",
        )

    ########## Stage 1: Data Preparation ##########
    preparation_group = parser.add_argument_group(
        "stage 1: data preparation"
    )
    preparation_group.add_argument(
        "--source_data",
        required=True,
        help="Source file used to construct the LoopSage input.",
    )

    ########## Stage 2: LoopSage Simulation ##########
    simulation_group = parser.add_argument_group(
        "stage 2: LoopSage simulation"
    )
    simulation_group.add_argument(
        "--loopsage_executable",
        # default="./loopsage_wrapper.sh",
        # default="loopsage",
        default="/home/michade/mambaforge/envs/xloopsage/bin/loopsage",
        help="Path to the LoopSage executable."
    )

    ########## Stage 3: Output Preprocessing ##########
    preprocessing_group = parser.add_argument_group(
        "stage 3: output preprocessing"
    )

    ########## Stage 4: Linking-Number Calculation ##########
    linking_group = parser.add_argument_group(
        "stage 4: linking-number calculation"
    )
    linking_group.add_argument(
        "--linking_workers",
        type=int,
        default=1,
        help="Number of worker processes."
    )
    linking_group.add_argument(
        "--topoly_min_size",
        type=int,
        default=5
    )
    linking_group.add_argument(
        "--topoly_threshold",
        type=float,
        default=0.5
    )
    linking_group.add_argument(
        "--topoly_start_frame",
        type=int,
        default=0
    )
    linking_group.add_argument(
        "--topoly_frame_interval",
        type=int,
        default=2
    )

    return parser


def run_pipeline(args: argparse.Namespace, loopsage_args: dict) -> None:
    ensure_positive_workers(args.linking_workers)

    run_id = RunID(
        dataset=args.dataset,
        chromosome=loopsage_args.get('chrom'),
        ccd_id=int(args.ccd_id),
        variant=args.variant,
        evp=loopsage_args.get('ev_p'),
        rep=args.rep,
    )
    run = RunInfo(
        root=Path(args.output_root),
        args=loopsage_args,
        id=run_id
    )

    print_message(run_id, f"Pipeline directory: {run.repldir}")    
    print_message(run_id, f"Pipeline will start from stage {args.start_from}.")
    print_message(run_id, f"Pipeline will stop after stage {args.stop_after}.")

    for stage_number, stage in STAGES.items():
        if stage_number < args.start_from:
            print_message(run_id, f"Skipping stage {stage_number} ({stage.name}).")
            continue
        stage.execute(
            args=args,
            run=run,
            force=getattr(args, f"force_{stage.number}", False)
        )

        if args.stop_after == stage_number:
            print_message(run_id, f"Requested stop after stage {stage_number} ({stage.name}).")
            break
        
    print_message(run_id, "Pipeline completed successfully.")


def get_loopsage_args(tokens: list[str]) -> dict:
    loopsage_args = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        val = tokens[i+1]
        loopsage_args[key.lstrip('-')] = val
        i += 2
    return loopsage_args


def main() -> int:
    parser = create_argument_parser()
    args, tokens = parser.parse_known_args()
    loopsage_args = get_loopsage_args(tokens)

    try:
        os.makedirs(args.tempdir, exist_ok=True)
        run_pipeline(args, loopsage_args)
        return 0

    except KeyboardInterrupt:
        print(
            "\nPipeline interrupted by the user.",
            file=sys.stderr,
            flush=True,
        )
        return 130

    except PipelineError as exc:
        print(f"\nPIPELINE ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    except Exception as exc:
        print(
            f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    # Required for ProcessPoolExecutor, particularly on Windows.
    sys.exit(main())
