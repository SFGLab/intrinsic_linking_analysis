from __future__ import annotations

import itertools
import re
import os
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Callable, Type, Set, Iterable, Union

import pyranges1 as pr
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats


def compare_boxplot(df, ax, y, ylab, legend=False, test=True):
    ax = sns.boxplot(df, x='has_minors', hue='has_minors', y=y, palette='colorblind', ax=ax)
    ax.set_ylabel(ylab)
    ax.set_xlabel('')
    ax.set_xticklabels(['non-linked', 'linked'])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    if legend:        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["non-linked", "linked"], title="CCD type");
    else:
        ax.get_legend().remove()
    if test:
        non_linked = df.query('has_minors == False')[y]
        linked = df.query('has_minors == True')[y]
        stat, pval = scipy.stats.mannwhitneyu(non_linked, linked, alternative='two-sided')
        med_non_linked = np.median(non_linked)
        med_linked = np.median(linked)
        print(f"Mann-Whitney U test: statistic={stat:.4f}, p-value={pval:.6f} (data: {y}). Median: non-linked={med_non_linked:.4f}, linked={med_linked:.4f}")
    return ax

def compare_barplot(df, ax, y, ylab, legend=False, test=True):
    ax = sns.barplot(
        data=df,
        x='has_minors',
        hue='has_minors',
        y=y,
        palette='colorblind',
        ax=ax,
        # ci=None,
        # dodge=False,
    )
    ax.set_ylabel(ylab)
    ax.set_xlabel('')
    ax.set_xticklabels(['non-linked', 'linked'])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["non-linked", "linked"], title="CCD type")
    else:
        ax.get_legend().remove()
    if test:
        non_linked = df.query('has_minors == False')[y]
        linked = df.query('has_minors == True')[y]
        stat, pval = scipy.stats.mannwhitneyu(non_linked, linked, alternative='two-sided')
        med_non_linked = np.median(non_linked)
        med_linked = np.median(linked)
        print(f"Mann-Whitney U test: statistic={stat:.4f}, p-value={pval:.6f} (data: {y}). Median: non-linked={med_non_linked:.4f}, linked={med_linked:.4f}")
    return ax


RENAME_TO_PYRANGES = {s: s.capitalize() for s in ['chromosome', 'start', 'end']}


def load_cknots_base_data(base_results_dir='./data/', as_pyranges=True):
    ccds = pd.read_csv(
        os.path.join(base_results_dir, 'all_ccds.csv'),
        index_col=CCD_INDEX_NAMES,
        dtype=CCD_INDEX_DTYPES
    )
    ccds['length'] = ccds['end'] - ccds['start']
    minors = pd.read_csv(
        os.path.join(base_results_dir, 'all_minors.csv'),
        index_col=CCD_INDEX_NAMES + ['minor_id'],
        dtype=CCD_INDEX_DTYPES
    )
    if as_pyranges:
        ccds_pr = pr.PyRanges(ccds.reset_index().rename(columns=RENAME_TO_PYRANGES))
        minors_pr = pr.PyRanges(minors.reset_index().rename(columns=RENAME_TO_PYRANGES))
        return ccds_pr, minors_pr
    else:
        return ccds, minors


HumanChromosomeDtype = pd.api.types.CategoricalDtype(
    # 22 autosomes + sex chromosomes + mitochondrial ("chrM")
    ['chr%d' % i for i in range(1, 22 + 1)] + ['chrX', 'chrY', 'chrM'],
    ordered=True
)

GenomeDtype = pd.api.types.CategoricalDtype(['hg19', 'hg38'], ordered=True)
ProteinDtype = pd.api.types.CategoricalDtype(['CTCF'], ordered=True)
CellLinesDtype = pd.api.types.CategoricalDtype(['GM12878', 'H1ESC', 'HFFC6', 'WTC11'], ordered=True)
DatasetDtype = pd.api.types.CategoricalDtype(['GM12878lr', 'GM12878', 'H1ESC', 'HFFC6', 'WTC11'], ordered=True)

FULL_CCD_INDEX_DTYPES = {
    'dataset': DatasetDtype,
    'protein': ProteinDtype,    
    'min_petcount': 'int16',
    'chromosome': HumanChromosomeDtype,
    'ccd_id': 'int16'
}
FULL_CCD_INDEX_NAMES = list(FULL_CCD_INDEX_DTYPES.keys())


CCD_INDEX_DTYPES = {
    'dataset': DatasetDtype,
    'chromosome': HumanChromosomeDtype,
    'ccd_id': 'int16'
}
CCD_INDEX_NAMES = list(CCD_INDEX_DTYPES.keys())


def parse_chromosome(input):
    if input.startswith('chr'):
        if input[3] == '_':
            code = input[4:]
        else:
            code = input[3:]
    else:
        code = input
    if code not in ('X', 'Y', 'M'):
        code = int(code)
        if code == 23:
            code = 'X'
    return f'chr{code}'


def parse_node_name(node_str: str) -> Tuple[str, int]:
    chromosome, coord_str = node_str.split('_')
    coord = int(coord_str)
    return chromosome, coord


class LinearMinor:
    __slots__ = ['_idx', '_segments', '_segment_coords', '_edges', '_chromosome', '_coordinates']
    def __init__(self, idx):
        self._idx = idx
        self._segments = []
        self._segment_coords = []
        self._edges = {}
        self._chromosome = None
        self._coordinates = None

    def __eq__(self, other):
        return self._chromosome == other._chromosome and self._idx == other._idx

    def __hash__(self):
        return (self._chromosome, self._idx)

    def __repr__(self):
        return f'LM({self._chromosome}-{self._idx:03d})'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def read(minor_file, node_name_to_id = None):
        minors = []
        chromosome = None
        min_coord = 10 ** 16
        max_coord = -1
        with open(minor_file) as f:
            for il, line in enumerate(f.readlines()):
                line = line.strip()
                if line.startswith('MINOR'):
                    if len(minors) > 0:
                        minors[-1]._chromosome = chromosome
                        minors[-1]._coordinates = (min_coord, max_coord)
                    minor = LinearMinor(len(minors))
                    minors.append(minor)
                elif line.startswith('segment'):
                    m = re.search(r'start=\((\d+)=(\w+)\) end=\((\d+)=(\w+)\)', line)
                    assert m is not None, f'Malformed line {il} in "{minor_file}": {line}'
                    start_node_idx = int(m.group(1))
                    start_node_name = m.group(2)
                    end_node_idx = int(m.group(3))
                    end_node_name = m.group(4)
                    start_chromosome, start_pos = parse_node_name(start_node_name)
                    end_chromosome, end_pos = parse_node_name(end_node_name)
                    if chromosome is None:
                        chromosome = start_chromosome
                    assert chromosome == start_chromosome
                    assert chromosome == end_chromosome
                    if node_name_to_id is not None:
                        assert node_name_to_id[start_node_name] == start_node_idx
                        assert node_name_to_id[end_node_name] == end_node_idx
                    assert start_chromosome == end_chromosome
                    min_coord = min(min_coord, start_pos)
                    max_coord = max(max_coord, end_pos)
                    minor.segments.append(range(start_node_idx, end_node_idx + 1))
                    minor.segement_coords.append((start_pos, end_pos))
                elif line.startswith('from'):
                    m = re.search(r'from (\d+) to (\d+), eid=(\d+), left=\((\d+)=(\w+)\), right=\((\d+)=(\w+)\)', line)
                    assert m is not None, f'Malformed line {il} in "{minor_file}": {line}'
                    from_ = int(m.group(1))
                    to_ = int(m.group(2))
                    left_node_idx = int(m.group(4))
                    left_node_name = m.group(5)
                    right_node_idx = int(m.group(6))
                    right_node_name = m.group(7)
                    if node_name_to_id is not None:
                        assert node_name_to_id[left_node_name] == left_node_idx
                        assert node_name_to_id[right_node_name] == right_node_idx
                    minor._edges[(from_, to_)] = (left_node_idx, right_node_idx)
            if len(minors) > 0:
                minors[-1]._chromosome = chromosome
                minors[-1]._coordinates = (min_coord, max_coord)

        return minors

    def nodes(self):
        return itertools.chain.from_iterable(self.segments)

    def __getitem__(self, key: Tuple[int, int]) -> str:
        return self._edges[key]

    def graph_edges(self):
        return self._edges.values()

    @property
    def edges(self):
        return self._edges

    @property
    def segments(self):
        return self._segments
    
    @property
    def segement_coords(self):
        return self._segment_coords

    @property
    def idx(self):
        return self._idx

    @property
    def chromosome(self):
        return self._chromosome

    @property
    def coordinates(self):
        return self._coordinates    
    
    @property
    def start(self):
        return self._coordinates[0]
    
    @property
    def end(self):
        return self._coordinates[1]

    def add_info_to_graph(self, g: ig.Graph):
        cols = ['minors', 'idx_in_minor']
        for c in cols:
            assert c in g.vs.attribute_names()
            assert c in g.es.attribute_names()
        for i, seg in enumerate(self.segments):
            for u in seg:
                g.vs[u]['minors'].append(self)
                g.vs[u]['idx_in_minor'].append(i)
        for (i, j), (u, v) in self.edges.items():
            eid = g.get_eid(u, v)  # will raise an error if the edge does not exist, which is good, as it should
            g.es[eid]['minors'].append(self)
            g.es[eid]['idx_in_minor'].append((i, j))

    @staticmethod
    def add_multiple_info_to_graph(minors: List[LinearMinor], g: ig.Graph):
        cols = ['minors', 'idx_in_minor']
        for c in cols:
            g.vs[c] = [[] for _ in range(g.vcount())]
        for c in cols:
            g.es[c] = [[] for _ in range(g.ecount())]
        for m in minors:
            m.add_info_to_graph(g)
            

def read_graph_from_cknots_file(fn):
    node_name_to_id = {}
    g = ig.Graph(directed=False)
    chromosome = None    
    with open(fn) as f:
        for row_string in f.readlines():
            row = row_string.split()
            if row[0] == 'NODE':
                name = row[1]
                node_chromosome, coord = parse_node_name(name)
                u = g.add_vertex(coord=coord)
                assert u.index not in node_name_to_id.values(), "Expected unique node ids"
                node_name_to_id[name] = u.index                                
                if chromosome is None:
                    chromosome = node_chromosome
                assert chromosome == node_chromosome, "We support only cis (intrachromosomal) interactions"
                assert u == 0 or coord >= g.vs[u.index - 1]['coord'], "Nodes are expected to be linearly sorted"
            elif row[0] == 'EDGE':
                u = node_name_to_id[row[1]]
                v = node_name_to_id[row[2]]
                if u > v:
                    u, v = v, u
                elif u == v:  # skip self-loops
                    continue                
                eid = g.get_eid(u, v, error=False)
                petcount = int(row[3])
                distance = g.vs[v]['coord'] - g.vs[u]['coord']          
                loop_id = int(row[4])
                if eid == -1:   # new edge
                    g.add_edge(
                        u, v,
                        petcount=petcount,
                        distance=distance,
                        is_contact=True,
                        is_strand=False,
                        loop_count=1,
                        loop_ids=[loop_id]
                    )
                else:  # ad-hoc merge multi-loop edges
                    g.es[eid]['petcount'] += int(row[3])
                    g.es[eid]['loop_count'] += 1
                    g.es[eid]['loop_ids'].append(loop_id)                    
            else:
                raise ValueError(f"Malformed row: {row_string}")
    # add strand edges:
    for u in range(len(node_name_to_id) - 1):
        v = u + 1
        eid = g.get_eid(u, v, error=False)
        if eid == -1:  # new edge
            g.add_edge(
                u, v,
                petcount=0,
                distance=(g.vs[v]['coord'] - g.vs[u]['coord']),
                is_contact=False,
                is_strand=True,
                loop_count=0,
                loop_ids=[]
            )
        else:  # there already is a contact edge
            g.es[eid]['is_strand'] = True
    return g, node_name_to_id


# https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not/4404056#4404056
def is_sorted_ascending(lst):
    for i, element in enumerate(lst[1:]):
        if element < lst[i - 1]:
            return False
    return True


def pd_apply_long(df: pd.DataFrame, fun, sort=True, **xtra_kwargs):
    dfs = [fun(row, **xtra_kwargs) for _, row in df.iterrows()]
    res_df = pd.concat(dfs, keys=df.index, sort=sort)
    return res_df

def read_narrowpeak_bigbed(bigbed_path):
    """
    Read an ENCODE narrowPeak-style bigBed file into a pandas DataFrame.

    Expected logical columns:
        chrom, start, end, name, score, strand,
        signalValue, pValue, qValue, peak

    The `peak` column is the summit offset relative to peak start.
    """

    bb = pyBigWig.open(bigbed_path)

    if not bb.isBigBed():
        bb.close()
        raise ValueError(f"Not a bigBed file: {bigbed_path}")

    rows = []

    try:
        chromosomes = bb.chroms()

        for chrom, chrom_length in chromosomes.items():
            entries = bb.entries(chrom, 0, chrom_length)

            if entries is None:
                continue

            for start, end, extra_fields in entries:
                fields = extra_fields.split("\t")

                if len(fields) < 7:
                    raise ValueError(
                        f"Expected at least 7 extra narrowPeak fields, "
                        f"but found {len(fields)} for "
                        f"{chrom}:{start}-{end}: {fields}"
                    )

                name = fields[0]
                score = fields[1]
                strand = fields[2]
                signal_value = fields[3]
                p_value = fields[4]
                q_value = fields[5]
                peak_offset = fields[6]

                rows.append(
                    {
                        "Chromosome": chrom,
                        "Start": int(start),
                        "End": int(end),
                        "Name": name,
                        "Score": pd.to_numeric(score, errors="coerce"),
                        "Strand": strand,
                        "SignalValue": pd.to_numeric(
                            signal_value, errors="coerce"
                        ),
                        "PValue": pd.to_numeric(
                            p_value, errors="coerce"
                        ),
                        "QValue": pd.to_numeric(
                            q_value, errors="coerce"
                        ),
                        "PeakOffset": pd.to_numeric(
                            peak_offset, errors="coerce"
                        ),
                    }
                )
    finally:
        bb.close()

    peaks = pd.DataFrame(rows)

    if peaks.empty:
        return peaks

    peaks["PeakID"] = np.arange(len(peaks), dtype=int)

    # Convert narrowPeak summit offset to a genomic coordinate.
    peaks["Summit"] = (
        peaks["Start"] + peaks["PeakOffset"]
    ).astype("Int64")

    # Some files may use -1 when a summit is unavailable.
    invalid_summit = (
        peaks["PeakOffset"].isna()
        | (peaks["PeakOffset"] < 0)
        | (peaks["Summit"] < peaks["Start"])
        | (peaks["Summit"] >= peaks["End"])
    )

    # Fall back to the peak midpoint when summit information is missing.
    peaks.loc[invalid_summit, "Summit"] = (
        (
            peaks.loc[invalid_summit, "Start"]
            + peaks.loc[invalid_summit, "End"]
        )
        // 2
    ).astype("Int64")

    peaks["Summit"] = peaks["Summit"].astype(int)

    return peaks

def prepare_ccds(ccds):
    """
    Validate and prepare a PyRanges object containing CCD intervals.

    The input must have:
        Chromosome, Start, End

    If CCD_ID is absent, one is generated.
    """

    if not isinstance(ccds, pr.PyRanges):
        raise TypeError("ccds must be a PyRanges object")

    ccd_df = ccds.copy()

    required = {"Chromosome", "Start", "End"}
    missing = required - set(ccd_df.columns)

    if missing:
        raise ValueError(
            f"CCDs are missing required columns: {sorted(missing)}"
        )

    if "CCD_ID" not in ccd_df.columns:
        ccd_df["CCD_ID"] = [
            f"CCD_{i:06d}" for i in range(len(ccd_df))
        ]

    if ccd_df["CCD_ID"].duplicated().any():
        duplicated = ccd_df.loc[
            ccd_df["CCD_ID"].duplicated(), "CCD_ID"
        ].tolist()

        raise ValueError(
            "CCD_ID values must be unique. Duplicates include: "
            + ", ".join(map(str, duplicated[:10]))
        )

    ccd_df["Start"] = ccd_df["Start"].astype(int)
    ccd_df["End"] = ccd_df["End"].astype(int)
    ccd_df["CCDLength"] = ccd_df["End"] - ccd_df["Start"]

    if (ccd_df["CCDLength"] <= 0).any():
        raise ValueError("All CCDs must have End > Start")

    return ccd_df

def peaks_to_summit_pyranges(peaks):
    """
    Convert peaks to one-base summit intervals.

    Using summits avoids assigning one broad peak to two adjacent CCDs
    merely because the peak interval crosses their boundary.
    """

    summit_df = peaks[
        [
            "Chromosome",
            "Summit",
            "PeakID",
            "SignalValue",
            "Score",
            "PValue",
            "QValue",
        ]
    ].copy()

    summit_df["Start"] = summit_df["Summit"]
    summit_df["End"] = summit_df["Summit"] + 1

    summit_df = summit_df[
        [
            "Chromosome",
            "Start",
            "End",
            "PeakID",
            "Summit",
            "SignalValue",
            "Score",
            "PValue",
            "QValue",
        ]
    ]

    return pr.PyRanges(summit_df)

def make_boundary_windows(ccd_df, chrom_sizes, boundary_radius=50_000):
    """
    Create left- and right-boundary windows.

    Each boundary window is:
        boundary - boundary_radius
        boundary + boundary_radius

    Windows are truncated at chromosome boundaries.
    """

    rows = []

    for row in ccd_df.itertuples(index=False):
        chrom = row.Chromosome

        if chrom not in chrom_sizes:
            continue

        chrom_length = chrom_sizes[chrom]

        boundaries = [
            ("left", int(row.Start)),
            ("right", int(row.End)),
        ]

        for side, boundary_position in boundaries:
            start = max(0, boundary_position - boundary_radius)
            end = min(
                chrom_length,
                boundary_position + boundary_radius,
            )

            if end <= start:
                continue

            rows.append(
                {
                    "Chromosome": chrom,
                    "Start": start,
                    "End": end,
                    "CCD_ID": row.CCD_ID,
                    "BoundarySide": side,
                    "BoundaryPosition": boundary_position,
                    "BoundaryWindowLength": end - start,
                }
            )

    return pd.DataFrame(rows)

def bigwig_mean(bw, chrom, start, end):
    """
    Return the exact mean bigWig signal over an interval.

    Missing signal is treated as zero. This is appropriate for an ENCODE
    fold-change signal track when uncovered bases should contribute no
    enrichment.
    """

    if chrom not in bw.chroms():
        return np.nan

    chrom_length = bw.chroms(chrom)

    start = max(0, int(start))
    end = min(int(end), chrom_length)

    if end <= start:
        return np.nan

    result = bw.stats(
        chrom,
        start,
        end,
        type="mean",
        exact=True,
    )[0]

    # pyBigWig returns None when the interval contains no signal entries.
    if result is None:
        return 0.0

    return float(result)


def merge_intervals(intervals):
    """
    Merge overlapping or directly adjacent intervals.
    """

    if not intervals:
        return []

    intervals = sorted(
        (int(start), int(end))
        for start, end in intervals
        if end > start
    )

    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [tuple(interval) for interval in merged]


def bigwig_mean_over_intervals(bw, chrom, intervals):
    """
    Calculate a length-weighted mean over the union of several intervals.

    Overlapping intervals are merged first, preventing double-counting.
    Missing bigWig bases are treated as zero.
    """

    merged = merge_intervals(intervals)

    if not merged:
        return np.nan

    weighted_sum = 0.0
    total_length = 0

    for start, end in merged:
        interval_length = end - start
        mean_signal = bigwig_mean(bw, chrom, start, end)

        if np.isnan(mean_signal):
            continue

        weighted_sum += mean_signal * interval_length
        total_length += interval_length

    if total_length == 0:
        return np.nan

    return weighted_sum / total_length

def calculate_ctcf_metrics(
    ccds,
    peak_bigbed_path,
    signal_bigwig_path,
    boundary_radius=50_000,
):
    """
    Calculate CTCF peak and continuous-signal metrics for each CCD.

    Parameters
    ----------
    ccds
        PyRanges object with Chromosome, Start and End columns.
        CCD_ID is optional.

    peak_bigbed_path
        ENCODE IDR-thresholded narrowPeak bigBed.

    signal_bigwig_path
        ENCODE fold-change-over-control bigWig.

    boundary_radius
        Number of bases on each side of a CCD boundary.
        Default: 50 kb, resulting in a nominal 100 kb window
        around each boundary.

    Returns
    -------
    metrics
        pandas DataFrame with one row per CCD.

    peaks
        pandas DataFrame containing all peaks read from the bigBed.

    overall_peak_assignments
        PyRanges join result assigning peak summits to CCDs.

    boundary_peak_assignments
        PyRanges join result assigning peak summits to boundary windows.
    """

    ccd_df = prepare_ccds(ccds)

    print(f"Reading peaks from {peak_bigbed_path}")
    peaks = read_narrowpeak_bigbed(peak_bigbed_path)
    print(f"Read {len(peaks):,} peaks")

    if peaks.empty:
        raise ValueError("No peaks were read from the bigBed file")

    peak_summits = peaks_to_summit_pyranges(peaks)
    ccd_pr = pr.PyRanges(ccd_df)

    # ------------------------------------------------------------
    # 1. Assign peak summits to complete CCD intervals
    # ------------------------------------------------------------

    print("Assigning peak summits to CCDs")

    overall_peak_assignments = ccd_pr.join_overlaps(
        peak_summits,
        suffix="_peak",
    )

    overall_join_df = overall_peak_assignments.copy()

    # One row per CCD-peak assignment.
    overall_join_df = overall_join_df.drop_duplicates(
        subset=["CCD_ID", "PeakID"]
    )

    overall_agg = (
        overall_join_df
        .groupby("CCD_ID", observed=True)
        .agg(
            ctcf_peak_count=("PeakID", "nunique"),
            ctcf_peak_signal_median=("SignalValue", "median"),
            ctcf_peak_signal_mean=("SignalValue", "mean"),
            ctcf_peak_signal_max=("SignalValue", "max"),
            ctcf_peak_signal_sum=("SignalValue", "sum"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 2. Construct boundary windows
    # ------------------------------------------------------------

    bw = pyBigWig.open(signal_bigwig_path)

    if not bw.isBigWig():
        bw.close()
        raise ValueError(f"Not a bigWig file: {signal_bigwig_path}")

    chrom_sizes = bw.chroms()

    boundary_df = make_boundary_windows(
        ccd_df=ccd_df,
        chrom_sizes=chrom_sizes,
        boundary_radius=boundary_radius,
    )

    boundary_pr = pr.PyRanges(boundary_df)

    # ------------------------------------------------------------
    # 3. Assign peaks to boundary windows
    # ------------------------------------------------------------

    print("Assigning peak summits to boundary windows")

    boundary_peak_assignments = boundary_pr.join_overlaps(
        peak_summits,
        suffix="_peak",
    )

    boundary_join_df = boundary_peak_assignments.copy()

    boundary_join_df = boundary_join_df.drop_duplicates(
        subset=["CCD_ID", "BoundarySide", "PeakID"]
    )

    boundary_side_agg = (
        boundary_join_df
        .groupby(
            ["CCD_ID", "BoundarySide"],
            observed=True,
        )
        .agg(
            boundary_peak_count=("PeakID", "nunique"),
        )
        .reset_index()
    )

    boundary_side_counts = (
        boundary_side_agg
        .pivot(
            index="CCD_ID",
            columns="BoundarySide",
            values="boundary_peak_count",
        )
        .fillna(0)
        .rename(
            columns={
                "left": "ctcf_left_boundary_peak_count",
                "right": "ctcf_right_boundary_peak_count",
            }
        )
        .reset_index()
    )

    for column in [
        "ctcf_left_boundary_peak_count",
        "ctcf_right_boundary_peak_count",
    ]:
        if column not in boundary_side_counts.columns:
            boundary_side_counts[column] = 0

    # Unique peaks near either boundary. This does not count the same peak
    # twice if the two boundary windows overlap.
    boundary_unique_agg = (
        boundary_join_df
        .groupby("CCD_ID", observed=True)
        .agg(
            ctcf_boundary_peak_count=("PeakID", "nunique"),
            ctcf_boundary_peak_signal_median=(
                "SignalValue",
                "median",
            ),
            ctcf_boundary_peak_signal_mean=(
                "SignalValue",
                "mean",
            ),
            ctcf_boundary_peak_signal_max=(
                "SignalValue",
                "max",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 4. Merge peak statistics into one row per CCD
    # ------------------------------------------------------------

    metrics = ccd_df.copy()

    metrics = metrics.merge(
        overall_agg,
        on="CCD_ID",
        how="left",
    )

    metrics = metrics.merge(
        boundary_side_counts,
        on="CCD_ID",
        how="left",
    )

    metrics = metrics.merge(
        boundary_unique_agg,
        on="CCD_ID",
        how="left",
    )

    count_columns = [
        "ctcf_peak_count",
        "ctcf_boundary_peak_count",
        "ctcf_left_boundary_peak_count",
        "ctcf_right_boundary_peak_count",
    ]

    for column in count_columns:
        metrics[column] = (
            metrics[column]
            .fillna(0)
            .astype(int)
        )

    # ------------------------------------------------------------
    # 5. Overall occurrence and occupancy
    # ------------------------------------------------------------

    metrics["ctcf_peaks_per_mb"] = (
        metrics["ctcf_peak_count"]
        / metrics["CCDLength"]
        * 1_000_000
    )

    metrics["ctcf_occupied"] = (
        metrics["ctcf_peak_count"] > 0
    )

    # ------------------------------------------------------------
    # 6. Boundary occurrence and occupancy
    # ------------------------------------------------------------

    metrics["ctcf_left_boundary_occupied"] = (
        metrics["ctcf_left_boundary_peak_count"] > 0
    )

    metrics["ctcf_right_boundary_occupied"] = (
        metrics["ctcf_right_boundary_peak_count"] > 0
    )

    metrics["ctcf_any_boundary_occupied"] = (
        metrics["ctcf_left_boundary_occupied"]
        | metrics["ctcf_right_boundary_occupied"]
    )

    metrics["ctcf_both_boundaries_occupied"] = (
        metrics["ctcf_left_boundary_occupied"]
        & metrics["ctcf_right_boundary_occupied"]
    )

    metrics["ctcf_occupied_boundary_count"] = (
        metrics["ctcf_left_boundary_occupied"].astype(int)
        + metrics["ctcf_right_boundary_occupied"].astype(int)
    )

    # Denominator is the union of the two boundary windows.
    boundary_union_lengths = {}

    for ccd_id, group in boundary_df.groupby(
        "CCD_ID",
        observed=True,
    ):
        intervals = list(zip(group["Start"], group["End"]))
        merged = merge_intervals(intervals)

        boundary_union_lengths[ccd_id] = sum(
            end - start for start, end in merged
        )

    metrics["BoundaryUnionLength"] = (
        metrics["CCD_ID"]
        .map(boundary_union_lengths)
    )

    metrics["ctcf_boundary_peaks_per_mb"] = (
        metrics["ctcf_boundary_peak_count"]
        / metrics["BoundaryUnionLength"]
        * 1_000_000
    )

    # ------------------------------------------------------------
    # 7. Continuous fold-change signal
    # ------------------------------------------------------------

    print("Calculating continuous bigWig signal")

    overall_signal = {}
    left_boundary_signal = {}
    right_boundary_signal = {}
    combined_boundary_signal = {}

    boundary_lookup = {
        ccd_id: group.copy()
        for ccd_id, group in boundary_df.groupby(
            "CCD_ID",
            observed=True,
        )
    }

    try:
        for row in metrics.itertuples(index=False):
            ccd_id = row.CCD_ID
            chrom = row.Chromosome

            overall_signal[ccd_id] = bigwig_mean(
                bw,
                chrom,
                row.Start,
                row.End,
            )

            ccd_boundaries = boundary_lookup.get(ccd_id)

            if ccd_boundaries is None:
                left_boundary_signal[ccd_id] = np.nan
                right_boundary_signal[ccd_id] = np.nan
                combined_boundary_signal[ccd_id] = np.nan
                continue

            left = ccd_boundaries[
                ccd_boundaries["BoundarySide"] == "left"
            ]

            right = ccd_boundaries[
                ccd_boundaries["BoundarySide"] == "right"
            ]

            if len(left):
                left_row = left.iloc[0]

                left_boundary_signal[ccd_id] = bigwig_mean(
                    bw,
                    chrom,
                    left_row["Start"],
                    left_row["End"],
                )
            else:
                left_boundary_signal[ccd_id] = np.nan

            if len(right):
                right_row = right.iloc[0]

                right_boundary_signal[ccd_id] = bigwig_mean(
                    bw,
                    chrom,
                    right_row["Start"],
                    right_row["End"],
                )
            else:
                right_boundary_signal[ccd_id] = np.nan

            intervals = list(
                zip(
                    ccd_boundaries["Start"],
                    ccd_boundaries["End"],
                )
            )

            combined_boundary_signal[ccd_id] = (
                bigwig_mean_over_intervals(
                    bw,
                    chrom,
                    intervals,
                )
            )

    finally:
        bw.close()

    metrics["ctcf_fc_mean"] = (
        metrics["CCD_ID"].map(overall_signal)
    )

    metrics["ctcf_fc_left_boundary_mean"] = (
        metrics["CCD_ID"].map(left_boundary_signal)
    )

    metrics["ctcf_fc_right_boundary_mean"] = (
        metrics["CCD_ID"].map(right_boundary_signal)
    )

    metrics["ctcf_fc_boundary_mean"] = (
        metrics["CCD_ID"].map(combined_boundary_signal)
    )

    # Mean of the two boundary-level means, giving each boundary equal
    # weight even if a chromosome-end window was truncated.
    metrics["ctcf_fc_mean_of_boundaries"] = (
        metrics[
            [
                "ctcf_fc_left_boundary_mean",
                "ctcf_fc_right_boundary_mean",
            ]
        ]
        .mean(axis=1)
    )

    # A useful boundary-to-whole-CCD contrast.
    metrics["ctcf_fc_boundary_vs_overall"] = (
        metrics["ctcf_fc_boundary_mean"]
        - metrics["ctcf_fc_mean"]
    )

    # Use a pseudocount only for the ratio.
    metrics["ctcf_fc_boundary_over_overall"] = (
        (metrics["ctcf_fc_boundary_mean"] + 1e-6)
        / (metrics["ctcf_fc_mean"] + 1e-6)
    )

    return (
        metrics,
        peaks,
        overall_peak_assignments,
        boundary_peak_assignments,
    )