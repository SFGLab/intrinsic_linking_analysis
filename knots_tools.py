from __future__ import annotations

import itertools
import re
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Callable, Type, Set, Iterable, Union

import numpy as np
import pandas as pd
import igraph as ig

import polychrom.polymer_analyses as pol


HumanChromosomeDtype = pd.api.types.CategoricalDtype(
    # 22 autosomes + sex chromosomes + mitochondrial ("chrM")
    ['chr%d' % i for i in range(1, 22 + 1)] + ['chrX', 'chrY', 'chrM'],
    ordered=True
)

DatasetDtype = pd.api.types.CategoricalDtype(['GM12878lr', 'GM12878', 'H1ESC', 'HFFC6', 'WTC11'], ordered=True)


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
                    minor.segment_coords.append((start_pos, end_pos))
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
    def segment_coords(self):
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


def get_loop(X, s, e):
    assert s < e
    k = e - s + 1  # no.beads without closing segment
    curve = np.empty((k + 1, 3), dtype=X.dtype)
    curve[:k] = X[s:e + 1]
    curve[k] = X[s]  # add closing segment
    return curve

def get_linking_number(curve1, curve2):
    return pol.getLinkingNumber(
        curve1, curve2,
        simplify=True,  # Otherwise won't give correct resutls
        randomOffset=False,
        verbose=False
    )
    
def get_loop_pair_infos(X, R):
    nl = R.shape[0]
    df = pd.DataFrame.from_records([
        {
            'loop1': i, 'loop2': j,
            'start1': R[i, 0], 'end1': R[i, 1],
            'start2': R[j, 0], 'end2': R[j, 1],
            'linking_number': get_linking_number(
                get_loop(X, R[i, 0], R[i, 1]),
                get_loop(X, R[j, 0], R[j, 1])                
            ),
            'len1': R[i, 1] - R[i, 0],
            'len2': R[j, 1] - R[j, 0],
            'overlap': max(0, min(R[i, 1], R[j, 1]) - max(R[i, 0], R[j, 0]))
        }
        for i in range(nl)
        for j in range(i + 1, nl)
    ])
    df['abs_linking_number'] = df['linking_number'].abs()
    return df.sort_values(['overlap', 'abs_linking_number'], ascending=[True, False])
