import os
import re

import pandas as pd


GENOMIC_COORD_REXP = r'(?P<chromosome>chr(?:\d+|X|Y)):(?P<start>\d+)-(?P<end>\d+)'


def _expand_coords(coords_col):
    df = coords_col.str.extract(GENOMIC_COORD_REXP, expand=True)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df


def from_txt(path):
    raw_df = pd.read_table(path, sep='\t')
    df = pd.DataFrame({
        'GEM_ID_long': raw_df['GEM_ID'],
        'GEM_span': raw_df['GEM_span'].astype(int),
        'GEM_coordinate': raw_df['GEM_coordinate'],
        'n_fragments': raw_df['Fragment_number'].astype(int),
        'List_of_fragment_coordinates': raw_df['List_of_fragment_coordinates'].str.split(';')
    })
    df = pd.concat([
        df.drop(columns='GEM_coordinate'),
        _expand_coords(df['GEM_coordinate']).add_prefix('gem_')
    ], axis=1)
    df = df.sort_values(['gem_start', 'gem_end']).reset_index(drop=True)
    df = df.explode('List_of_fragment_coordinates')
    df.index.names = ['GEM_idx']
    df = df.reset_index()
    df = pd.concat([
        df.drop(columns='List_of_fragment_coordinates'),
        _expand_coords(df['List_of_fragment_coordinates']),
        df['List_of_fragment_coordinates'].str.extract(fr'\((?P<fragment_no>\d+)\)', expand=False).astype(int)
    ], axis=1)
    df = df.reset_index(drop=True).set_index(['GEM_idx', 'fragment_no']).sort_index()
    return df


def from_bed(path):
    df = pd.read_table(path, sep='\t')
    df.columns = ['chromosome', 'start', 'end', 'unknown', 'GEM_ID']
    df = df.drop(columns='unknown')

    def _calc_gem(gdf):
        res_df = pd.DataFrame({
            'n_fragments': len(gdf),
            'gem_start': gdf.start.min(),
            'gem_end': gdf.end.max()
        }, index=[0])
        res_df.index.names = ['tmp']
        return res_df

    by_gem = df.groupby('GEM_ID', sort=False).apply(_calc_gem).reset_index('tmp', drop=True)
    df = pd.merge(df, by_gem, on=['GEM_ID'])
    df = df.sort_values(by=['gem_start', 'start']).reset_index(drop=True)
    df['fragment_no'] = df.groupby('GEM_ID', sort=False).cumcount() + 1
    df['GEM_span'] = df.gem_end - df.gem_start
    # print(df.groupby('GEM_ID', sort=False).ngroup())
    df['GEM_idx'] = df.groupby('GEM_ID', sort=False).ngroup()
    df = df.rename(columns={'chromosome': 'gem_chromosome'})
    df.index.names = ['idx_frag']
    assert df.gem_start.is_monotonic_increasing
    return df
