import glob
import os
import re
import requests
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .log import logger


def _cols_without(df, s):
    return [c for c in df.columns if s not in c]


def read_tsvs(path, features=tuple()):
    '''
    Reads AIRR-formatted input files into a single DataFrame and populates
    common fields.

    Parameters
    ----------
    path : str
        Path to directory containing ``.pooled.tsv`` files
    features : list, optional
        List of features which are encoded in the file names.

    Returns
    -------
    Single DataFrame containing the concatenated AIRR-formatted data.

    '''
    if features and isinstance(features, str):
        features = [features]
    assert 'subject' not in features

    dfs = []
    for fn in glob.glob(os.path.join(path, '*.pooled.tsv')):
        df = pd.read_csv(fn, sep='\t', dtype={'subject': str})
        if features:
            values = os.path.basename(fn).split('.')[1].split('_AND_')
            for i, feature in enumerate(values):
                df[features[i]] = feature
        df['copies_fraction'] = df.copies / df.copies.sum()
        df['copies_percent'] = 100 * df['copies_fraction']
        df['shm'] = 100 * (1 - df['avg_v_identity'])
        df['clones'] = 1
        df = df.sort_values('copies', ascending=False)

        dfs.append(df)

    return pd.concat(dfs)


def read_metadata(path):
    '''
    Reads a metadata file into a `pd.DataFrame`, prefixing `METADATA_` to each
    field and setting the `replicate_name` to its index.

    Parameters
    ----------
    path : str
        Path to metadata file

    Returns
    -------
    `pd.DataFrame` containing the metadata.

    '''
    metadata = pd.read_csv(path, sep='\t').set_index('replicate_name')
    metadata.columns = [f'METADATA_{c}' for c in metadata.columns]
    return metadata


def read_directory(path):
    '''
    Reads AIRR-formatted TSV files and joins it with an associated
    `metadata.tsv` file to return a unified `pd.DataFrame`.

    Parameters
    ----------
    path : str
        Path to AIRR-formatted files and `metadata.tsv`

    Returns
    -------
    `pd.DataFrame` with AIRR-seq data and metadata.

    '''
    df = read_tsvs(path, ['replicate_name'])
    metadata = read_metadata(os.path.join(path, 'metadata.tsv'))
    df = df.join(metadata, on='replicate_name', rsuffix='__DROP')

    df = df[_cols_without(df, '__DROP')]
    return df


def save_fig_and_data(
    name, df, path='./', ext='pdf', **kwargs
):  # pragma: no cover
    '''
    Saves the most recently generated figure and associated data to files.

    Parameters
    ----------
    name : str
        The filename to use for both the figure and data file.
    df : pd.DataFrame
        The DataFrame used to generate the figure.
    path : str, optional
        Path to directory into which the files should be saved.
    ext : str, optional
        The extension of the figure file.  Defaults to pdf but can be any image
        format such as ``png``.
    kwargs : dict
        Additional parameters which will be passed to ``df.to_csv``

    '''

    path = os.path.join(path, name)
    df.to_csv(f'{path}.tsv', sep='\t', **kwargs)
    plt.savefig(f'{path}.{ext}', bbox_inches='tight')


def _run_job_and_get_result(prefix, route, out_name):  # pragma: no cover
    resp = requests.get(f'{prefix}/{route}')
    uid = resp.json()['uid']
    logger.info(f'Job for "{route}" has UUID {uid}')
    while True:
        resp = requests.get(f'{prefix}/export/job/{uid}', timeout=None)
        if resp.status_code == 200:
            break
        time.sleep(5)
        logger.info(f'Waiting... (current code {resp.status_code})')

    zipfn = f'{out_name}.zip'
    with open(zipfn, mode='wb') as fh:
        fh.write(resp.content)

    with zipfile.ZipFile(zipfn, 'r') as fh:
        fh.extractall(out_name)
    os.remove(zipfn)


def pull_immunedb_metadata(endpoint):
    resp = requests.post(f'{endpoint}/samples/list').json()
    return pd.DataFrame(
        [
            {
                'replicate_name': r['name'],
                'subject': r['subject']['identifier'],
                **r['metadata'],
            }
            for r in resp
        ],
    )


def pull_immunedb_data(
    endpoint, db_name, out_name, skip_existing=True
):  # pragma: no cover
    '''
    Downloads unpooled clonal data from an ImmuneDB instance.

    Parameters
    ----------
    endpoint : str
        The endpoint to the hosted ImmuneDB instance.  For example
        ``https://mydomain.com/immunedb``.
    db_name : str
        The database name itself.  For example ``my_db``.
    out_name : str
        The name of the directory into which the data will be saved.

    Returns
    -------
    A ``pd.DataFrame`` with all clonal data downloaded from the ImmuneDB
    instance.

    Examples
    --------
    .. code-block:: python

        >>> io.pull_immunedb_data(
            'https://mydomain.com/immunedb',
            'my_db',
            'my_db_data'
        )


    '''
    try:
        os.mkdir(out_name)
        endpoint = f'{endpoint}/api/{db_name}'
        logger.info(f'Downloading data for {db_name}')
        _run_job_and_get_result(
            endpoint,
            'export/clones?format=immunedb&pool_on=sample&samples=T10000',
            out_name,
        )

        metadata = pull_immunedb_metadata(endpoint)
        metadata.to_csv(f'{out_name}/metadata.tsv', sep='\t', index=False)
        logger.info(f'Complete!  Data is in directory "{out_name}".')
    except FileExistsError as e:
        if not skip_existing:
            raise e

    return read_directory(out_name)


DEFAULT_METADATA_REGEX = re.compile(
    r'(?P<METADATA_sequencing_date>\d{4}-\d{2}-\d{2})'
    r'-(?P<METADATA_species>(human|mouse))'
    r'-(?P<METADATA_locus>(\w+))'
    r'-(?P<subject>(\w+))'
    r'-rep(?P<METADATA_replicate_number>(\d+))'
)


def convert_igblast(path, metadata_regex=DEFAULT_METADATA_REGEX):
    dfs = []
    for fn in glob.glob(os.path.join(path, '*.tsv')):
        df = pd.read_csv(fn, sep='\t')
        df['replicate_name'] = os.path.basename(fn).split('.')[0]
        metadata = re.search(metadata_regex, fn)
        for k, v in metadata.groupdict().items():
            df[k] = v
        dfs.append(df)

    df = pd.concat(dfs)
    df['v_call'] = df['v_call'].str.split('*').str[0]
    df['j_call'] = df['j_call'].str.split('*').str[0]

    meta_keys = metadata.groupdict().keys()
    df['copies'] = 1

    df = (
        df.groupby(
            [
                'replicate_name',
                'v_call',
                'j_call',
                'junction_aa',
                'productive',
                'junction_length',
                *meta_keys,
            ]
        )
        .agg(
            {
                'v_sequence_alignment': lambda s: s.iloc[0],
                'junction': lambda s: s.iloc[0],
                'v_identity': np.mean,
                'copies': np.sum,
            }
        )
        .reset_index()
    )

    remaps = {
        'v_call': 'v_gene',
        'j_call': 'j_gene',
        'productive': 'functional',
        'junction': 'cdr3_nt',
        'junction_length': 'cdr3_num_nts',
        'junction_aa': 'cdr3_aa',
        'v_identity': 'avg_v_identity',
        'v_sequence_alignment': 'top_copy_seq',
        'copies': 'copies',
        'replicate_name': 'replicate_name',
        **{k: k for k in meta_keys},
    }
    df = df[remaps.keys()].rename(remaps, axis=1)
    df['METADATA_replicate_number'] = df['METADATA_replicate_number'].astype(
        int
    )
    df['clone_id'] = range(1, len(df) + 1)
    df['clones'] = 1

    return df.sort_values(list(df.columns), ascending=False)
