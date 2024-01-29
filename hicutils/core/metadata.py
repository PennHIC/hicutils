import numpy as np


def make_metadata_table(df, pool):
    '''
    Generates a metadata table from a pooled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use for the metadata table.
    pool : str
        The pooling column to use for each row of the table.

    Returns
    -------
    A metadata table, indexed by ``pool``.

    '''
    pdf = (
        df.groupby(pool)
        .agg(
            {
                'subject': lambda s: s.nunique(),
                'replicate_name': lambda s: s.nunique(),
                'instances': np.sum,
                'copies': np.sum,
                'cdr3_num_nts': np.mean,
                'avg_v_identity': np.mean,
            }
        )
        .rename({'instances': 'uniques'}, axis=1)
    )
    pdf = pdf.rename(
        {
            'subject': 'subjects',
            'replicate_name': 'replicates',
        },
        axis=1,
    )
    pdf['in_frame'] = df.groupby(pool).apply(
        lambda d: len(d[d.functional == 'T']) / len(d)
    )
    pdf['clones'] = df.groupby(pool).clone_id.nunique()
    pdf['productive_clones'] = (
        df[df.functional == 'T'].groupby(pool).clone_id.nunique()
    )
    return pdf
