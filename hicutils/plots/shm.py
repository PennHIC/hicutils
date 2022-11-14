import re

import pandas as pd
import seaborn as sns


def _add_counts(df, field):
    sizes = df.groupby(field).size()
    df[field] = df[field].apply(lambda f: f'{f} ({sizes[f]})')
    return df


def _get_shm(pdf, df, pool, size_metric):
    total = df[df[pool] == pdf.name[1]][size_metric].sum()
    ret = pd.Series({'size': 100 * pdf[size_metric].sum() / total})
    return ret


def plot_shm_distribution(df, pool, size_metric, **kwargs):
    '''
    Plots the SHM distribution of a pooled DataFrame using either clones,
    copies, or uniques as a size metric.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the SHM distribution.
    pool : str
        The pool to use for plotting.
    size_metric : str
        The metric to determine each clones' size.  Must be ``clones``,
        ``copies``, or ``uniques``.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert size_metric in ('clones', 'copies', 'uniques')
    df = df.copy()
    df = _add_counts(df, pool)
    df['shm'] = df['shm'].round()
    df = (
        df
        .groupby(['shm', pool])
        .apply(_get_shm, df, pool, size_metric).reset_index()
    )

    with sns.plotting_context('poster'):
        g = sns.relplot(
            data=df,
            x='shm',
            y='size',
            hue=pool,
            kind='line',
            height=kwargs.pop('height', 8),
            aspect=kwargs.pop('aspect', 1.5),
            **kwargs
        )
        g.set(
            xlabel='SHM (% of Mutated V-gene NT)',
            ylabel=f'% of {size_metric}',
        )
    return g, df


def plot_shm_aggregate(df, pool, **kwargs):
    '''
    Categorically plots the SHM of each pool.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the SHM.
    pool : str
        The pool to use for plotting.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    g = sns.catplot(
        data=df,
        x=pool,
        y='shm',
        kind=kwargs.pop('kind', 'violin'),
        **kwargs
    )
    g.set(xlabel='', ylabel='SHM %')
    return g, df


def _clone_frac_norm(df):
    df = df.groupby('shm_bucket').clone_id.nunique()
    df = 100 * df / df.sum()
    return df.to_frame()


def _get_bucket(shm, buckets=(1, 2, 5, 10, 20)):
    buckets = [0, *buckets]
    for i, b in enumerate(buckets[:-1]):
        if b <= shm < buckets[i + 1]:
            return f'[{b}-{buckets[i + 1]})'
    return f'{buckets[-1]}+'


def _sort_buckets(buckets):
    bucket_info = [
        (
            int(re.search(r'\d+', b).group()) if '-' in b
            else int(re.search(r'\d+', b).group()) + 1,
            b
        )
        for b in buckets
    ]
    return [
        buckets.index(b[1]) for b in sorted(bucket_info, key=lambda b: b[0])
    ]


def plot_shm_range(df, pool, buckets=(1, 10, 25), **kwargs):
    '''
    Plot the range of clonal SHM for each pool.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the SHM.
    pool : str
        The pool to use for plotting.
    buckets : list(int)
        A list of cut-points to bin SHM.  The default is ``(1, 10, 25)``
        meaning clones will be stratified by SHM into the buckets ``[1, 10)``,
        ``[10, 25)``, and ``25+``.  All intervals are left-closed; that is the
        lesser value in each interval is inclusive and the greater value is
        exclusive.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    buckets = [b for b in buckets if b < df.shm.max()]
    df = df.copy()
    df['shm_bucket'] = df['shm'].apply(_get_bucket, buckets=buckets)
    df = (
        df
        .groupby(pool)
        .apply(_clone_frac_norm)
        .unstack()['clone_id']
    )
    df = df[[df.columns[i] for i in _sort_buckets(list(df.columns))]]
    with sns.plotting_context('poster'):
        g = (
            df
            .plot
            .bar(
                stacked=True,
                figsize=kwargs.pop('figsize', (8, 6)),
                color=kwargs.pop('color', sns.color_palette()[1:]),
                legend='reverse'
            )
        )
        g.set(xlabel='', ylabel='% of Mutated Clones')
        handles, labels = g.get_legend_handles_labels()
        g.legend(
            reversed(handles),
            reversed(labels),
            loc='upper right',
            bbox_to_anchor=(1.22, 1),
            title='SHM %'
        )

    return g, df
