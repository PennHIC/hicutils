from collections import Counter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import logomaker

from .heatmap import basic_clustermap


def _get_counts(pdf, size_metric):
    # TODO: Fix this to count clones only once if size_metric = clones
    return pd.DataFrame(pdf.cdr3_aa.apply(
        lambda r: pd.Series(Counter(r))
    ).fillna(0).mul(pdf[size_metric], axis=0).sum()).T


def plot_cdr3_aa_usage(df, pool, size_metric='clones', normalize_by='rows',
                       cluster_by='both', figsize=(20, 10)):
    '''
    Plots CDR3 amino-acid usage separated by pool.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use as the source of CDR3 amino-acid usage
        information.
    pool : str
        The pooling column to use for each row of the heatmap.
    size_metric : str
        The size metric which is plotted as the intensity of each cell.  Must
        be one of ``clones``, ``copies``, or ``uniques``.
    normalize_by : str
        Sets how to normalize the plot.  If set to ``rows`` (the default) each
        row is normalized to sum to one.  Setting it to ``cols`` causes each
        column (amino-acid) to sum to one.
    cluster_by : str (``rows``, ``cols``, or ``both``) or None
        Sets which clustering to display.  Valid values are ``rows``, ``cols``,
        ``both``, or clustering can be disabled with ``None``.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert size_metric in ('clones', 'copies', 'uniques')
    df = df.copy()

    pdf = pd.concat(
        {k: _get_counts(d, size_metric) for k, d in df.groupby(pool)},
        sort=True
    ).fillna(0)
    pdf.index = pdf.index.droplevel(1)

    g = basic_clustermap(pdf, normalize_by, cluster_by, figsize=figsize)
    return g, pdf


def plot_cdr3_logo(df, by, length, hide_ambig=True, **kwargs):
    '''
    Creates a logo plot for CDR3 strings of a given length either by amino-acid
    or nucleotide.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use as the source of CDR3 information.
    by : str
        Either ``cdr3_aa`` to plot amino-acids or ``cdr3_nt`` to plot
        nucleotides.
    length : int
        The length of CDR3s to plot.  Interpreted as the length of ``by``.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert by in ('cdr3_aa', 'cdr3_nt')
    cdrs = df[df[by].str.len() == length][by]
    m = (
        logomaker.alignment_to_matrix(cdrs)
        .pipe(logomaker.transform_matrix, normalize_values=True)
    )
    if hide_ambig:
        if by == 'cdr3_nt' and 'N' in m.columns:
            m = m.drop('N', axis=1)
        if by == 'cdr3_aa' and 'X' in m.columns:
            m = m.drop('X', axis=1)
    color_scheme = kwargs.pop(
        'color_scheme',
        'skylign_protein' if by == 'cdr3_aa' else 'classic'
    )
    g = logomaker.Logo(
        m,
        color_scheme=color_scheme,
        show_spines=False
    )
    g.ax.set(xlabel='Position', ylabel='Fraction of Total')
    return g, m


def plot_cdr3_spectratype(df, color_top=10, **kwargs):
    '''
    Plots CDR3 length while annotating and highlighting the top ``color_top``
    clones.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use for plotting CDR3 length.
    color_top : int
        The number of clones to highlight (default 10).

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    all_df = df.groupby('cdr3_num_nts').copies_percent.sum().reset_index()
    top_df = df.sort_values('copies_percent', ascending=False)[:color_top]
    cdf = (
        pd
        .concat([top_df, all_df], sort=False)
        .fillna('')
        .sort_values('copies_percent', ascending=False)
    )[['cdr3_num_nts', 'copies_percent', 'cdr3_aa']]

    colors = ['#dddddd'] + sns.color_palette()
    g = sns.catplot(
        data=cdf,
        x='cdr3_num_nts',
        y='copies_percent',
        hue='cdr3_aa',
        kind='bar',
        dodge=False,
        height=kwargs.pop('height', 6),
        aspect=kwargs.pop('aspect', 2),
        legend=False,
        errorbar=None,
        palette=colors
    )

    g.set(xticklabels=[
        l if i % 3 == 0 else ''
        for i, l in enumerate(g.ax.get_xticklabels())
    ])
    g.set(xlabel='CDR3 Length (NT)', ylabel='% Total Copies')
    g.despine(left=True)
    plt.legend(loc='upper right')

    return g, cdf


def plot_cdr3_distribution(df, pool, size_metric='clones', **kwargs):
    '''
    Plots CDR3 length distribution.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use for plotting CDR3 length.
    pool : str
        The pooling column to use for hue value.
    size_metric : str
        The size metric to use as the height for each bar.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert size_metric in ('clones', 'copies', 'uniques')

    def _norm(df):
        df[size_metric] /= df[size_metric].sum()
        return df

    pdf = (
        df
        .groupby([pool, 'cdr3_num_nts'])[size_metric].sum()
        .to_frame()
        .reset_index()
        .groupby(pool).apply(_norm)
    )
    g = sns.catplot(
        data=pdf,
        x='cdr3_num_nts',
        y=size_metric,
        hue=pool,
        kind=kwargs.pop('kind', 'bar'),
        **kwargs
    )
    g.set(xlabel='CDR3 length (NT)', ylabel='Clone Fraction')
    return g, pdf
