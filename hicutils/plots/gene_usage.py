import numpy as np
import seaborn as sns

from .heatmap import basic_clustermap


def plot_gene_heatmap(df, pool, gene, size_metric='clones',
                      normalize_by='rows', cluster_by='both',
                      figsize=(30, 10)):
    '''
    Generates a gene-usage heatmap showing the utilization of each V or J gene
    based on pools.

    Parameters
    -----------
    df : pd.DataFrame
        The DataFrame to use as the source of gene usage information.
    pool : str
        The pooling column to use for each row of the heatmap.
    gene : str (``v_gene`` or ``j_gene``)
        The gene to plot. Must be either ``v_gene`` or ``j_gene``.
    size_metric : str
        The size metric which is plotted as the intensity of each cell.  Must
        be one of ``clones``, ``copies``, or ``uniques``.
    normalize_by : str
        Sets how to normalize the plot.  If set to ``rows`` (the default) each
        row is normalized to sum to one.  Setting it to ``cols`` causes each
        column (gene) to sum to one.
    cluster_by : str (``rows``, ``cols``, or ``both``) or None
        Sets which clustering to display.  Valid values are ``rows``, ``cols``,
        ``both``, or clustering can be disabled with ``None``.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert gene in ('v_gene', 'j_gene')
    assert size_metric in ('clones', 'copies', 'uniques')

    pdf = df.pivot_table(
        index=pool, columns=gene, values=size_metric, aggfunc=np.sum
    ).fillna(0)

    total_clones = df.groupby(pool).clone_id.nunique()
    pdf.index = [
        f'{c} ({int(total_clones.loc[c])})'
        for c in pdf.index
    ]

    g = basic_clustermap(pdf, normalize_by, cluster_by, figsize)
    return g, pdf


def plot_gene_frequency(df, pool, gene, size_metric='clones', by=None,
                        **kwargs):
    '''
    Generates a gene-usage dot/bar plot showing the utilization of each V or J
    gene based on pools.

    Parameters
    -----------
    df : pd.DataFrame
        The DataFrame to use as the source of gene usage information.
    pool : str
        The pooling column to use for each row of the heatmap.
    gene : str (``v_gene`` or ``j_gene``)
        The gene to plot. Must be either ``v_gene`` or ``j_gene``.
    size_metric : str
        The size metric which is plotted as the intensity of each cell.  Must
        be one of ``clones``, ``copies``, or ``uniques``.
    by : str
        The feature to use as the ``hue`` variable for the plot.  Must be
        included in the ``pool`` parameter.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    assert gene in ('v_gene', 'j_gene')
    assert size_metric in ('clones', 'copies', 'uniques')

    if type(pool) == str:
        pool = [pool]
    pdf = df.groupby([*pool, gene])
    if size_metric == 'clones':
        pdf = pdf.clone_id.nunique().to_frame().reset_index().rename(
            {
                'clone_id': 'clones'
            },
            axis=1
        )
    else:
        pdf = pdf[size_metric].sum().to_frame().reset_index()

    pdf['freq'] = (
        pdf
        .groupby(pool)[size_metric]
        .apply(lambda c: 100 * c / c.sum())
    )

    g = sns.catplot(
        data=pdf,
        x=gene,
        y='freq',
        hue=by,
        height=kwargs.pop('height', 6),
        aspect=kwargs.pop('aspect', 3),
        dodge=kwargs.pop('dodge', True),
        kind=kwargs.pop('kind', 'strip'),
        **kwargs
    )
    g.set(xlabel='', ylabel=f'{size_metric.capitalize()} %')
    g.axes[0][0].set_xticklabels(g.axes[0][0].get_xticklabels(), rotation=90)
    return g, pdf
