import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def basic_clustermap(df, normalize_by, cluster_by, min_frequency=0, **kwargs):
    assert normalize_by in ('rows', 'cols', None)
    assert cluster_by in ('rows', 'cols', 'both', None)
    if normalize_by == 'rows':
        df = df.div(df.sum(axis=1), axis=0)
    elif normalize_by == 'cols':
        df = df.div(df.sum(axis=0), axis=1)

    df[df < min_frequency] = np.nan
    df = df.dropna(axis=1, how='all').fillna(0)

    g = sns.clustermap(
        data=df,
        cmap='coolwarm',
        figsize=kwargs.pop('figsize', (20, len(df) * 2)),
        mask=df == 0,
        linewidths=1,
        row_cluster=cluster_by in ('both', 'rows') and len(df) > 2,
        col_cluster=cluster_by in ('both', 'cols') and len(df.columns) > 2,
        xticklabels=True,
        yticklabels=True,
        **kwargs
    )
    g.cax.set_visible(False)
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    plt.setp(g.ax_heatmap.get_yticklabels(), va='center')
    return g
