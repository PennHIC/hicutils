from collections import Counter
import re

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_clone_sizes(df, cutoff=None, **kwargs):
    '''
    Plots the distribution of clone sizes in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the clone size distribution.
    cutoff : int or None
        Aggregate all clones with ``cutoff`` or more copies into one bin on the
        right side of the graph.  This is useful to condense the tail of the
        plotted distribution.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''
    df = (
        df.groupby('copies')
        .clone_id.nunique()
        .to_frame()
        .rename(
            {
                'clone_id': 'clones',
            },
            axis=1,
        )
    )
    df['clones'] = 100 * df['clones'] / df['clones'].sum()
    df = df.reindex(range(df.index.min(), df.index.max())).reset_index()
    if cutoff:
        clones = df[df['copies'] >= cutoff].clones.sum()
        df = df[df['copies'] < cutoff]
        df = pd.concat(
            [df, pd.DataFrame([{'copies': f'{cutoff}+', 'clones': clones}])]
        )

    g = sns.catplot(
        data=df,
        x='copies',
        y='clones',
        kind='bar',
        color=sns.color_palette()[0],
        height=kwargs.pop('height', 6),
        aspect=kwargs.pop('aspect', 1.8),
        **kwargs,
    )
    g.set(xlabel='Copies', ylabel='% of Clones')

    return g, df


def plot_top_clones(
    df, cutoff=20, annotate=False, color=sns.color_palette()[3], figsize=(12, 8)
):
    '''
    Plots the copy-number frequency of the top ``cutoff`` clones (default 20).
    Optionally, the ``annotate`` keyword can be set to one or more clone
    features to annotate each bar.  For example setting ``annotate=('v_gene',
    'cdr3_aa')`` will show the V-gene and CDR3 AA for each clone.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the top clones.
    cutoff : int
        The number of clones to plot, defaults to 20.
    annotate : str, list, or None
        The feature(s) to annotate for each clone
    color : str
        The color to use for bars.
    figsize : tuple
        The ``(width, height)`` of the plot.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''

    if isinstance(annotate, str):
        annotate = [annotate]
    df = df.copy()
    df = df.sort_values('copies', ascending=False)
    df['copies_percent'] = 100 * df['copies'] / df['copies'].sum()
    df['rank'] = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    cdf = df[:cutoff]
    g = sns.barplot(x='rank', y='copies_percent', data=cdf, color=color, ax=ax)
    g.set_xlabel('Size')
    g.set_ylabel('% of Clones')

    if annotate:
        for i, p in enumerate(g.patches):
            ax.annotate(
                ' '.join([str(s) for s in df.iloc[i][annotate]]),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha='center',
                va='center',
                fontsize=10,
                color='black',
                rotation=90,
                xytext=(0, 30),
                textcoords='offset points',
            )
    a = plt.axes([0.69, 0.58, 0.2, 0.2], facecolor='y')
    colors = [sns.color_palette()[3], sns.color_palette('Reds', n_colors=5)[1]]
    frac = df[:cutoff]['copies_percent'].sum()

    top_df = pd.DataFrame(
        {'percent': [frac, max(0, 100 - frac)]}, index=['top', 'rest']
    )

    ax = top_df.plot.pie(
        y='percent',
        colors=colors,
        legend=False,
        labels=[f'{round(frac, 1)}%', ''],
        ax=a,
    )
    ax.set_title(f'% of Total Copies\n({df.copies.sum()})')
    ax.set_ylabel('')

    return g, cdf


def _range_for_pool(df, portions, total_clones, intervals):
    pool = df.name
    total_clones[pool] += len(df)

    for start, end in zip(intervals[:-1], intervals[1:]):
        portions.append(
            {
                'pool': pool,
                'start': start,
                'end': end,
                'copies': df[start:end].copies.sum(),
            }
        )
    portions.append(
        {
            'pool': pool,
            'start': end,
            'end': '+',
            'copies': df[end:].copies.sum(),
        }
    )
    return portions


def _get_d(df, d):
    return (
        df.sort_values('copies', ascending=False).head(d).copies.sum()
    ) / df.copies.sum()


def _label(r):
    if r['end'] != '+':
        return f'{r["start"] + 1}-{r["end"]}'
    return f'{r["start"] + 1}+'


def plot_ranges(df, pool, intervals=(10, 100, 1000), order_func=None, **kwargs):
    intervals = [0, *intervals]
    portions = []
    total_clones = Counter()

    df.groupby(pool).apply(_range_for_pool, portions, total_clones, intervals)

    portions = pd.DataFrame(portions)
    portions['pool'] = portions['pool'].apply(
        lambda p: f'{p} ({total_clones[p]})'
    )
    portions['range'] = portions.apply(_label, axis=1)
    pdf = portions.pivot_table(
        index='pool', columns='range', values='copies', aggfunc=np.sum
    )
    pdf = pdf.div(-pdf.sum(axis=1), axis=0)

    order = pd.Series(
        int(re.search(r'\d+', c).group(0)) for c in pdf.columns
    ).argsort()

    pdf = pdf[pdf.columns[order]].reindex(
        pdf[pdf.columns[:-1]].sum(axis=1).sort_values().index
    )

    if order_func:
        pdf = order_func(pdf)
    else:
        d20s = list(
            df.groupby(pool)
            .apply(_get_d, 20)
            .sort_values(ascending=False)
            .index
        )
        pdf['order'] = [
            d20s.index(label.rsplit(' (', 1)[0]) for label in pdf.index
        ]
        pdf = pdf.sort_values('order').drop('order', axis=1)

    colors = [
        *kwargs.pop('color', sns.color_palette()[: len(intervals) - 1]),
        (0.86, 0.86, 0.86),  # gray
    ]

    ax = pdf.plot.bar(
        stacked=True, figsize=kwargs.get('figsize', (10, 5)), color=colors
    )
    ax.set_yticklabels([round(abs(tick), 2) for tick in ax.get_yticks()])
    ax.set_xlabel('')
    ax.set_ylabel('Fraction of Copies')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    return ax, pdf


def plot_d_index(
        df,
        pool,
        cutoff=20,
        **kwargs):
    '''
    Plots the Dx index for clones in ``df``.  The default cutoff value is 20
    and the generated figure is a dot plot of Dx values stratified by ``pool``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame used to plot the top clones.
    cutoff : int
        The D-value to use as a cutoff, defaults to 20.

    Returns
    -------
    A tuple ``(g, df)`` where ``g`` is a handle to the plot and ``df`` is the
    underlying DataFrame.

    '''
    df = (
        df
        .groupby(pool)
        .apply(_get_d, cutoff)
        .to_frame()
        .reset_index()
        .rename({0: 'd'}, axis=1)
    )

    g = sns.catplot(
        data=df,
        x=pool,
        y='d',
        **kwargs
    )
    g.axes.flatten()[0].set_xticklabels(
        g.axes.flatten()[0].get_xticklabels(),
        rotation=90
    )
    g.set(xlabel='', ylabel=f'D{cutoff} index')

    return g, df
