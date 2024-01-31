import numpy as np


def filter_by_overall_copies(df, copies, field='clone_id'):
    '''
    Removes clones identified by ``field`` (default ``clone_id``) from a
    DataFrame with *less than* ``copies`` total copies across all pools.

    Changing ``field`` changes the definition of a clone.  For example, setting
    ``field`` to ``'cdr3_aa'`` will defined clones by their CDR3 AA sequence.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    copies : int
        The minimum copy number of each clone required to be included in the
        resulting DataFrame.

    Returns
    -------
    DataFrame filtered by copies.

    Examples
    --------
    The following removes all clones with less than 5 copies from ``df``:


    .. code-block:: python

        >>> df.copies.min()
        1
        >>> df = filter_by_overall_copies(df, 5)
        >>> df.copies.min()
        4


    '''
    valid_clones = df.groupby(field).copies.sum() >= copies
    valid_clones = valid_clones[valid_clones == True].index  # noqa: E712
    return df[df.clone_id.isin(valid_clones)]


def filter_functional(df, functional=True):
    '''
    Removes clones on functionality, by default removing non-functional clones.
    Setting ``functionality`` to ``False`` removes functional clones.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    functional : bool
        The functionality of the clones to include.  Set to ``True`` (the
        default) to include functional clones only.  Set to ``False`` to only
        include non-functional clones.

    Returns
    -------
    DataFrame filtered by functionality.

    '''

    return df[df.functional == ('T' if functional else 'F')]


def filter_by_gene_frequency(df, min_frequency, by='subject', gene='v_gene'):
    '''
    Removes clones in ``by`` (defaults to ``subject``) which have an overall
    ``gene`` usage less than or equal to ``min_frequency``.

    For example, if ``min_frequency=0.05`` and ``by='subject'``, all clones
    using a V-gene with a frequency less than or equal to 0.05 in a given
    subject are removed.

    df : pd.DataFrame
        The DataFrame to filter.
    min_frequency : float
        The minimum frequency of a gene in ``by`` that should be included.
    by : str
        The column on which to calculate frequency.  Defaults to ``subject``.
    gene : str
        The gene on which to filter.  Accepts ``v_gene`` or ``j_gene``
        defaulting to ``j_gene``

    Returns
    ------
    DataFrame filtered on gene frequency in ``by``.

    '''
    assert gene in ('v_gene', 'j_gene')

    def _filter_group(df):
        pdf = df.groupby(gene).clone_id.nunique().to_frame().reset_index()
        pdf.clone_id /= pdf.clone_id.sum()
        valid_genes = pdf[pdf.clone_id >= min_frequency][gene]
        return df[df[gene].isin(valid_genes)]

    return df.groupby(by, group_keys=False).apply(_filter_group)


def _overlap_pivot(df, pool):
    return df.pivot_table(
        index='clone_id', columns=pool, values='copies', aggfunc=np.sum
    )


def filter_number_of_pools(df, pool, n, func='greater_equal', limit_to=None):
    '''
    Filters clones based on the number of pools in which it occurs.
    df : pd.DataFrame
        The DataFrame to filter.
    pool : str
        The pool on which to filter.
    n : str
        The number of distinct pools a clone must be in to be included in the
        resulting DataFrame.
    func : function
        The comparison function to use between `n` and the number of
        occurrences of each clone.  The default is `greater_equal` meaning a
        clone must occur in ≥ `n` pools to be included.  Any `numpy` function
        may be used such as `equal` or `less_equal`.
    limit_to : list(str), str, None
        If specified, overlap will be limited to the specified pools.  This is
        useful to filter clones based on their overlap in a subset of pools.
    Returns
    -------
    DataFrame filtered by number of pools.

    '''

    func = getattr(np, func)
    counts = _overlap_pivot(df, pool)
    if limit_to:
        counts = counts[limit_to]
    counts = (counts / counts).sum(axis=1)
    counts = set(counts[func(counts, n)].index)
    return df[df.clone_id.isin(counts)]


def filter_by_presence(df, pool, pool_value):
    '''
    Filters clones based on presence in a given pool.

    df : pd.DataFrame
        The DataFrame to filter.
    pool : str
        The pool on which to filter.
    pool_value : str
        The pool value on which to filter.

    Returns
    -------
    DataFrame filtered by number of pools.

    '''
    if pool_value not in df[pool].unique():
        raise KeyError(f'"{pool_value}" is not a value for pool "{pool}"')
    overlap_df = df.pivot_table(
        index='clone_id', columns=pool, values='copies', aggfunc=np.sum
    ).fillna(0)

    clone_ids = overlap_df[overlap_df[pool_value] > 0].index
    return df[df.clone_id.isin(clone_ids)]


def remove_potential_contaminates(
    df, pool, pool_values, clone_feature='cdr3_nt'
):
    '''
    Removes clones based on ``clone_feature`` (defaults to CDR3 NT) which occur
    in  ``pool`` with values ``pool_values``.  For example, to remove all
    clones with CDR3 NT sequences found in subjects 'Fibroblast' and 'Water':

    .. code-block:: python

        remove_potential_contaminates(df, 'subject', ['Fibroblast', 'Water'])

    df : pd.DataFrame
        The DataFrame to filter.
    pool : str
        The pool to use for filtering.
    pool_values : list
        The values of ``pool`` which should be the basis of clonal exclusion.
    clone_feature : str
        The clone feature to use for filtering.  For example ``cdr3_nt`` (the
        default) will use the CDR3 NT sequence as the basis for removing other
        clones.

    Returns
    -------
    DataFrame with clones occurring in ``pool`` with values ``pool_values``
    excluded on the basis of ``clone_feature``.

    '''

    remove_values = df[df[pool].isin(pool_values)][clone_feature].unique()
    return df[~df[clone_feature].isin(remove_values)]
