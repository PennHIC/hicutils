import numpy as np

from .io import _cols_without


def _aggregate_pool(pool_df, pool_by):
    def _norm(df):
        df['avg_v_identity'] /= df['copies'].sum()
        return df

    name = pool_df.name
    pool_df = pool_df.sort_values('copies', ascending=False)
    pool_df['avg_v_identity'] = pool_df['avg_v_identity'] * pool_df['copies']
    funcs = {c: 'first' for c in pool_df.columns if 'METADATA_' not in c}
    funcs.update(
        {
            'instances': np.sum,
            'copies': np.sum,
            'top_copy_seq': lambda s: s.iloc[0],
            'avg_v_identity': np.sum,
        }
    )
    pool_df = pool_df.groupby('clone_id', as_index=False).agg(funcs)
    total_copies_by_clone = pool_df.groupby('clone_id').copies.sum()
    pool_df['total_copies'] = pool_df['clone_id'].apply(
        lambda c: total_copies_by_clone.loc[c]
    )
    pool_df['avg_v_identity'] /= pool_df['total_copies']
    pool_df = pool_df.drop('total_copies', axis=1)

    pool_df['shm'] = (100 * (1 - pool_df['avg_v_identity'])).round(4)
    pool_df['copies_fraction'] = pool_df['copies'] / pool_df['copies'].sum()
    pool_df['copies_percent'] = 100 * pool_df['copies_fraction']
    pool_df[[p.replace('METADATA_', '') for p in pool_by]] = name
    return pool_df.reset_index(drop=True)


def pool_by(df, pool_by):
    if isinstance(pool_by, str):
        pool_by = [pool_by]

    pool_by = [
        f'METADATA_{p}' if p not in ('subject', 'replicate_name') else p
        for p in pool_by
    ]
    df = df.groupby(pool_by, dropna=False).apply(_aggregate_pool, pool_by)
    return (
        df[_cols_without(df, 'METADATA_')]
        .reset_index(drop=True)
        .drop('replicate_name', axis=1)
    )
