import os

import pandas as pd


def is_expected(output_df, expected_df_path):
    if os.environ.get('GENERATE', False):
        print('Generating {}'.format(expected_df_path))
        output_df.to_csv(expected_df_path, sep='\t', index=False)

    output_df = output_df.reset_index(drop=True)
    expected_df = pd.read_csv(
        expected_df_path, sep='\t'
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(output_df, expected_df, check_names=False)
