import pytest

from hicutils.core import io, metadata
from .expected import is_expected


@pytest.mark.parametrize(
    'path',
    [
        'tests/input',
    ]
)
def test_read_tsvs(path):
    df = io.read_directory(path).sort_values('clone_id')
    is_expected(df, 'tests/expected/io_test.tsv')
    mdf = metadata.make_metadata_table(df, 'METADATA_disease').reset_index()
    is_expected(mdf, 'tests/expected/metadata.tsv')


@pytest.mark.parametrize(
    'path',
    [
        'tests/input/igblast',
    ]
)
def test_convert_igblast(path):
    df = io.convert_igblast(path)
    is_expected(df, 'tests/expected/igblast_test.tsv')
