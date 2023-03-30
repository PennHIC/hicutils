import pytest

from hicutils.core import io, metadata
from .expected import is_expected


@pytest.mark.parametrize(
    'path,features',
    [
        ('tests/input', 'disease'),
    ]
)
def test_read_tsvs(path, features):
    df = io.read_tsvs(path, features)
    is_expected(df, 'tests/expected/io_test.tsv')
    mdf = metadata.make_metadata_table(df, 'disease').reset_index()
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
