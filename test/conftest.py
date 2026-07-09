import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import tonic.datasets

# Set non-interactive backend before any matplotlib imports in CI
if os.environ.get("CI"):
    import matplotlib

    matplotlib.use("Agg")

_DSEC_FIXTURE_ROOT = Path(__file__).parent / "test_data" / "dsec"


@pytest.fixture
def dsec_fixture_dir(tmp_path):
    """Copy minimal DSEC fixtures into a temp save_to directory."""
    dest = tmp_path / "DSEC" / "thun_00_a"
    shutil.copytree(_DSEC_FIXTURE_ROOT / "thun_00_a", dest)
    return str(tmp_path)


@pytest.fixture
def dsec_no_download():
    """Prevent DSEC from attempting network downloads during tests."""
    with patch.object(tonic.datasets.DSEC, "_check_exists", return_value=None):
        yield
