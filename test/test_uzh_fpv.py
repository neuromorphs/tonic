from unittest.mock import patch

import pytest

import tonic


def test_single_recording():
    with patch.object(tonic.datasets.UZHFPV, "_check_exists", return_value=True):
        dataset = tonic.datasets.UZHFPV(
            save_to="data",
            recording="indoor_forward_3",
        )
    assert len(dataset) == 1


def test_multiple_recordings():
    with patch.object(tonic.datasets.UZHFPV, "_check_exists", return_value=True):
        dataset = tonic.datasets.UZHFPV(
            save_to="data",
            recording=["indoor_forward_3", "outdoor_forward_1"],
        )
    assert len(dataset) == 2


def test_all_recordings():
    with patch.object(tonic.datasets.UZHFPV, "_check_exists", return_value=True):
        dataset = tonic.datasets.UZHFPV(
            save_to="data",
            recording="all",
        )
    assert len(dataset) == len(tonic.datasets.UZHFPV.recordings)


def test_raises_exception_invalid_recording():
    with pytest.raises(RuntimeError):
        tonic.datasets.UZHFPV(
            save_to="data",
            recording="nonexistent_sequence",
        )


def test_raises_exception_invalid_recording_in_list():
    with pytest.raises(RuntimeError):
        tonic.datasets.UZHFPV(
            save_to="data",
            recording=["indoor_forward_3", "nonexistent_sequence"],
        )


def test_all_recording_names_valid():
    """Verify that all recordings are valid sequence names."""
    valid_prefixes = {"indoor_forward_", "indoor_45_", "outdoor_forward_", "outdoor_45_"}
    for rec in tonic.datasets.UZHFPV.recordings:
        assert any(rec.startswith(p) for p in valid_prefixes), (
            f"Recording '{rec}' does not start with a valid prefix."
        )


def test_ground_truth_availability():
    """Verify that ground truth availability flag is a boolean for all recordings."""
    for rec, has_gt in tonic.datasets.UZHFPV.recordings.items():
        assert isinstance(has_gt, bool), (
            f"Ground truth flag for '{rec}' is not a bool: {has_gt!r}"
        )
