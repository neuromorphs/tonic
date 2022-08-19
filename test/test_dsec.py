import tonic
import pytest


def test_data_selection():
    tonic.datasets.DSEC("data", split="thun_00_a", data_selection="image_timestamps")


def test_target_selection():
    tonic.datasets.DSEC(
        "data",
        split="thun_00_a",
        data_selection="image_timestamps",
        target_selection="disparity_timestamps",
    )


def test_raises_exception_wrong_recording_name():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            "data", split="wrong_name", data_selection=["image_timestamps"]
        )


def test_raises_exception_combination_test_train():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            "data",
            split=["zurich_city_11_c", "thun_01_a"],
            data_selection=["image_timestamps"],
        )


def test_raises_exception_targets_for_test():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            "data",
            split="test",
            data_selection=["image_timestamps"],
            target_selection="disparity_timestamps",
        )
