import tonic
import pytest


def test_data_selection():
    dataset = tonic.datasets.DSEC(
        "data", split="thun_00_a", data_selection="image_timestamps"
    )
    data, targets = dataset[0]
    assert len(data) == 1
    assert len(targets) == 0


def test_multi_data_selection():
    dataset = tonic.datasets.DSEC(
        "data",
        split="thun_00_a",
        data_selection=["image_timestamps", "image_exposure_timestamps_left"],
    )
    data, targets = dataset[0]
    assert len(data) == 2
    assert len(targets) == 0


def test_target_selection():
    dataset = tonic.datasets.DSEC(
        "data",
        split="thun_00_a",
        data_selection="image_timestamps",
        target_selection="disparity_timestamps",
    )
    data, targets = dataset[0]
    assert len(data) == 1
    assert len(targets) == 1


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
