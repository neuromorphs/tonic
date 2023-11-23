import pytest

import tonic


def test_train_split():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="train",
        data_selection=[],
    )
    assert len(dataset) == 41


def test_test_split():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="test",
        data_selection=[],
    )
    assert len(dataset) == 12


# @pytest.mark.skip("Data not available from CI server...")
def test_optical_flow_subset():
    with pytest.warns():
        dataset = tonic.datasets.DSEC(
            save_to="data",
            split="train",
            data_selection=[],
            target_selection="optical_flow_forward_timestamps",
        )
    assert len(dataset) == 18
    data, targets = dataset[0]
    assert len(data) == 0
    assert len(targets) == 1


# @pytest.mark.skip("Data not available from CI server...")
def test_data_selection():
    dataset = tonic.datasets.DSEC(
        save_to="data", split="thun_00_a", data_selection="image_timestamps"
    )
    data, targets = dataset[0]
    assert len(data) == 1
    assert len(targets) == 0


# @pytest.mark.skip("Data not available from CI server...")
def test_multi_data_selection():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="thun_00_a",
        data_selection=["image_timestamps", "image_exposure_timestamps_left"],
    )
    data, targets = dataset[0]
    assert len(data) == 2
    assert len(targets) == 0


# @pytest.mark.skip("Data not available from CI server...")
def test_target_selection():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="thun_00_a",
        data_selection=[],
        target_selection="disparity_timestamps",
    )
    data, targets = dataset[0]
    assert len(data) == 0
    assert len(targets) == 1


# @pytest.mark.skip("Data not available from CI server...")
def test_target_multiselection():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="thun_00_a",
        data_selection="image_timestamps",
        target_selection=["disparity_timestamps", "optical_flow_forward_timestamps"],
    )
    data, targets = dataset[0]
    assert len(data) == 1
    assert len(targets) == 2


# @pytest.mark.skip("Data not available from CI server...")
def test_optical_flow():
    dataset = tonic.datasets.DSEC(
        save_to="data",
        split="thun_00_a",
        data_selection="image_timestamps",
        target_selection=[
            "optical_flow_forward_event",
            "optical_flow_forward_timestamps",
        ],
    )
    data, targets = dataset[0]
    assert len(data) == 1
    assert len(targets) == 2


def test_raises_exception_wrong_recording_name():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            save_to="data", split="wrong_name", data_selection=["image_timestamps"]
        )


def test_raises_exception_data_name():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            save_to="data", split="thun_00_a", data_selection="wrong_data"
        )


def test_raises_exception_wrong_recording_name():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            save_to="data",
            split="thun_00_a",
            data_selection=[],
            target_selection="wrong_target",
        )


def test_raises_exception_combination_test_train():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            save_to="data",
            split=["zurich_city_11_c", "thun_01_a"],
            data_selection=["image_timestamps"],
        )


def test_raises_exception_targets_for_test():
    with pytest.raises(Exception):
        tonic.datasets.DSEC(
            save_to="data",
            split="test",
            data_selection=["image_timestamps"],
            target_selection="disparity_timestamps",
        )
