import numpy as np
from tonic.slicers import (
    SliceAtIndices,
    SliceAtTimePoints,
    SliceByEventCount,
    SliceByTime,
)
from tonic.functional.slicing import slice_at_indices


def test_slice_at_indices_class():

    slicer = SliceAtIndices(np.array([0, 3, 5]), np.array([3, 5, 7]))
    data = np.arange(7)

    slices = slicer.slice(data)
    assert len(slices) == 3


def test_slice_at_indices_method():

    data = np.arange(7)
    slices = slice_at_indices(data, np.array([0, 3, 5]), np.array([3, 5, 7]))

    assert len(slices) == 3


def test_slice_at_time_points_class():
    data = np.arange(100).astype(dtype=[("t", float)])
    slicer = SliceAtTimePoints(np.array([10, 20, 50]), np.array([20, 50, 100]))

    slices = slicer.slice(data)

    assert len(slices) == 3


def test_slice_by_event_count():
    data = np.arange(100)
    slicer = SliceByEventCount(event_count=50, overlap=25, include_incomplete=False)

    slices = slicer.slice(data)

    assert len(slices) == 3


def test_slice_by_time():
    data = np.arange(101).astype(dtype=[("t", float)])
    slicer = SliceByTime(time_window=5)

    slices = slicer.slice(data)

    assert len(slices) == 20
