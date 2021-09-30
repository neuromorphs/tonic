import pytest
import numpy as np
import tonic.transforms as transforms
from utils import create_random_input


class TestRepresentations:
    @pytest.mark.parametrize(
        "time_window, event_count, n_time_bins, n_event_bins, overlap,"
        " include_incomplete",
        [
            (2000, None, None, None, 0, False),
            (2000, None, None, None, 200, True),
            (1000, None, None, None, 100, True),
            (None, 2000, None, None, 0, False),
            (None, 2000, None, None, 200, True),
            (None, 2000, None, None, 100, True),
            (None, None, 5, None, 0, False),
            (None, None, 5, None, 0.1, False),
            (None, None, 5, None, 0.25, True),
            (None, None, None, 5, 0, True),
            (None, None, None, 5, 0.1, False),
            (None, None, None, 5, 0.25, False),
        ],
    )
    def test_representation_frame(
        self,
        time_window,
        event_count,
        n_time_bins,
        n_event_bins,
        overlap,
        include_incomplete,
    ):
        orig_events, sensor_size = create_random_input()

        transform = transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=time_window,
            event_count=event_count,
            n_time_bins=n_time_bins,
            n_event_bins=n_event_bins,
            overlap=overlap,
            include_incomplete=include_incomplete,
        )

        frames = transform(orig_events.copy())

        if time_window is not None:
            stride = time_window - overlap
            times = orig_events["t"]
            if include_incomplete:
                assert frames.shape[0] == int(
                    np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1
                )
            else:
                assert frames.shape[0] == int(
                    np.floor(((times[-1] - times[0]) - time_window) / stride) + 1
                )

        if event_count is not None:
            stride = event_count - overlap
            n_events = orig_events.shape[0]
            if include_incomplete:
                assert frames.shape[0] == int(
                    np.ceil((n_events - event_count) / stride) + 1
                )
            else:
                assert frames.shape[0] == int(
                    np.floor((n_events - event_count) / stride) + 1
                )

        if n_time_bins is not None:
            assert frames.shape[0] == n_time_bins

        if n_event_bins is not None:
            assert frames.shape[0] == n_event_bins

    @pytest.mark.parametrize(
        "surface_dimensions, tau,", [((15, 15), 100), ((3, 3), 10), (None, 1e4),],
    )
    def test_representation_time_surface(self, surface_dimensions, tau):
        orig_events, sensor_size = create_random_input()

        transform = transforms.ToTimesurface(
            sensor_size=sensor_size, surface_dimensions=surface_dimensions, tau=tau,
        )

        surfaces = transform(orig_events.copy())

        assert surfaces.shape[0] == len(orig_events)
        assert surfaces.shape[1] == 2
        if surface_dimensions:
            assert surfaces.shape[2:] == surface_dimensions
        else:
            assert surfaces.shape[3] == sensor_size[1]
            assert surfaces.shape[2] == sensor_size[0]

    @pytest.mark.parametrize("n_time_bins", [(10), (1)])
    def test_representation_voxel_grid(self, n_time_bins):
        orig_events, sensor_size = create_random_input()

        transform = transforms.ToVoxelGrid(
            sensor_size=sensor_size, n_time_bins=n_time_bins
        )

        volumes = transform(orig_events.copy())
        assert volumes.shape == (n_time_bins, *sensor_size[:2])
