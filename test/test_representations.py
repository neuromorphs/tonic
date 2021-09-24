import pytest
import numpy as np
import tonic.transforms as transforms
import utils


class TestRepresentations:
    @pytest.mark.parametrize(
        "time_window, spike_count, n_time_bins, n_event_bins, overlap,"
        " include_incomplete, merge_polarities",
        [
            (2000, None, None, None, 0, False, True),
            (2000, None, None, None, 200, True, False),
            (1000, None, None, None, 100, True, True),
            (None, 2000, None, None, 0, False, True),
            (None, 2000, None, None, 200, True, False),
            (None, 2000, None, None, 100, True, True),
            (None, None, 5, None, 0, False, True),
            (None, None, 5, None, 0.1, False, False),
            (None, None, 5, None, 0.25, True, False),
            (None, None, None, 5, 0, True, False),
            (None, None, None, 5, 0.1, False, True),
            (None, None, None, 5, 0.25, False, False),
        ],
    )
    def test_representation_frame(
        self,
        ordering,
        time_window,
        spike_count,
        n_time_bins,
        n_event_bins,
        overlap,
        include_incomplete,
        merge_polarities,
    ):
        (orig_events, sensor_size,) = create_random_input()
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=time_window,
            spike_count=spike_count,
            n_time_bins=n_time_bins,
            n_event_bins=n_event_bins,
            overlap=overlap,
            include_incomplete=include_incomplete,
            merge_polarities=merge_polarities,
        )

        frames = transform(events=orig_events.copy())

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

        if spike_count is not None:
            stride = spike_count - overlap
            n_events = orig_events.shape[0]
            if include_incomplete:
                assert frames.shape[0] == int(
                    np.ceil((n_events - spike_count) / stride) + 1
                )
            else:
                assert frames.shape[0] == int(
                    np.floor((n_events - spike_count) / stride) + 1
                )

        if n_time_bins is not None:
            assert frames.shape[0] == n_time_bins

        if n_event_bins is not None:
            assert frames.shape[0] == n_event_bins

        if merge_polarities:
            assert frames.shape[1] == 1

    @pytest.mark.parametrize(
        "merge_polarities", [(True), (False),],
    )
    def test_representation_sparse_tensor(self, merge_polarities):
        (orig_events, sensor_size,) = create_random_input()
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.ToSparseTensor(
            sensor_size=sensor_size, merge_polarities=merge_polarities,
        )

        sparse_tensor = transform(events=orig_events.copy())

        assert sparse_tensor.coalesce().values().sum() == orig_events.shape[0]
        assert sparse_tensor.shape[0] == int(orig_events[:, t_index][-1] + 1)
        assert sparse_tensor.shape[1] == 1 if merge_polarities else 2
        assert sparse_tensor.shape[2:] == sensor_size

    @pytest.mark.parametrize(
        "merge_polarities", [(True), (False),],
    )
    def test_representation_dense_tensor(self, merge_polarities):
        (orig_events, sensor_size,) = create_random_input()
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.ToDenseTensor(
            sensor_size=sensor_size, merge_polarities=merge_polarities,
        )

        tensor = transform(events=orig_events.copy())

        assert tensor.sum() == orig_events.shape[0]
        assert tensor.shape[0] == int(orig_events[:, t_index][-1]) + 1
        assert tensor.shape[1] == 1 if merge_polarities else 2
        assert tensor.shape[2:] == sensor_size

    @pytest.mark.parametrize(
        "surface_dimensions, tau, merge_polarities",
        [((15, 15), 100, True), ((3, 3), 10, False), (None, 1e4, False),],
    )
    def test_representation_time_surface(
        self, surface_dimensions, tau, merge_polarities
    ):
        (orig_events, sensor_size,) = create_random_input()

        transform = transforms.ToTimesurface(
            sensor_size=sensor_size,
            surface_dimensions=surface_dimensions,
            tau=tau,
            merge_polarities=merge_polarities,
        )

        surfaces = transform(events=orig_events.copy())

        assert surfaces.shape[0] == len(orig_events)
        assert surfaces.shape[1] == 1 if merge_polarities else 2
        if surface_dimensions:
            assert surfaces.shape[2:] == surface_dimensions
        else:
            assert surfaces.shape[2:] == sensor_size

    @pytest.mark.parametrize("n_time_bins", [(10), (1)])
    def test_representation_voxel_grid(self, n_time_bins):
        (orig_events, sensor_size,) = create_random_input()

        transform = transforms.ToVoxelGrid(
            sensor_size=sensor_size, n_time_bins=n_time_bins
        )

        volumes = transform(events=orig_events.copy())
        assert volumes.shape == (n_time_bins, *sensor_size[::-1])
