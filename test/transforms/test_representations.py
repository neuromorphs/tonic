import pytest
import numpy as np
import tonic.transforms as transforms
import utils


class TestRepresentations:
    @pytest.mark.parametrize(
        "ordering, time_window, spike_count, n_time_bins, n_event_bins, overlap,"
        " include_incomplete, merge_polarities",
        [
            ("xytp", 2000, None, None, None, 0, False, True),
            ("txyp", 2000, None, None, None, 200, True, False),
            ("xytp", 1000, None, None, None, 100, True, True),
            ("txyp", None, 2000, None, None, 0, False, True),
            ("xytp", None, 2000, None, None, 200, True, False),
            ("txyp", None, 2000, None, None, 100, True, True),
            ("xytp", None, None, 5, None, 0, False, True),
            ("xytp", None, None, 5, None, 0.1, False, False),
            ("xytp", None, None, 5, None, 0.25, True, False),
            ("xytp", None, None, None, 5, 0, True, False),
            ("xytp", None, None, None, 5, 0.1, False, True),
            ("xytp", None, None, None, 5, 0.25, False, False),
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
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.ToFrame(
            time_window=time_window,
            spike_count=spike_count,
            n_time_bins=n_time_bins,
            n_event_bins=n_event_bins,
            overlap=overlap,
            include_incomplete=include_incomplete,
            merge_polarities=merge_polarities,
        )

        frames, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        if time_window is not None:
            stride = time_window - overlap
            times = orig_events[:, t_index]
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
        "ordering, surface_dimensions, tau, merge_polarities",
        [("xytp", (15, 15), 100, True), ("typx", (3, 3), 10, False)],
    )
    def test_representation_time_surface(
        self, ordering, surface_dimensions, tau, merge_polarities
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        transform = transforms.ToTimesurface(
            surface_dimensions=surface_dimensions,
            tau=tau,
            merge_polarities=merge_polarities,
        )

        surfaces, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert surfaces.shape[0] == len(orig_events)
        assert surfaces.shape[1] == 1 if merge_polarities else 2
        assert surfaces.shape[2:] == surface_dimensions

    @pytest.mark.parametrize("ordering, n_time_bins", [("xytp", 10), ("typx", 1)])
    def test_representation_voxel_grid(self, ordering, n_time_bins):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        transform = transforms.ToVoxelGrid(n_time_bins=n_time_bins)

        volumes, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )
        assert volumes.shape == (n_time_bins, *sensor_size[::-1])
