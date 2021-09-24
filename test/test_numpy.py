import pytest
import numpy as np
import tonic.functional as F
import utils


class TestFunctionalNumpy:
    @pytest.mark.parametrize("ordering", ["typx"])
    def testMixEvents(self, ordering):
        (stream1, images, sensor_size,) = utils.create_random_input(dtype)
        (stream2, images, sensor_size,) = utils.create_random_input(dtype)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        events = (stream1, stream2)

        mixed_events_no_offset, _ = F.mix_ev_streams_numpy(
            events, offsets=None, check_conflicts=False, sensor_size=sensor_size,
        )

        mixed_events_random_offset, _ = F.mix_ev_streams_numpy(
            events, offsets="Random", check_conflicts=False, sensor_size=sensor_size,
        )

        mixed_events_defined_offset, _ = F.mix_ev_streams_numpy(
            events, offsets=(0, 100), check_conflicts=False, sensor_size=sensor_size,
        )

        mixed_events_conflict, num_conflicts = F.mix_ev_streams_numpy(
            (stream1, stream1),
            offsets=None,
            check_conflicts=True,
            sensor_size=sensor_size,
        )

        no_offset_monotonic = np.all(
            mixed_events_no_offset[1:, t_index] >= mixed_events_no_offset[:-1, t_index],
            axis=0,
        )
        random_offset_monotonic = np.all(
            mixed_events_random_offset[1:, t_index]
            >= mixed_events_random_offset[:-1, t_index],
            axis=0,
        )
        defined_offset_monotonic = np.all(
            mixed_events_defined_offset[1:, t_index]
            >= mixed_events_defined_offset[:-1, t_index],
            axis=0,
        )
        conflict_offset_monotonic = np.all(
            mixed_events_conflict[1:, t_index] >= mixed_events_conflict[:-1, t_index],
            axis=0,
        )
        all_colisions_detected = len(stream1) == num_conflicts

        assert (
            all_colisions_detected
        ), "Missed some event colisions, may cause processing problems."
        assert no_offset_monotonic, "Result was not monotonic."
        assert random_offset_monotonic, "Result was not monotonic."
        assert defined_offset_monotonic, "Result was not monotonic."
        assert conflict_offset_monotonic, "Result was not monotonic."
