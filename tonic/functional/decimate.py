import numpy as np


def decimate_numpy(events: np.ndarray, n: int):
    """Returns 1/n events for each pixel location.

    Parameters:
        events: structured numpy array of events
        n: filter rate
    """

    assert "x" in events.dtype.names
    assert n > 0, "n has to be an integer greater than zero."

    max_x = np.max(events["x"])

    output_events = []
    if "y" in events.dtype.names:
        max_y = np.max(events["y"])
        memory = np.zeros((max_x + 1, max_y + 1))

        for event in events:
            memory[event["x"], event["y"]] += 1
            if memory[event["x"], event["y"]] >= n:
                memory[event["x"], event["y"]] = 0
                output_events.append(event)

    else:
        memory = np.zeros((max_x + 1))

        for event in events:
            memory[event["x"]] += 1
            if memory[event["x"]] >= n:
                memory[event["x"]] = 0
                output_events.append(event)

    return np.array(output_events)
