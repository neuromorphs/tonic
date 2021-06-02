import numpy as np
import torch


def to_sparse_tensor_pytorch(events, sensor_size, ordering, merge_polarities=False):
    """Sparse Tensor PyTorch representation. See https://pytorch.org/docs/stable/sparse.html for details.

    Args:
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        sparse tensor in TxWxH format
    """
    assert "x" and "y" and "t" and "p" in ordering
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")

    if len(events.shape) != 2:
        raise RuntimeError(
            "Will only convert to sparse tensor from (N,E) (i.e., a list of events)"
            " dimension."
        )

    if merge_polarities:
        events[:, p_index] = np.zeros(n_of_events)
    else:
        pols = events[:, p_index]
        pols[pols == 0] = -1

    max_time = int(max(events[:, t_index]) + 1)
    max_x = int(max(events[:, x_index]) + 1)
    max_y = int(max(events[:, y_index]) + 1)

    indices = torch.LongTensor(events[:, [t_index, x_index, y_index]]).T
    values = torch.FloatTensor(events[:, p_index])
    return torch.sparse.FloatTensor(
        indices, values, torch.Size([max_time, max_x, max_y])
    )
