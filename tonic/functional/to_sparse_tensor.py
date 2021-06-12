import numpy as np
import torch


def to_sparse_tensor_pytorch(events, sensor_size, ordering, merge_polarities=False):
    """Sparse Tensor PyTorch representation. See https://pytorch.org/docs/stable/sparse.html for details.
    A sparse tensor will convert the events to indices of (txy) and values of (p). Note that the indices will
    always be in this order regardless of original events ordering. 

    Args:
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not. 
                                If True, all values in the sparse Tensor will be 1, otherwise -1 or 1. 

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
        events[:, p_index] = 1
    else:
        pols = events[:, p_index]
        pols[pols == 0] = -1

    max_time = int(max(events[:, t_index]) + 1)

    indices = torch.LongTensor(events[:, [t_index, x_index, y_index]]).T
    values = torch.FloatTensor(events[:, p_index])
    return torch.sparse.FloatTensor(
        indices, values, torch.Size([max_time, *sensor_size])
    )
