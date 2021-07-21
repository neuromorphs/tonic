import numpy as np


def get_indices_values(events, sensor_size, ordering, merge_polarities):
    """Sparse Tensor PyTorch representation. See https://pytorch.org/docs/stable/sparse.html for details
    about sparse tensors. A sparse tensor will use the events as indices in the order (tpxy) and values
    of 1 for each index, which signify a spike. The shape of the tensor will be (TCWH).

    Parameters:
        merge_polarities (bool): flag that decides whether to combine polarities into a single channel
                                or split them into separate channels. If True, the number of channels for
                                indices is 1, otherwise it's the number of different polarities. Regardless
                                of this flag, all values assigned to indices will be 1, which signify a spike.

    Returns:
        sparse tensor in TxCxWxH format, where T is timesteps, C is the number of channels for each polarity,
        and W and H are always the size of the sensor.
    """
    assert "x" and "t" and "p" in ordering
    x_index = ordering.find("x")
    t_index = ordering.find("t")
    p_index = ordering.find("p")

    if len(events.shape) != 2:
        raise RuntimeError(
            "Will only convert to sparse tensor from array of shape (N,E)."
        )

    # in any case, all the values in the sparse tensor will be 1, signifying a spike
    values = np.ones(events.shape[0])

    # prevents polarities used as indices that are not 0
    if len(np.unique(events[:, p_index])) == 1:
        merge_polarities = True

    if merge_polarities:  # the indices need to start at 0
        events[:, p_index] = 0
        n_channels = 1
    else:  # avoid any negative indices
        events[events[:, p_index] == -1, p_index] = 0
        n_channels = len(np.unique(events[:, p_index]))

    max_time = int(max(events[:, t_index]) + 1)

    if "y" in ordering:
        y_index = ordering.find("y")
        indices = events[:, [t_index, p_index, x_index, y_index]]
    else:
        indices = events[:, [t_index, p_index, x_index]]

    return indices, values, max_time, n_channels


def to_sparse_tensor_pytorch(events, sensor_size, ordering, merge_polarities):
    try:
        import torch
    except ImportError:
        raise ImportError(
            "The sparse tensor transform needs PyTorch installed. Please install a"
            " stable version "
            + "of PyTorch or alternatively install Tonic with optional PyTorch"
            " dependencies."
        )
    indices, values, max_time, n_channels = get_indices_values(
        events, sensor_size, ordering, merge_polarities
    )
    indices = torch.LongTensor(indices).T
    values = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(
        indices, values, torch.Size([max_time, n_channels, *sensor_size])
    )


def to_sparse_tensor_tensorflow(events, sensor_size, ordering, merge_polarities):
    indices, values, max_time, n_channels = get_indices_values(
        events, sensor_size, ordering, merge_polarities
    )
    try:
        import tensorflow
    except ImportError:
        raise ImportError(
            "The sparse tensor transform needs PyTorch installed. Please install a"
            " stable version "
            + "of PyTorch or alternatively install Tonic with optional PyTorch"
            " dependencies."
        )
    return tensorflow.sparse.SparseTensor(
        indices, values, torch.Size([max_time, n_channels, *sensor_size])
    )
