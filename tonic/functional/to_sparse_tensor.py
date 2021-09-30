import numpy as np


def get_indices_values(events, sensor_size, merge_polarities):
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
    assert "x" and "t" and "p" in events.dtype.names

    # in any case, all the values in the sparse tensor will be 1, signifying a spike
    values = np.ones(events.shape[0], dtype=int)

    # prevents polarities used as indices that are not 0
    if len(np.unique(events["p"])) == 1:
        merge_polarities = True

    if merge_polarities:  # the indices need to start at 0
        events["p"] = 0
        sensor_size[2] = 1
    else:  # avoid any negative indices
        events["p"][events["p"] == -1] = 0

    max_time = int(max(events["t"]) + 1)

    indices = np.column_stack((events["t"], events["p"], events["x"]))

    if "y" in events.dtype.names:
        indices = np.column_stack((indices, events["y"]))

    #     import ipdb; ipdb.set_trace()
    return indices, values, max_time, sensor_size


def to_sparse_tensor_pytorch(events, sensor_size, merge_polarities):
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed. Please choose different backend.")
    indices, values, max_time, sensor_size = get_indices_values(
        events, sensor_size, merge_polarities
    )
    indices = torch.LongTensor(indices).T
    values = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(
        indices, values, torch.Size([max_time, sensor_size[2], *sensor_size[:2]])
    )


def to_sparse_tensor_tensorflow(events, sensor_size, merge_polarities):
    indices, values, max_time, n_channels = get_indices_values(
        events, sensor_size, merge_polarities
    )
    try:
        import tensorflow
    except ImportError:
        raise ImportError("TensorFlow not installed. Please choose different backend.")
    return tensorflow.sparse.SparseTensor(
        indices, values, torch.Size([max_time, n_channels, *sensor_size])
    )
