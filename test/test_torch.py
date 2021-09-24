import pytest
import numpy as np
import torch
import tonic.functional as F
import utils


class TestFunctionalTorch:
    def findXytpPermutation(self, ordering):
        x_index = ordering.find("x")
        y_index = ordering.find("y")
        t_index = ordering.find("t")
        p_index = ordering.find("p")
        return x_index, y_index, t_index, p_index

    @pytest.mark.parametrize(
        "ordering, merge_polarities",
        [("xytp", True), ("typx", False), ("txp", True), ("xtp", False),],
    )
    def testToSparseTensor(self, ordering, merge_polarities):
        (
            events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input(dtype)
        tensor = F.to_sparse_tensor_pytorch(
            events,
            sensor_size=sensor_size,
            ordering=ordering,
            merge_polarities=merge_polarities,
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_sparse
        assert len(tensor.shape) == len(ordering), (
            "Tensor does not have the right dimensions, should be equivalent to length"
            " of ordering (TCXY) or (TCX)"
        )
        assert (
            events.shape[0] == tensor.coalesce().values().sum()
        ), "Sparse tensor values should contain as many 1s as there are events."
        assert (
            tensor.size()[2:] == sensor_size
        ), "Sparse tensor should have sensor size dimensions."
        assert tensor.size()[0] > 10, (
            "There are probably more than 10 timestamps in the original events. This"
            " error normally occurs when converting float timestamp representations to"
            " integer indices for sparse tensors."
        )
        if merge_polarities:
            assert tensor.shape[1] == 1
        else:
            assert tensor.shape[1] == len(np.unique(events['p'])), (
                "Amount of channels is not equivalent to unique number of polarities in"
                " events"
            )
