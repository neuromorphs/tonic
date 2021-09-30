import unittest
import numpy as np
import pytest
import tonic.datasets as datasets


# @pytest.mark.skip(reason="Dataset tests are super slow!")
class TestDatasets:
    download = False

    @pytest.mark.parametrize("n_events, true_label, n_samples", [(15951, 0, 100800)])
    def testASLDVS(self, n_events, true_label, n_samples):
        dataset = datasets.ASLDVS(save_to="./data", download=self.download)
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "recording, n_events, n_images, n_samples",
        [("shapes_6dof", 17962477, 1356, 1),],
    )
    def testDAVISDATA(self, recording, n_events, n_images, n_samples):
        dataset = datasets.DAVISDATA(
            save_to="./data", recording=recording, download=self.download,
        )
        events, imu, images, ground_truth = dataset[0]

        assert events.shape[0] == n_events
        assert images["frames"].shape[0] == n_images
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "train, n_events, true_label, n_samples",
        [(True, 927713, 3, 1077), (False, 944776, 3, 264)],
    )
    def testDVSGesture(self, train, n_events, true_label, n_samples):
        dataset = datasets.DVSGesture(
            save_to="./data", train=train, download=self.download
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "train, n_events, true_label, n_samples",
        [(True, 4278, 11, 8156), (False, 11273, 10, 2264)],
    )
    def testSHD(self, train, n_events, true_label, n_samples):
        dataset = datasets.SHD(save_to="./data", train=train, download=self.download)
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "split, n_events, true_label, n_samples",
        [
            ("train", 12618, 0, 75466),
            ("test", 11671, 0, 20382),
            ("valid", 14591, 0, 9981),
        ],
    )
    def testSSC(self, split, n_events, true_label, n_samples):
        dataset = datasets.SSC(save_to="./data", split=split, download=self.download)
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "scene, n_events_left, n_events_right, n_samples",
        [("indoor_flying", 14071304, 12005558, 4),],
    )
    def testMVSEC(self, scene, n_events_left, n_events_right, n_samples):
        dataset = datasets.MVSEC(save_to="./data", scene=scene, download=self.download)
        data = dataset[0]
        events_left = data[0]
        events_right = data[1]

        assert events_left.shape[0] == n_events_left
        assert events_right.shape[0] == n_events_right
        assert len(dataset) == n_samples
        assert events_left.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "n_events, true_label, n_samples, walk_subset",
        [(915556, 0, 304, True), (227321, 0, 1342, False)],
    )
    def testNavGestures(self, n_events, true_label, n_samples, walk_subset):
        dataset = datasets.NavGesture(
            save_to="./data", walk_subset=walk_subset, download=self.download
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "n_events, true_label, n_samples", [(88007, "BACKGROUND_Google", 8709)]
    )
    def testNCALTECH101(self, n_events, true_label, n_samples):
        dataset = datasets.NCALTECH101(save_to="./data", download=self.download)
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype

    @pytest.mark.parametrize(
        "train, n_events, true_label, n_samples",
        [(True, 1309, 0, 15422), (False, 2308, 0, 8607)],
    )
    def testNCARS(self, train, n_events, true_label, n_samples):
        dataset = datasets.NCARS(save_to="./data", train=train, download=self.download)
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "train, first_saccade_only, n_events, true_label, n_samples",
        [
            (True, False, 5520, 3, 60000),
            (False, False, 6144, 3, 10000),
            (False, True, 2126, 3, 10000),
        ],
    )
    def testNMNIST(self, train, first_saccade_only, n_events, true_label, n_samples):
        dataset = datasets.NMNIST(
            save_to="./data",
            train=train,
            download=self.download,
            first_saccade_only=first_saccade_only,
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "train, n_events, true_label, n_samples",
        [(True, 898, "man-jr-b-6", 8621), (False, 1257, "man-im-b-6", 8697),],
    )
    def testNTIDIGITS(self, train, n_events, true_label, n_samples):
        dataset = datasets.NTIDIGITS(
            save_to="./data", train=train, download=self.download,
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "train, n_events, true_label, n_samples",
        [(True, 4819, 3, 48), (False, 5366, 3, 20)],
    )
    def testPOKERDVS(self, train, n_events, true_label, n_samples):
        dataset = datasets.POKERDVS(
            save_to="./data", train=train, download=self.download
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "n_events, true_label, n_samples", [(4138, 5, 60000),],
    )
    def testSMNIST(self, n_events, true_label, n_samples):
        dataset = datasets.SMNIST(
            save_to="./data", num_neurons=99, download=self.download,
        )
        events, label = dataset[0]

        assert events.shape[0] == n_events
        assert label == true_label
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3

    @pytest.mark.parametrize(
        "n_events, n_images, n_samples", [(536715651, 28915, 6),],
    )
    def testVPR(self, n_events, n_images, n_samples):
        dataset = datasets.VPR(save_to="./data", download=self.download,)
        events, imu, images = dataset[0]

        assert events.shape[0] == n_events
        assert len(images["frames"]) == n_images
        assert len(dataset) == n_samples
        assert events.dtype == dataset.dtype
        assert len(dataset.sensor_size) == 3
