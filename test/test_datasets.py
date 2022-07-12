import numpy as np
import tonic.datasets as datasets
import dataset_utils
import os
from utils import create_random_input
from tonic.download_utils import download_url


base_url = "https://www.neuromorphic-vision.com/public/downloads/dataset_samples/"


class ASLDVSTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.ASLDVS
    FEATURE_TYPES = (datasets.ASLDVS.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "ASLDVS/a")
        os.makedirs(testfolder, exist_ok=True)
        filename = "a_0244.mat"
        download_url(url=base_url + filename, root=testfolder, filename=filename)
        return {"n_samples": 1}


class DVSGestureTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSGesture
    FEATURE_TYPES = (datasets.DVSGesture.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "DVSGesture/ibmGestureTrain/user24_led")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(dtype=datasets.DVSGesture.dtype)
        events = np.lib.recfunctions.structured_to_unstructured(events)
        np.save(testfolder + "/0.npy", events)
        np.save(testfolder + "/1.npy", events)
        return {"n_samples": 2}


class DVSGestureTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSGesture
    FEATURE_TYPES = (datasets.DVSGesture.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "DVSGesture/ibmGestureTest/user24_led")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(dtype=datasets.DVSGesture.dtype)
        events = np.lib.recfunctions.structured_to_unstructured(events)
        np.save(testfolder + "/0.npy", events)
        return {"n_samples": 1}


class NCaltech101TestCase(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NCALTECH101
    FEATURE_TYPES = (datasets.NCALTECH101.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NCALTECH101/Caltech101/airplanes/")
        os.makedirs(testfolder, exist_ok=True)
        filename = "image_0006.bin"
        download_url(url=base_url + filename, root=testfolder, filename=filename)
        return {"n_samples": 1}


class NMNISTTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NMNIST/Train/1/")
        os.makedirs(testfolder, exist_ok=True)
        filename = "image_0006.bin"
        download_url(url=base_url + filename, root=testfolder, filename="24901.bin")
        return {"n_samples": 1}


class NMNISTTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NMNIST/Test/1/")
        os.makedirs(testfolder, exist_ok=True)
        filename = "image_0006.bin"
        download_url(url=base_url + filename, root=testfolder, filename="04652.bin")
        return {"n_samples": 1}


class DVSLipTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSLip
    FEATURE_TYPES = (datasets.DVSLip.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "DVSLip/DVS-Lip/train/accused")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(
            dtype=np.dtype([("t", "<i4"), ("x", "i1"), ("y", "i1"), ("p", "i1")])
        )
        np.save(testfolder + "/0.npy", events)
        np.save(testfolder + "/1.npy", events)
        return {"n_samples": 2}


class DVSLipTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.DVSLip
    FEATURE_TYPES = (datasets.DVSLip.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "DVSLip/DVS-Lip/test/accused")
        os.makedirs(testfolder, exist_ok=True)
        events, sensor_size = create_random_input(
            dtype=np.dtype([("t", "<i4"), ("x", "i1"), ("y", "i1"), ("p", "i1")])
        )
        np.save(testfolder + "/0.npy", events)
        return {"n_samples": 1}
