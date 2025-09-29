import os
import shutil

import dataset_utils
import h5py
import numpy as np
from utils import create_random_input

import tonic.datasets as datasets


class ASLDVSTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.ASLDVS
    FEATURE_TYPES = (datasets.ASLDVS.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "ASLDVS/a")
        os.makedirs(testfolder, exist_ok=True)
        # Copy from local test data instead of downloading
        source_file = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_asldvs.mat"
        )
        shutil.copy(source_file, os.path.join(testfolder, "a_0244.mat"))
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


class EBSSATestCase(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.EBSSA
    FEATURE_TYPES = (datasets.EBSSA.dtype,)
    TARGET_TYPES = (np.ndarray,)
    KWARGS = {"split": "labelled"}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "EBSSA")
        os.makedirs(testfolder, exist_ok=True)
        # Copy from local test data instead of downloading
        source_file = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_ebssa.h5"
        )
        shutil.copy(source_file, os.path.join(testfolder, "labelled_ebssa.h5"))
        return {"n_samples": 1}


class ThreeET_EyetrackingTestCase(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.ThreeET_Eyetracking
    FEATURE_TYPES = (datasets.ThreeET_Eyetracking.dtype,)
    TARGET_TYPES = (np.ndarray,)
    KWARGS = {"split": "train"}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "ThreeET_Eyetracking")
        os.makedirs(testfolder, exist_ok=True)
        os.makedirs(os.path.join(testfolder, "data"), exist_ok=True)
        os.makedirs(os.path.join(testfolder, "labels"), exist_ok=True)
        # write one line of file name into train_files.txt under testfolder
        os.system("echo testcase > " + os.path.join(testfolder, "train_files.txt"))
        filename = "testcase"

        # Copy from local test data instead of downloading
        source_h5 = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_threeet_eyetracking.h5"
        )
        source_txt = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_threeet_eyetracking.txt"
        )
        shutil.copy(source_h5, os.path.join(testfolder, "data", filename + ".h5"))
        shutil.copy(source_txt, os.path.join(testfolder, "labels", filename + ".txt"))

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
        # Copy from local test data instead of downloading
        source_file = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_ncaltech101.bin"
        )
        shutil.copy(source_file, os.path.join(testfolder, filename))
        return {"n_samples": 1}


class NMNISTTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NMNIST/Train/1/")
        os.makedirs(testfolder, exist_ok=True)
        filename = "00015.bin"
        # Copy from local test data instead of downloading
        source_file = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_nmnist.bin"
        )
        shutil.copy(source_file, os.path.join(testfolder, filename))
        return {"n_samples": 1}


class NMNISTTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NMNIST/Test/1/")
        os.makedirs(testfolder, exist_ok=True)
        filename = "00015.bin"
        # Copy from local test data instead of downloading
        source_file = os.path.join(
            os.path.dirname(__file__), "test_data", "sample_nmnist.bin"
        )
        shutil.copy(source_file, os.path.join(testfolder, filename))
        return {"n_samples": 1}


def create_ntidigits_data(filename, n_samples):
    with h5py.File(filename, mode="w") as write_file:
        # Generate random times and units
        times = np.random.random(size=(n_samples, 100)).astype(np.float16)
        units = np.random.randint(0, 64, size=times.shape, dtype=np.uint16)

        # Generate random sequences of symbols
        symbols = ["o", "1", "2", "3", "4", "5", "6", "7", "8", "9", "z"]
        sequences = np.array(
            [
                "speaker-a-"
                + "".join(
                    np.random.choice(
                        symbols, size=np.random.randint(1, 7), replace=True
                    )
                )
                for _ in range(n_samples)
            ]
        ).astype("|S18")

        for partition in ["train", "test"]:
            # Create a dictionary to store the addresses and timestamps of each speaker
            addresses = {}
            timestamps = {}
            for i in range(n_samples):
                speaker = sequences[i]
                if speaker not in addresses:
                    addresses[speaker] = units[i]
                    timestamps[speaker] = times[i]

            # Create a group for addresses and timestamps, Store each speaker's data as a dataset in the group
            train_address_group = write_file.create_group(
                "{}_addresses".format(partition)
            )
            for speaker, addresses in addresses.items():
                train_address_group.create_dataset(
                    speaker, data=np.array(addresses, dtype=np.uint16)
                )

            train_timestamps_group = write_file.create_group(
                "{}_timestamps".format(partition)
            )
            for speaker, timestamps in timestamps.items():
                train_timestamps_group.create_dataset(
                    speaker, data=np.array(timestamps, dtype=np.float16)
                )

            # Create datasets for labels
            write_file.create_dataset("{}_labels".format(partition), data=sequences)


class NTIDIGITS18TestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NTIDIGITS18
    FEATURE_TYPES = (datasets.NTIDIGITS18.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NTIDIGITS18/")
        os.makedirs(testfolder, exist_ok=True)
        create_ntidigits_data(testfolder + "n-tidigits.hdf5", n_samples=2)
        return {"n_samples": 2}


class NTIDIGITS18TestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NTIDIGITS18
    FEATURE_TYPES = (datasets.NTIDIGITS18.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "NTIDIGITS18/")
        os.makedirs(testfolder, exist_ok=True)
        create_ntidigits_data(testfolder + "n-tidigits.hdf5", n_samples=2)
        return {"n_samples": 2}


def create_hsd_data(filename, n_samples):
    with h5py.File(filename, mode="w") as write_file:
        times = np.random.random(size=(n_samples, 100)).astype(np.float16)
        units = (np.random.random(size=(n_samples, 100)) * 700).astype(np.uint16)
        keys = ["zero", "one"]
        speaker = (np.random.random(size=n_samples) * 20).astype(np.uint16)
        write_file.create_dataset("spikes/units", data=units)
        write_file.create_dataset("spikes/times", data=times)
        write_file.create_dataset("labels", data=[1] * n_samples)
        write_file.create_dataset("extra/keys", data=keys)
        write_file.create_dataset("extra/speaker", data=speaker)


class SHDTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.SHD
    FEATURE_TYPES = (datasets.SHD.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "SHD/")
        os.makedirs(testfolder, exist_ok=True)
        create_hsd_data(testfolder + "shd_train.h5", n_samples=2)
        return {"n_samples": 2}


class SHDTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.SHD
    FEATURE_TYPES = (datasets.SHD.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "SHD/")
        os.makedirs(testfolder, exist_ok=True)
        create_hsd_data(testfolder + "shd_test.h5", n_samples=1)
        return {"n_samples": 1}


class SSCTestCaseTrain(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.SSC
    FEATURE_TYPES = (datasets.SSC.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"split": "train"}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "SSC/")
        os.makedirs(testfolder, exist_ok=True)
        create_hsd_data(testfolder + "ssc_train.h5", n_samples=3)
        return {"n_samples": 3}


class SSCTestCaseValid(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.SSC
    FEATURE_TYPES = (datasets.SSC.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"split": "valid"}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "SSC/")
        os.makedirs(testfolder, exist_ok=True)
        create_hsd_data(testfolder + "ssc_valid.h5", n_samples=4)
        return {"n_samples": 4}


class SSCTestCaseTest(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.SSC
    FEATURE_TYPES = (datasets.SSC.dtype,)
    TARGET_TYPES = (int,)
    KWARGS = {"split": "test"}

    def inject_fake_data(self, tmpdir):
        testfolder = os.path.join(tmpdir, "SSC/")
        os.makedirs(testfolder, exist_ok=True)
        create_hsd_data(testfolder + "ssc_test.h5", n_samples=5)
        return {"n_samples": 5}
