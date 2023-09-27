import shutil
from pathlib import Path

import numpy as np
import prototype_dataset_utils as dataset_utils

import tonic.prototype.datasets as datasets

PATH_TO_TEST_DATA = Path(".", "test", "test_data")

########
# NMNIST
########


def create_nmnist_zip_archive(tmpdir: str, folder: str, filename: str):
    """This function recreates the same structure that we expect in the dataset:

    [Train,Test]----!                 !---0/sample.bin                 !---1/sample.bin
    !---                 !---9/sample.bin
    """
    filename = filename.split(".")[0]  # Stripping the .zip extension.
    folder_path = Path(tmpdir, "NMNIST", folder)
    # We create a folder for each label.
    for i in range(10):
        destination_path = Path(folder_path, str(i))
        destination_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            PATH_TO_TEST_DATA / "sample_nmnist.bin",
            destination_path / "sample_nmnist.bin",
        )
    # We compress the archive.
    shutil.make_archive(
        base_name=Path(tmpdir, "NMNIST", filename),
        format="zip",
        root_dir=folder_path.parent,
        base_dir=folder,
    )
    # We remove the folder previously created.
    shutil.rmtree(folder_path)


class NMNISTTestCase_Train_Uncompressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True, "keep_compressed": False, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TRAIN_FOLDER, datasets.NMNIST._TRAIN_FILENAME
        )


class NMNISTTestCase_Train_Compressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True, "keep_compressed": True, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TRAIN_FOLDER, datasets.NMNIST._TRAIN_FILENAME
        )


class NMNISTTestCase_Train_Uncompressed_FirstSaccade(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True, "keep_compressed": False, "first_saccade_only": True}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TRAIN_FOLDER, datasets.NMNIST._TRAIN_FILENAME
        )


class NMNISTTestCase_Test_Uncompressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": False, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )


class NMNISTTestCase_Test_Compressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": True, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )


class NMNISTTestCase_Test_Uncompressed_FirstSaccade(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": False, "first_saccade_only": True}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )


##########
# STMNIST
##########


def create_stmnist_zip_archive(tmpdir: str, filename: str):
    """This function recreates the same structure that we expect in the dataset:
    data_submission/----!

    !---0/sample.mat !---1/sample.mat !--- !---9/sample.mat
    """
    folder_path = Path(tmpdir, "STMNIST", "data_submission")
    # We create a folder for each label.
    for i in range(10):
        destination_path = Path(folder_path, str(i))
        destination_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            PATH_TO_TEST_DATA / "sample_stmnist.mat",
            destination_path / "sample_stmnist.mat",
        )
    # We compress the archive.
    shutil.make_archive(
        base_name=Path(tmpdir, "STMNIST", filename),
        format="zip",
        base_dir=folder_path,
        root_dir=folder_path,
    )


class STMNISTTestCase_Uncompressed(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.STMNIST
    FEATURE_TYPES = (datasets.STMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"keep_compressed": False}

    def inject_fake_data(self, tmpdir):
        create_stmnist_zip_archive(tmpdir, datasets.STMNIST.__name__)


class STMNISTTestCase_Compressed(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.STMNIST
    FEATURE_TYPES = (datasets.STMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"keep_compressed": True}

    def inject_fake_data(self, tmpdir):
        create_stmnist_zip_archive(tmpdir, datasets.STMNIST.__name__)


#######
# NCARS
#######


def create_ncars_folder(tmpdir, folder_name):
    for label_folder_name in ("background", "cars"):
        test_dir = Path(tmpdir, "NCARS", folder_name, label_folder_name)
        test_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            PATH_TO_TEST_DATA / "sample_ncars.dat",
            test_dir / "sample_ncars.dat",
        )


class NCARSTestCase_Train(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NCARS
    FEATURE_TYPES = (datasets.NCARS._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True}

    def inject_fake_data(self, tmpdir):
        create_ncars_folder(tmpdir, datasets.NCARS._TRAIN_PATH)


class NCARSTestCase_Test(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NCARS
    FEATURE_TYPES = (datasets.NCARS._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False}

    def inject_fake_data(self, tmpdir):
        create_ncars_folder(tmpdir, datasets.NCARS._TEST_PATH)


#############################
# Gen4AutomotiveDetectionMini
#############################


def create_minidataset_folder(tmpdir, split_folder):
    testfolder = Path(
        tmpdir, "Gen4AutomotiveDetectionMini", "mini_dataset", split_folder
    )
    testfolder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PATH_TO_TEST_DATA / "sample_ncars.dat",
        Path(testfolder, "moorea_2019-02-19_004_td_610500000_670500000_td.dat"),
    )
    label = np.array([0.4, 0.2, 0.1, 0.1])
    np.save(testfolder / "moorea_2019-02-19_004_td_610500000_670500000_bbox.npy", label)


class Gen4AutomotiveDetectionMiniTestCase_Train(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen4AutomotiveDetectionMini
    FEATURE_TYPES = (datasets.Gen4AutomotiveDetectionMini._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "train"}

    def inject_fake_data(self, tmpdir):
        create_minidataset_folder(
            tmpdir, datasets.Gen4AutomotiveDetectionMini._TRAIN_FOLDER
        )


class Gen4AutomotiveDetectionMiniTestCase_Valid(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen4AutomotiveDetectionMini
    FEATURE_TYPES = (datasets.Gen4AutomotiveDetectionMini._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "valid"}

    def inject_fake_data(self, tmpdir):
        create_minidataset_folder(
            tmpdir, datasets.Gen4AutomotiveDetectionMini._VALID_FOLDER
        )


class Gen4AutomotiveDetectionMiniTestCase_Test(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen4AutomotiveDetectionMini
    FEATURE_TYPES = (datasets.Gen4AutomotiveDetectionMini._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "test"}

    def inject_fake_data(self, tmpdir):
        create_minidataset_folder(
            tmpdir, datasets.Gen4AutomotiveDetectionMini._TEST_FOLDER
        )


#########################
# Gen1AutomotiveDetection
#########################


def create_gen1dataset_folder(tmpdir, split_folder):
    testfolder = Path(tmpdir, "detection_dataset_duration_60s_ratio_1.0", split_folder)
    testfolder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        PATH_TO_TEST_DATA / "sample_ncars.dat",
        Path(testfolder, "moorea_2019-02-19_004_td_610500000_670500000_td.dat"),
    )
    label = np.array([0.4, 0.2, 0.1, 0.1])
    np.save(testfolder / "moorea_2019-02-19_004_td_610500000_670500000_bbox.npy", label)


class Gen1AutomotiveDetectionTestCase_Train(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen1AutomotiveDetection
    FEATURE_TYPES = (datasets.Gen1AutomotiveDetection._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "train"}

    def inject_fake_data(self, tmpdir):
        create_gen1dataset_folder(
            tmpdir, datasets.Gen1AutomotiveDetection._TRAIN_FOLDER
        )


class Gen1AutomotiveDetectionTestCase_Valid(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen1AutomotiveDetection
    FEATURE_TYPES = (datasets.Gen1AutomotiveDetection._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "valid"}

    def inject_fake_data(self, tmpdir):
        create_gen1dataset_folder(
            tmpdir, datasets.Gen1AutomotiveDetection._VALID_FOLDER
        )


class Gen1AutomotiveDetectionTestCase_Test(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.Gen1AutomotiveDetection
    FEATURE_TYPES = (datasets.Gen1AutomotiveDetection._DTYPE,)
    TARGET_TYPES = (np.array,)
    KWARGS = {"split": "test"}

    def inject_fake_data(self, tmpdir):
        create_gen1dataset_folder(tmpdir, datasets.Gen1AutomotiveDetection._TEST_FOLDER)
