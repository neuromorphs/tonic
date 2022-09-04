import tonic.prototype.datasets as datasets
import prototype_dataset_utils as dataset_utils
import os
import shutil


########
# NMNIST
########


def create_nmnist_zip_archive(tmpdir: str, folder: str, filename: str):
    filename = filename.split(".")[0]  # Stripping the .zip extension.
    folder_path = os.path.join(tmpdir, folder)
    # We create a folder for each label.
    for i in range(10):
        destination_path = os.path.join(folder_path, str(i))
        os.makedirs(destination_path, exist_ok=True)
        shutil.copyfile(
            "./test/test_data/sample_nmnist.bin",
            os.path.join(destination_path, "sample_nmnist.bin"),
        )
    # We compress the archive.
    shutil.make_archive(
        base_name=os.path.join(tmpdir, filename),
        format="zip",
        base_dir=folder_path,
        root_dir=folder_path,
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
        return {"n_samples": 10}


class NMNISTTestCase_Train_Compressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True, "keep_compressed": True, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TRAIN_FOLDER, datasets.NMNIST._TRAIN_FILENAME
        )
        return {"n_samples": 10}


class NMNISTTestCase_Train_Uncompressed_FirstSaccade(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": True, "keep_compressed": False, "first_saccade_only": True}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TRAIN_FOLDER, datasets.NMNIST._TRAIN_FILENAME
        )
        return {"n_samples": 10}


class NMNISTTestCase_Test_Uncompressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": False, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )
        return {"n_samples": 10}


class NMNISTTestCase_Test_Compressed_AllSaccades(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": True, "first_saccade_only": False}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )
        return {"n_samples": 10}


class NMNISTTestCase_Test_Uncompressed_FirstSaccade(dataset_utils.DatasetTestCase):
    DATASET_CLASS = datasets.NMNIST
    FEATURE_TYPES = (datasets.NMNIST._DTYPE,)
    TARGET_TYPES = (int,)
    KWARGS = {"train": False, "keep_compressed": False, "first_saccade_only": True}

    def inject_fake_data(self, tmpdir):
        create_nmnist_zip_archive(
            tmpdir, datasets.NMNIST._TEST_FOLDER, datasets.NMNIST._TEST_FILENAME
        )
        return {"n_samples": 10}
