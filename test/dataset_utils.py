import shutil
import unittest
from typing import Any, Dict, Union
from unittest.mock import patch

import numpy as np


class DatasetTestCase(unittest.TestCase):
    DATASET_CLASS = None
    FEATURE_TYPES = None

    _CHECK_FUNCTIONS = {"check_md5", "check_integrity", "check_exists"}
    _DOWNLOAD_EXTRACT_FUNCTIONS = {
        "download_url",
        "download_file_from_google_drive",
        "extract_archive",
        "download_and_extract_archive",
    }

    def inject_fake_data(
        self, tmpdir: str, config: Dict[str, Any]
    ) -> Union[int, Dict[str, Any]]:
        """Inject fake data for dataset into a temporary directory.

        During the creation of the dataset the download and extract logic is disabled. Thus, the fake data injected
        here needs to resemble the raw data, i.e. the state of the dataset directly after the files are downloaded and
        potentially extracted.
        Args:
            tmpdir (str): Path to a temporary directory. For most cases this acts as root directory for the dataset
                to be created and in turn also for the fake data injected here.
            config (Dict[str, Any]): Configuration that will be passed to the dataset constructor. It provides at least
                fields for all dataset parameters with default values.
        Needs to return one of the following:
            1. (int): Number of examples in the dataset to be created, or
            2. (Dict[str, Any]): Additional information about the injected fake data. Must contain the field
                ``"num_examples"`` that corresponds to the number of examples in the dataset to be created.
        """
        raise NotImplementedError(
            "You need to provide fake data in order for the tests to run."
        )

    def create_dataset(self, inject_fake_data: bool = True, **kwargs: Any):
        tmpdir = "/tmp/"
        info = self._inject_fake_data(tmpdir)

        if inject_fake_data:
            with patch.object(self.DATASET_CLASS, "_check_exists", return_value=True):
                dataset = self.DATASET_CLASS(**self.KWARGS)
        else:
            dataset = self.DATASET_CLASS(**self.KWARGS)
        return dataset, info

    def _inject_fake_data(self, tmpdir):
        info = self.inject_fake_data(tmpdir)
        if info is None:
            raise UsageError(
                "The method 'inject_fake_data' needs to return at least an integer indicating the number of "
                "examples for the current configuration."
            )
        return info

    def _patch_checks(self):
        return {
            patch.object(self.DATASET_CLASS, function, return_value=True)
            for function in self._CHECK_FUNCTIONS
        }

    #     def test_download_started(self):
    #         with pytest.raises((FileNotFoundError, RuntimeError)):
    #             dataset, info = self.create_dataset(inject_fake_data=False)

    def test_feature_types(self):
        dataset, info = self.create_dataset()
        data, target = dataset[0]

        if type(data) != tuple:
            data = (data,)
        if type(target) != tuple:
            target = (target,)
        assert len(data) == len(self.FEATURE_TYPES)
        assert len(target) == len(self.TARGET_TYPES)

        for (data_piece, feature_type) in zip(data, self.FEATURE_TYPES):
            if type(data_piece) == np.ndarray:
                assert data_piece.dtype == feature_type
            else:
                assert type(data_piece) == feature_type

    def test_num_examples(self):
        dataset, info = self.create_dataset()
        assert len(dataset) == info["n_samples"]

    @classmethod
    def setUpClass(cls):
        cls.KWARGS.update({"save_to": "/tmp"})
        shutil.rmtree("/tmp/" + cls.DATASET_CLASS.__name__, ignore_errors=True)
        super().setUpClass()
