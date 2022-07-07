import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class DVSLip(Dataset):
    """`DVS-Lip <https://sites.google.com/view/event-based-lipreading>`_
    ::

        @inproceedings{tan2022multi,
            title={Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading},
            author={Tan, Ganchao and Wang, Yang and Han, Han and Cao, Yang and Wu, Feng and Zha, Zheng-Jun},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={20094--20103},
            year={2022}
        }

        Implementation inspired from original script: https://github.com/tgc1997/event-based-lip-reading/blob/main/utils/dataset.py

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view"
    filename = "DVS-Lip.zip"
    base_folder = "DVS-Lip"
    file_md5 = "2dcb959255122d4cdeb6094ca282494b"

    sensor_size = (128, 128, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    classes = [
        "accused",
        "action",
        "allow",
        "allowed",
        "america",
        "american",
        "another",
        "around",
        "attacks",
        "banks",
        "become",
        "being",
        "benefit",
        "benefits",
        "between",
        "billion",
        "called",
        "capital",
        "challenge",
        "change",
        "chief",
        "couple",
        "court",
        "death",
        "described",
        "difference",
        "different",
        "during",
        "economic",
        "education",
        "election",
        "england",
        "evening",
        "everything",
        "exactly",
        "general",
        "germany",
        "giving",
        "ground",
        "happen",
        "happened",
        "having",
        "heavy",
        "house",
        "hundreds",
        "immigration",
        "judge",
        "labour",
        "leaders",
        "legal",
        "little",
        "london",
        "majority",
        "meeting",
        "military",
        "million",
        "minutes",
        "missing",
        "needs",
        "number",
        "numbers",
        "paying",
        "perhaps",
        "point",
        "potential",
        "press",
        "price",
        "question",
        "really",
        "right",
        "russia",
        "russian",
        "saying",
        "security",
        "several",
        "should",
        "significant",
        "spend",
        "spent",
        "started",
        "still",
        "support",
        "syria",
        "syrian",
        "taken",
        "taking",
        "terms",
        "these",
        "thing",
        "think",
        "times",
        "tomorrow",
        "under",
        "warning",
        "water",
        "welcome",
        "words",
        "worst",
        "years",
        "young",
    ]  # 100 labels

    ambiguous_classes = [
        "action",
        "allow",
        "allowed",
        "america",
        "american",
        "around",
        "being",
        "benefit",
        "benefits",
        "billion",
        "called",
        "challenge",
        "change",
        "court",
        "difference",
        "different",
        "election",
        "evening",
        "giving",
        "ground",
        "happen",
        "happened",
        "having",
        "heavy",
        "legal",
        "little",
        "meeting",
        "million",
        "missing",
        "needs",
        "number",
        "numbers",
        "paying",
        "press",
        "price",
        "russia",
        "russian",
        "spend",
        "spent",
        "syria",
        "syrian",
        "taken",
        "taking",
        "terms",
        "these",
        "thing",
        "think",
        "times",
        "words",
        "worst",
    ]  # the 25 word pairs that are ambiguous (see paper)

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(DVSLip, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.url = self.base_url
        self.folder_name = os.path.join(
            self.base_folder, "train" if self.train else "test"
        )

        if not self._check_exists():
            if self._is_file_present():  # check if zip file is manually downloaded
                archive = os.path.join(self.location_on_system, self.filename)
                print(f"Extracting {archive} to {self.location_on_system}...")
                extract_archive(
                    archive,
                    remove_finished=True,
                )
            else:
                print(
                    f"""
                    WARNING: this dataset is available from Google Drive and must be downloaded manually.
                    Please download the zip file ( {self.url} ) and place it in {self.location_on_system}."""
                )
                exit()

        file_path = os.path.join(self.location_on_system, self.folder_name)

        for act_dir in os.listdir(file_path):
            label = self.classes.index(act_dir)

            for file in os.listdir(os.path.join(file_path, act_dir)):
                if file.endswith("npy"):
                    self.targets.append(label)
                    self.data.append(os.path.join(file_path, act_dir, file))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        orig_events = np.load(self.data[index])

        events = np.empty(orig_events.shape, dtype=self.dtype)
        events["x"] = orig_events["x"]
        events["y"] = orig_events["y"]
        events["t"] = orig_events["t"]
        events["p"] = orig_events["p"]

        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.isdir(
            os.path.join(
                self.location_on_system, self.folder_name
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".npy")
