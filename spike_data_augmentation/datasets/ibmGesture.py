import os
import os.path
import numpy as np
from .dataset import Dataset
from .utils import check_integrity, download_and_extract_archive


class IBMGesture(Dataset):
    """IBMGesture <http://research.ibm.com/dvsgesture/> data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transform: list of transforms to apply to the data
        download: choose to download data or not
    """

    # Train: https://1drv.ms/u/s!Asr9bJkBENkkgalxyTOcGKSBWG3rOQ?e=dDs9Jb
    # Test : https://1drv.ms/u/s!Asr9bJkBENkkgalwF9rrPL_P6t47ng?e=QOAI67
    # Train: https://rktdwg.bn.files.1drv.com/y4mr62ZioKqrwsz5Qlp_hPpfW_nZaveyfaxzCi9CQpP3TKpjnGmBTGkJHzhZnxzGslAT_EHdpfMjxN3krXYVhWdqalrH_wGmKSZMiPKGJc2lsz68P71IKTMGn42ZfvT6qf1g_M1dazYVUjfDJ6BD5LefVCcF90vxcsHIqQK0JOonqiHCdBFIBW7f5_2qzcvu5nn/ibmGestureTrain.tar.gz?download&psid=1
    # Test : https://rktcwg.bn.files.1drv.com/y4mc9Bm7TkqVlFPO_iOErKQl7LJV-CqfcXgxsATXnZLpyMOfxZrEB5RmuDWgkKgYrO8gcoulEL6RxNHgCUOXlxNyI7A1R4pSZbS0G9NPOxbfw50ZgrMVaw91jFfn6Ruh0jJ_fQ_CVPnk776dyhnfCEJ4Z0KDYAJ2JgmaUviHng-inpnz4BvRa4dWrTxMKeMCsl3/ibmGestureTest.tar.gz?download&psid=1
    base_url = ""
    test_zip = "https://rktcwg.bn.files.1drv.com/y4mc9Bm7TkqVlFPO_iOErKQl7LJV-CqfcXgxsATXnZLpyMOfxZrEB5RmuDWgkKgYrO8gcoulEL6RxNHgCUOXlxNyI7A1R4pSZbS0G9NPOxbfw50ZgrMVaw91jFfn6Ruh0jJ_fQ_CVPnk776dyhnfCEJ4Z0KDYAJ2JgmaUviHng-inpnz4BvRa4dWrTxMKeMCsl3/ibmGestureTest.tar.gz?download&psid=1"
    train_zip = "https://rktdwg.bn.files.1drv.com/y4mr62ZioKqrwsz5Qlp_hPpfW_nZaveyfaxzCi9CQpP3TKpjnGmBTGkJHzhZnxzGslAT_EHdpfMjxN3krXYVhWdqalrH_wGmKSZMiPKGJc2lsz68P71IKTMGn42ZfvT6qf1g_M1dazYVUjfDJ6BD5LefVCcF90vxcsHIqQK0JOonqiHCdBFIBW7f5_2qzcvu5nn/ibmGestureTrain.tar.gz?download&psid=1"
    test_md5 = "56070E45DADAA85FFF82E0FBFBC06DE5"
    train_md5 = "3A8F0D4120A166BAC7591F77409CB105"
    test_filename = "ibmGestureTest.tar.gz"
    train_filename = "ibmGestureTrain.tar.gz"
    classes = [
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    ]

    sensor_size = (128, 128)
    ordering = "xypt"

    def __init__(self, save_to, train=True, transform=None, download=False):
        super(IBMGesture, self).__init__(save_to, transform=transform)
        # We will not be loading everything into memory. Instead, we will keep a list of samples into file
        # Could have reused self.data for that purpose as well.
        self.samples = []

        self.train = train
        self.location_on_system = save_to

        if train:
            self.url = self.train_zip
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "ibmGestureTrain"
        else:
            self.url = self.test_zip
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "ibmGestureTest"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file_path = self.location_on_system + "/" + self.folder_name
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.samples.append(path + "/" + file)
                    self.targets.append(int(file[:-4]))

    def __getitem__(self, index):
        event = np.load(self.samples[index])
        target = self.targets[index]
        if self.transform is not None:
            event = self.transform(event, self.sensor_size, self.ordering)
        return event, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )

    def _check_integrity(self):
        root = self.location_on_system
        fpath = os.path.join(root, self.filename)

        if not check_integrity(fpath, self.file_md5):
            return False
        return True
