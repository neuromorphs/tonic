import os
import numpy as np
from .utils import check_integrity, download_and_extract_archive
from .dataset import Dataset
from numpy.lib import recfunctions as rfn


class POKERDVS(Dataset):
    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    filename = "pips_selection.zip"
    url = base_url + filename
    file_md5 = "586FF69997FC4143F607459F2FB89BEE"

    classes = ["cl", "he", "di", "sp"]
    sensor_size = (35, 35)
    ordering = "txyp"

    def __init__(self, save_to, transform=None, download=True):
        super(POKERDVS, self).__init__(save_to, transform=transform)

        counts = dict(zip(self.classes, [0, 0, 0, 0]))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        import loris

        file_path = self.location_on_system + "/pips"
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("dat"):
                    label = file[:2]
                    if counts[label] < 17:
                        counts[label] += 1
                        event_file = loris.read_file(path + "/" + file)
                        events = event_file["events"]
                        events["y"] = 239 - events["y"]
                        events["is_increase"] = events["is_increase"].astype(np.int8)
                        events = rfn.structured_to_unstructured(events)
                        self.data.append(events)
                        self.targets.append(label)

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        return events, target

    def __len__(self):
        return len(self.data)

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
