import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.io as scio

from tonic.dataset import Dataset
from tonic.io import events_struct, make_structured_array


class EBSSA(Dataset):
    """`EBSSA <https://www.westernsydney.edu.au/icns/resources/reproducible_research3/publication_s
    upport_materials2/space_imaging>`_

    There are six different splits provided in this dataset. The labelled section of the dataset contains 84 recordings and 84 label files. The unlabelled section of the dataset contains 153 recordings in folders marked "Unlabelled".

    ::

        @article{afshar2020event,
            title={Event-based object detection and tracking for space situational awareness},
            author={Afshar, Saeed and Nicholson, Andrew Peter and Van Schaik, Andre and Cohen, Gregory},
            journal={IEEE Sensors Journal},
            volume={20},
            number={24},
            pages={15117--15132},
            year={2020},
            publisher={IEEE}
        }


    Parameters:
        save_to (string): Location to save files to on disk.
        split (string): Which split to load. One of "labelled", "unlabelled", "all".
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    # These are Google drive file IDs found by right clicking on the file and selecting "Get shareable link"
    file_ids = {
        "labelled": [
            "1_GRmzrCvdMXbzlb64Jh2i_4r7WCOvssu",
            "16iGEqKCI2cWEsFuQlJ_VCloIMPpe5ZgB",
            "1PK13T4ACwKMXZOP3QF3LN-EL-TGqVlXR",
        ],
        "unlabelled": [],
    }

    folder_name = ""
    sensor_size = (240, 180, 2)
    dtype = events_struct
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.split = split
        if split != "labelled":
            raise NotImplementedError("Only labelled split is supported at the moment")

        if not self._check_exists():
            self.download()

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    self.data.append(path + "/" + file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        data = scio.loadmat(self.data[index])
        td = data["TD"]

        t = td["ts"][0][0].flatten()
        x = td["x"][0][0].flatten() - 1
        y = td["y"][0][0].flatten() - 1
        p = td["p"][0][0].flatten()
        events = make_structured_array(x, y, t, p)
        start_ts = float(events["t"][0])
        events["t"] = events["t"] - start_ts

        obj = data["Obj"]
        annotation_time_window = 10_000
        bb_width = 10

        obj_t = obj["ts"][0][0].flatten() - start_ts
        obj_t = (obj_t / annotation_time_window).round() * annotation_time_window
        x_min = obj["x"][0][0].flatten() - 1 - bb_width / 2
        x_min = x_min.clip(min=0)
        y_min = obj["y"][0][0].flatten() - 1 - bb_width / 2
        y_min = y_min.clip(min=0)
        obj_id = obj["id"][0][0].flatten()

        x_max = x_min + bb_width
        x_max = x_max.clip(max=239)
        y_max = y_min + bb_width
        y_max = y_max.clip(max=179)

        dtype = np.dtype(
            [
                ("t", np.int64),
                ("x_min", float),
                ("y_min", float),
                ("x_max", float),
                ("y_max", float),
                ("id", np.uint8),
            ]
        )
        target = make_structured_array(
            obj_t, x_min, y_min, x_max, y_max, obj_id, dtype=dtype
        )

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._folder_contains_at_least_n_files_of_type(
            len(self.file_ids[self.split]), ".mat"
        )

    def download(self):
        import gdown

        os.makedirs(self.location_on_system, exist_ok=True)
        for id in self.file_ids[self.split]:
            if not os.path.exists(os.path.join(self.location_on_system, id)):
                gdown.download(
                    id=id,
                    output=os.path.join(self.location_on_system, id) + ".mat",
                    quiet=False,
                )
