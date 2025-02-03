import os
import numpy as np
from typing import Any, Callable, Optional, Tuple
from tonic.download_utils import extract_archive
from tonic.io import make_structured_array

class NERDD:
    """`NeRDD <https://github.com/MagriniGabriele/NeRDD>`_
    Neuromorphic Event-based Reactive Driving Dataset.

    Parameters:
        root (string): Location to save files to on disk.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    filename = "NeRDD.zip"
    folder_name = "DATA"
    sensor_size = (1280, 720, 2)
    dtype = np.dtype([("t", np.int32), ("x", np.int16), ("y", np.int16), ("p", bool)])
    ordering = dtype.names

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
    ):
        self.data = []
        self.targets = []
        self.location_on_system = root
        self.transforms = transforms

        if not self._check_extracted():
            if not self._check_zip_exists():
                raise FileNotFoundError(
                    f"The dataset archive '{self.filename}' was not found in {self.location_on_system}. "
                    "Please download it manually and place it in the specified directory."
                )
            self._extract_archive()

        self._load_dataset_structure()

    def _check_zip_exists(self) -> bool:
        """Check if the zip file exists in the specified directory."""
        return os.path.isfile(os.path.join(self.location_on_system, self.filename))

    def _check_extracted(self) -> bool:
        """Check if the dataset has been extracted and contains at least 10 .npz files."""
        data_path = os.path.join(self.location_on_system, self.folder_name)
        if not os.path.isdir(data_path):
            return False
        npz_files = []
        for root, _, files in os.walk(data_path):
            npz_files.extend([f for f in files if f.endswith(".npz")])
        return len(npz_files) == 115

    def _extract_archive(self):
        """Extract the dataset archive."""
        print(f"Extracting {self.filename}...")
        archive_path = os.path.join(self.location_on_system, self.filename)
        extract_archive(archive_path)
        print(f"Extraction complete. Files are now in {self.location_on_system}.")

    def _load_dataset_structure(self):
        """Load the dataset files and their corresponding labels."""
        data_path = os.path.join(self.location_on_system, self.folder_name)
        archives = [
            f for f in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, f))
        ]
        
        for archive in archives:
            archive_path = os.path.join(data_path, archive)
            scenes = [
                f for f in os.listdir(archive_path) 
                if os.path.isdir(os.path.join(archive_path, f))
            ]
            
            for scene in scenes:
                scene_path = os.path.join(archive_path, scene)

                event_file = os.path.join(scene_path, "Event", "output_events.npz")
                label_file = os.path.join(scene_path, "ev_rgb_coordinates.txt")
                
                if os.path.exists(event_file) and os.path.exists(label_file):
                    self.data.append((event_file, label_file, int(archive[8:]), int(scene)))
                else:
                    print(f"Skipping scene {scene} in archive {archive}: missing files.")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        event_file, label_file, archive, scene = self.data[index]
        events = np.load(event_file)["data"]
        events = make_structured_array(
            events[:, 3],  # t
            events[:, 0],  # x
            events[:, 1],  # y
            events[:, 2],  # p
            dtype=self.dtype,
        )

        with open(label_file, "r") as f:
            bboxes = []
            for line in f.readlines():
                timestamp, bbox_str = line.strip().split(":")
                bbox_coords = list(map(float, bbox_str.split(",")))
                bboxes.append([float(timestamp)] + bbox_coords)
            bboxes = np.array(bboxes, dtype=np.float32)

        targets = {
            'archive' : archive,
            'scene' : scene,
            'bboxes' : bboxes,
        }

        # Apply transforms if provided
        if self.transforms is not None:
            events, targets = self.transforms(events, targets)

        return events, targets

    def __len__(self):
        return len(self.data)