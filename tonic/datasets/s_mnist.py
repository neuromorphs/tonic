import os
import numpy as np
from struct import unpack
from tonic.dataset import Dataset
from tonic.download_utils import download_and_extract_archive, extract_archive


class SMNIST(Dataset):
    """Spiking sequential MNIST
    Sequential MNIST (sMNIST) is a standard benchmark task for time series
    classification where each input consists of sequences of 784 pixel
    values created by unrolling the MNIST digits, pixel by pixel. In this
    spiking version, each of the 99 input neurons is associated with a
    particular threshold for the grey value, and this input neuron fires
    whenever the grey value crosses its threshold in the transition from
    the previous to the current pixel.

    Parameters:
        save_to (string):                       Location to save files to on disk.
        train (bool):                           If True, uses training subset,
                                                otherwise testing subset.
        duplicate (bool):                       If True, emits two spikes
                                                per threshold crossing
        num_neurons (integer):                  How many neurons to use to encode
                                                thresholds(must be odd)
        dt (float):                             Duration(in microseconds)
                                                of each timestep
        download (bool):                        Choose to download data or
                                                verify existing files. If True
                                                and a file with the same name
                                                and correct hash is already
                                                in the directory, download is
                                                automatically skipped.
        transform (callable, optional):         A callable of transforms
                                                to apply to the data.
        target_transform (callable, optional):  A callable of transforms to
                                                apply to the targets/labels.

     Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    train_images_file = "train-images-idx3-ubyte"
    train_labels_file = "train-labels-idx1-ubyte"
    test_images_file = "t10k-images-idx3-ubyte"
    test_labels_file = "t10k-labels-idx1-ubyte"
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self,
        save_to,
        train=True,
        duplicate=True,
        num_neurons=99,
        dt=1000.0,
        download=True,
        transform=None,
        target_transform=None,
    ):
        super(SMNIST, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.location_on_system = os.path.join(save_to, "smnist")
        self.train = train
        self.duplicate = duplicate
        self.sensor_size = (num_neurons, 1, 1)
        self.dt = dt

        if (num_neurons % 2) == 0:
            raise Exception("Number of neurons must be odd")

        self.images_file = self.train_images_file if train else self.test_images_file
        self.labels_file = self.train_labels_file if train else self.test_labels_file

        if download:
            self.download()

        # Open images file
        with open(os.path.join(self.location_on_system, self.images_file), "rb") as f:
            image_data = f.read()

            # Unpack header from first 16 bytes of buffer and verify
            magic, num_items, num_rows, num_cols = unpack(">IIII", image_data[:16])
            assert magic == 2051
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            self.image_data = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual (flattened) images
            self.image_data = np.reshape(self.image_data, (num_items, 28 * 28))

        # Open labels file
        with open(os.path.join(self.location_on_system, self.labels_file), "rb") as f:
            label_data = f.read()

            # Unpack header from first 8 bytes of buffer and verify
            magic, num_items = unpack(">II", label_data[:8])
            assert magic == 2049

            # Convert remainder of buffer to numpy bytes
            self.label_data = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert self.label_data.shape == (self.image_data.shape[0],)

    def __getitem__(self, index):
        image = self.image_data[index]

        # Determine how many neurons should encode onset and offset
        half_size = self.sensor_size[0] // 2

        # Determine thresholds of each neuron
        thresholds = np.linspace(0.0, 254.0, half_size).astype(np.uint8)

        # Determine for which pixels each neuron is above or below its threshol
        lower = image[:, None] < thresholds[None, :]
        higher = image[:, None] >= thresholds[None, :]

        # Get onsets and offset (transitions between lower and higher) spike times and ids
        on_spike_time, on_spike_idx = np.where(np.logical_and(lower[:-1], higher[1:]))
        off_spike_time, off_spike_idx = np.where(np.logical_and(higher[:-1], lower[1:]))
        off_spike_idx += half_size

        # Get times when image is 255 and create matching neuron if
        touch_spike_time = np.where(image == 255)[0]
        touch_spike_idx = (
            np.ones(touch_spike_time.shape, dtype=np.int64) * self.sensor_size[0]
        )

        # Combine all spike times and ids together
        spike_time = np.concatenate((on_spike_time, off_spike_time, touch_spike_time))
        spike_idx = np.concatenate((on_spike_idx, off_spike_idx, touch_spike_idx))
        spike_idx = self.sensor_size[0] - spike_idx

        # Sort, first by spike time and then by spike index
        spike_order = np.lexsort((spike_idx, spike_time))
        spike_time = spike_time[spike_order]
        spike_idx = spike_idx[spike_order]

        # If we should duplicate each spike
        if self.duplicate:
            # Repeat spike indices
            spike_idx = np.repeat(spike_idx, 2)

            # Double spike times
            double_spike_time = spike_time * 2

            # Interleave
            spike_time = np.empty(2 * double_spike_time.shape[0], dtype=np.int64)
            spike_time[0::2] = double_spike_time
            spike_time[1::2] = double_spike_time + 1

        # stack and add artificial polarity of 1
        events = np.column_stack(
            (spike_time * self.dt, spike_idx, np.ones(spike_idx.shape[0]))
        )
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        target = self.label_data[index]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return self.image_data.shape[0]

    def download(self):
        for f in [self.images_file, self.labels_file]:
            download_and_extract_archive(
                self.base_url + f + ".gz", self.location_on_system, filename=f + ".gz"
            )
