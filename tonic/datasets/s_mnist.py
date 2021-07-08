import os
import numpy as np
from struct import unpack
from .dataset import Dataset
from .download_utils import (
    download_and_extract_archive,
    extract_archive,
)

class SMNIST(Dataset):
    """Spiking sequential MNIST"""

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    train_images_file = "train-images-idx3-ubyte"
    train_labels_file = "train-labels-idx1-ubyte"
    test_images_file = "t10k-images-idx3-ubyte"
    test_labels_file = "t10k-labels-idx1-ubyte"
    sensor_size = (100,)
    ordering = "txp"
    
    def __init__(self, save_to, train=True, download=True, transform=None, target_transform=None):
        super(SMNIST, self).__init__(save_to, transform=transform, 
                                     target_transform=target_transform)
        self.location_on_system = save_to
        self.train = train

        self.images_file = self.train_images_file if train else self.test_images_file
        self.labels_file = self.train_labels_file if train else self.test_labels_file
            
        if download:
            self.download()
        
        # Open images file
        with open(self.images_file, "rb") as f:
            image_data = f.read()
            
            # Unpack header from first 16 bytes of buffer and verify
            magic, num_items, num_rows, num_cols = unpack('>IIII', 
                                                          image_data[:16])
            assert magic == 2051
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            self.image_data = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual (flattened) images
            self.image_data = np.reshape(self.image_data, 
                                         (num_items, 28 * 28))
        
        # Open labels file
        with open(self.labels_file, "rb") as f:
            label_data = f.read()

            # Unpack header from first 8 bytes of buffer and verify
            magic, num_items = unpack('>II', label_data[:8])
            assert magic == 2049

            # Convert remainder of buffer to numpy bytes
            self.label_data = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert self.label_data.shape == (self.image_data.shape[0],)
            
    def __getitem__(self, index):
        image = self.image_data[index]
        
        # Determine how many neurons should encode onset and offset
        max_neuron = self.sensor_size[0] - 1
        mirrored_size = max_neuron // 2
        
        # Determine thresholds of each neuron
        thresholds = np.linspace(0., 254., mirrored_size).astype(np.uint8)
        
        # Determine for which pixels each neuron is above or below its threshol
        lower = image[:, None] < thresholds[None, :]
        higher = image[:, None] >= thresholds[None, :]
        
        # Get onsets and offset (transitions between lower and higher) spike times and ids
        on_spike_time, on_spike_idx = np.where(np.logical_and(lower[:-1], higher[1:]))
        off_spike_time, off_spike_idx = np.where(np.logical_and(higher[:-1], lower[1:]))
        off_spike_idx += mirrored_size
        
        # Get times when image is 255 and create matching neuron if
        touch_spike_time = np.where(image == 255)[0]
        touch_spike_idx = np.ones(touch_spike_time.shape, dtype=np.int64) * max_neuron

        # Combine all spike times and ids together
        spike_time = np.concatenate((on_spike_time, off_spike_time, touch_spike_time))
        spike_idx = np.concatenate((on_spike_idx, off_spike_idx, touch_spike_idx))
        
        # Sort, first by spike time and then by spike index
        spike_order = np.lexsort((spike_idx, spike_time))
        spike_time = spike_time[spike_order]
        spike_idx = spike_idx[spike_order]
        
        return spike_idx, spike_time
        """"// If we should be presenting the image\n"
        "if(timestep < (28 * 28 * 2)) {\n"
        "   const int mirroredTimestep = timestep / 2;\n"
            "if($(id) == 98) {\n"
        "       spike = (imgData[mirroredTimestep] == 255);\n"
        "   }\n"
        "   else if($(id) < 98 && mirroredTimestep < ((28 * 28) - 1)){\n"
        "       const int threshold = (int)((float)($(id) % 49) * (254.0 / 48.0));\n"
        "       // If this is an 'onset' neuron\n"
        "       if($(id) < 49) {\n"
        "           spike = ((imgData[mirroredTimestep] < threshold) && (imgData[mirroredTimestep + 1] >= threshold));\n"
        "       }\n"
        "       // If this is an 'offset' neuron\n"
        "       else {\n"
        "           spike = ((imgData[mirroredTimestep] >= threshold) && (imgData[mirroredTimestep + 1] < threshold));\n"
        "       }\n"
        "   }\n"
        "}\n"
        "// Otherwise, spike if this is the last 'touch' neuron\n"
        "else {\n"
        "   spike = ($(id) == 99);\n"
        "}\n");
 
        # adding artificial polarity of 1
        events = np.vstack((file["spikes/times"][index], file["spikes/units"][index], np.ones(file["spikes/times"][index].shape[0]))).T
        # convert to microseconds
        events[:,0] *= 1e6
        target = file["labels"][index].astype(int)
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target
        """

    def __len__(self):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        return len(file["labels"])

    def download(self):
        for f in [self.images_file, self.labels_file]:
            download_and_extract_archive(
                self.base_url + f + ".gz", self.location_on_system, filename=f + ".gz")
    
    