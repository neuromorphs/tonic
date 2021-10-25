How do I wrap my own recordings?
================================
If you have your own recordings on disk and want to make use of Tonic for quick dataloading and applying transformations, then you can wrap them in a custom class.
The easiest option is to make use of a torchvision `DatasetFolder <https://pytorch.org/vision/main/datasets.html#torchvision.datasets.DatasetFolder>`_ class. If that doesn't apply in your case, you can write your own class, where you provide a minimum set of methods ``__init__``, ``__getitem__`` and ``__len__`` and you are good to go. Here is a template class that reads event recordings from a local hdf5 file:
::

  import numpy as np
  import h5py
  from tonic.datasets import Dataset

  class MyRecordings(Dataset):
      sensor_size = (700,) # the sensor size of the event camera or the number of channels of the silicon cochlear that was used
      ordering = "txp" # the order in which your event channels are provided in your recordings

      def __init__(
          self,
          train=True,
          transform=None,
          target_transform=None,
      ):
          super(MyRecordings, self).__init__(
              transform=transform, target_transform=target_transform
          )
          self.train = train

          # replace the strings with your training/testing file locations or pass as an argument
          if train:
              self.filename = "/opt/data1/data/shd_train.h5"
          else:
              self.filename = "/opt/data1/data/shd_test.h5"

      def __getitem__(self, index):
          file = h5py.File(self.filename, "r")
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

      def __len__(self):
          file = h5py.File(self.filename, "r")
          return len(file["labels"])

Depending on the format of your recording files, your implementation might look a bit different. Oftentimes you will have a separate file for each recording. Or you might want to also load some image or IMU data. You can have a look at already existing datasets for some inspiration. :class:`DVSGesture` loads from multiple numpy files, :class:`DAVISDATA` or :class:`VPR` load events and other data from rosbag files, :class:`NCARS` loads eventstream files and :class:`ASLDVS` reads from matlab files.

Afterwards you can call certain samples from the dataset or use a DataLoader wrapper, which will make use of ``__getitem__`` and ``__len__`` functions internally.
::

  dataset = MyRecordings(train=True)
  events, target = dataset[100]

  import torch
  dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
  events, target = next(iter(dataloader))
