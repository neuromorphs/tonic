Prototype Tonic v2
==================

Lately torchvision is making efforts to change its datasets to a new format of iterable datasets. That means that the size of a dataset is not necessarily fixed upfront and samples can be streamed individually. As the sizes of datasets grow, it becomes infeasible to keep all samples in memory, sometimes even the list of file paths to all samples. This also quickly is becoming a problem for event-based data, as with increasing spatial sizes of event cameras the amount of events per second increases drastically.

To address this issue, PyTorch created a new package `torchdata <https://pytorch.org/data/main/index.html>`_ which focuses on separating transformation logic (such as augementations) from loading functionality (batching / multithreading). Tonic will also take advantage of these newest developments and will preferrably implement new datasets in the new iterable style. 