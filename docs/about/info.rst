About this project
==================

This project dates back to the Telluride neuromorphic workshop in 2019. The initial
idea was to use temporal data augmentation to train algorithms that are more robust
when it comes to event camera data.
Over time, the package has evolved to facilitate dataset downloads and the conversion
to different event representations. The goal is to have an easy-to-use interface
to help researchers with their daily endeavours.

Tonic caters to both the event-based world that works directly with events or time
surfaces as well as to more conventional frameworks which might convert events into
dense representations in one way or another. Currently, many such frameworks rely on image
datasets and convert pixel values to spikes to introduce a time dimension. We believe that
event-based datasets are a good match for such frameworks, and that it is not necessary
to convert images to spikes artificially.

For the near to mid-term future, we consider the following things important:

* Provide a well-tested and stable package that other packages can rely on
* Support more of the common benchmarking datasets that contain events and other data.
* Provide an interface to other transformation packages such as PyTorch Vision/Audio.
* Have an extensive documentation to make it easy for newcomers.

That being said, if you have any questions or feedback please don't hesitate to
get in touch! One way to do so is via our `GitHub Discussions page <https://github.com/neuromorphs/tonic/discussions>`_.

The project is currently maintained by `Gregor Lenz <https://lenzgregor.com/site/>`_.
