Reading event files
===================

Spike-based datasets are published in numerous data formats, scattered around the web and containing different data such as events, images, inertial measurement unit readings and many more. Here are some helper functions to decode your event files.

.. currentmodule:: tonic.io

.. autosummary::
    :toctree: generated/

    make_structured_array
    read_aedat4
    read_dvs_128
    read_dvs_ibm
    read_dvs_red
    read_dvs_346mini
    read_mnist_file
    read_aedat_header_from_file
    get_aer_events_from_file
    