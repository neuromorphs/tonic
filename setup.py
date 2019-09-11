import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spike_data_augmentation",
    version="0.0.2",
    author="The Neuromorphs of Telluride",
    author_email="event-data-augmentation@googlegroups.com",
    description="Spike manipulation and augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuromorphs/spike-data-augmentation",
    packages=["spike_data_augmentation"],
    install_requires=["numpy", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
)
