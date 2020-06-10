import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tonic",
    version="0.0.3",
    author="The Neuromorphs of Telluride",
    author_email="event-data-augmentation@googlegroups.com",
    description="Spike manipulation and augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuromorphs/tonic",
    include_package_data=False,
    packages=["tonic"],
    install_requires=["numpy", "loris", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Framework :: Tonic",
    ],
)
