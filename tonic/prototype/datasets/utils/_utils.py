import hashlib
from typing import Union
from pathlib import Path


def check_sha256(
    path: Union[str, Path], sha256_provided: str, chunk_size: int = 1024 * 1024
) -> None:
    """
    Function that checks the SHA256 of the archive/dataset.
    In torchvision they strongly recommend to switch to SHA256 from MD5.
    This function is inspired by torchvision.prototype.datasets.utils._resource.
    """
    sha256_computed = hashlib.sha256()
    chunk_reader = lambda fp: (fp.read(chunk_size), b"")
    with open(path, "rb") as fp:
        for chunk in chunk_reader(fp):
            sha256_computed.update(chunk)
    # Converting to hex format for comparison.
    sha256_computed = sha256_computed.hexdigest()
    if sha256_computed != sha256_provided:
        raise RuntimeError(
            f"The SHA256 provided does not match the actual one. \nComputed: {sha256_computed}.\nProvided: {sha256_provided}."
        )
