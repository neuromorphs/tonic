import hashlib
from typing import Union, Optional
from pathlib import Path


def check_sha256(
    fpath: Union[str, Path], sha256_provided: str, chunk_size: Optional[int] = 1024 * 1024
) -> None:
    """
    Function that checks the SHA256 of the archive/dataset.
    In torchvision they strongly recommend to switch to SHA256 from MD5.
    This function is inspired by torchvision.prototype.datasets.utils._resource.
    """
    sha256_computed = hashlib.sha256()
    with open(fpath, "rb") as fp:
        chunk_reader = lambda: fp.read(chunk_size)
        for chunk in iter(chunk_reader, b""):
            sha256_computed.update(chunk)
    # Converting to hex format for comparison.
    sha256_computed = sha256_computed.hexdigest()
    if sha256_computed != sha256_provided:
        raise RuntimeError(
            f"The SHA256 provided does not match the actual one. \nComputed: {sha256_computed}.\nProvided: {sha256_provided}."
        )