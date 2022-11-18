import hashlib
from pathlib import Path
from typing import Optional, Union


def check_sha256(
    fpath: Union[str, Path],
    sha256_provided: str,
    chunk_size: Optional[int] = 1024 * 1024,
) -> str:
    """Function that checks the SHA256 of the archive/dataset. In torchvision they strongly
    recommend to switch to SHA256 from MD5. This function is inspired by
    torchvision.prototype.datasets.utils._resource.

    Parameters:
        fpath: path to the archive/dataset.
        sha256_provided: the SHA256 sum to be checked.
        chunk_size: the file is binary read in chunks to not load it fully to memory. This is the size of each chunk.
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
    return sha256_computed
