import abc
import importlib
import pathlib
from typing import Any, Callable, Collection, Iterator, Optional, Tuple, Union

import numpy as np
from torchdata.datapipes.iter import IterDataPipe, Mapper

Sample = Tuple[np.ndarray, Any]


class Dataset(IterDataPipe[Sample], abc.ABC):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        keep_compressed: Optional[bool] = False,
        skip_sha256_check: Optional[bool] = True,
        dependencies: Optional[Collection[str]] = (),
    ) -> None:
        # Code for importing dependencies. Useful if one wants to
        # use proprietary data format!
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"{type(self).__name__}() depends on third-party package '{dependency}'",
                    f"Please, install it.",
                ) from None
        # Getting in root path.
        self._root = pathlib.Path(root).expanduser().resolve()
        # SHA256 skipping if the file has been already downloaded.
        self.skip_sha256 = skip_sha256_check
        # Flag to keep the archive compressed.
        self.keep_cmp = keep_compressed
        # Resource line, like...?
        resources = None
        # The datapipe.
        self._dp = self._datapipe_wrapper()

    def __iter__(self) -> Iterator[Sample]:
        """Iteration method for the data pipe."""
        yield from self._dp

    def _datapipe_wrapper(self):
        return self._datapipe()

    @abc.abstractmethod
    def _check_exists(self):
        pass

    @abc.abstractmethod
    def _datapipe(self):
        """The datapipe line."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """This should return the number of samples in the dataset.

        If available, also the division among train and test.
        """
        pass


from typing import Dict, List

# from tonic.prototype.datasets.utils import OnlineResource


class Dataset2(IterDataPipe[Dict[str, Any]], abc.ABC):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        skip_integrity_check: bool = False,
        dependencies: Collection[str] = (),
    ) -> None:
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"{type(self).__name__}() depends on the third-party package '{dependency}'. "
                    f"Please install it, for example with `pip install {dependency}`."
                ) from None

        self._root = pathlib.Path(root).expanduser().resolve()
        resources = [
            resource.load(self._root, skip_integrity_check=skip_integrity_check)
            for resource in self._resources()
        ]
        self._dp = self._datapipe(resources)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._dp

    @abc.abstractmethod
    def _resources(self):
        pass

    @abc.abstractmethod
    def _datapipe(
        self, resource_dps: List[IterDataPipe]
    ) -> IterDataPipe[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass
