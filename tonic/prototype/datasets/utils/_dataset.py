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
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
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
        # Getting trasforms.
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
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
        dp = self._datapipe()
        if self.transforms:
            dp = Mapper(dp, self.transforms)
        if self.transform:
            dp = Mapper(dp, self.transform, input_col=0)
        if self.target_transform:
            dp = Mapper(dp, self.target_transform, input_col=1)
        return dp

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
