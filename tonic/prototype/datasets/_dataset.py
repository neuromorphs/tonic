import abc
from typing import Any, Collection, Iterator, Optional, Sequence, Union, Dict
import importlib
import pathlib

from torchdata.datapipes.iter import IterDataPipe

class Dataset(IterDataPipe[Dict[str, Any]], abc.ABC):
    def __init__(
            self, 
            root: Union[str, pathlib.Path], 
            *, 
            dependencies: Optional[Collection[str]] = ()
            ) -> None: 
        # Code for importing dependencies. Useful if one wants to 
        # use proprietary data format!
        for dependency in dependencies:
            try: 
                importlib.import_module(dependency)
            except ModuleNotFoundError: 
                raise ModuleNotFoundError(
                        f"{type(self).__name__}() depends on third-party package '{dependency}'", 
                        f"Please, install it."
                        ) from None
        # Getting in root path. 
        self._root = pathlib.Path(root).expanduser().resolve()
        # Resource line, like...?
        resources = None
        # The datapipe. 
        self._dp = self._datapipe()


    def __iter__(self) -> Iterator[Dict[str, Any]]: 
        """
        Iteration method for the data pipe. 
        """
        yield from self._dp

    @abc.abstractmethod
    def _datapipe(self):
        """
        The datapipe line.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        """
        This should return the number of samples in the dataset.
        If available, also the division among train and test.
        """
        pass
