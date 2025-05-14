from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    An interface to implement calls to LLMs.
    """

    @abstractmethod
    def create_completion(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def acreate_completion(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def create_structured_completion(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    async def acreate_structured_completion(self, *args, **kwargs) -> Any:
        pass

