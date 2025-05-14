from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .types import Message, Tool


class BaseChatModel(ABC):
    """
    An interface to implement calls to LLMs.
    """

    @abstractmethod
    def create_completion(self, input: list[Message], **kwargs) -> str:
        pass

    @abstractmethod
    async def acreate_completion(self, input: list[Message], **kwargs) -> str:
        pass

    @abstractmethod
    def create_tool_completion(
            self,
            input: list[Message],
            tool: list[Tool],
            **kwargs
    ) -> list[Any]:
        pass

    @abstractmethod
    async def acreate_tool_completion(
            self,
            input: list[Message],
            tool: list[Tool],
            **kwargs
    ) -> list[Any]:
        pass

    @abstractmethod
    def create_structured_completion(
            self,
            input: list[Message],
            text_format: BaseModel,
            **kwargs
    ) -> dict:
        pass

    @abstractmethod
    async def acreate_structured_completion(
            self,
            input: list[Message],
            text_format: BaseModel,
            **kwargs
    ) -> dict:
        pass
