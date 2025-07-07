from typing import Any, Type

from pydantic import BaseModel

from .base_model import BaseChatModel
from .types import Message, Tool


class FakeChatModel(BaseChatModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_completion(self, input: list[Message], **kwargs) -> str:
        return 'fake response'

    async def acreate_completion(self, input: list[Message], **kwargs) -> str:
        return 'fake response'

    def create_tool_completion(self, input: list[Message], tools: list[Tool], **kwargs) -> list[Any]:
        return [tool['func'](**self.kwargs) for tool in tools]

    async def acreate_tool_completion(self, input: list[Message], tools: list[Tool], **kwargs) -> list[Any]:
        return [tool['func'](**self.kwargs) for tool in tools]

    def create_structured_completion(
            self,
            input: list[Message],
            text_format: Type[BaseModel],
            **kwargs
    ) -> BaseModel | None:
        return text_format(**self.kwargs)

    async def acreate_structured_completion(
            self,
            input: list[Message],
            text_format: Type[BaseModel],
            **kwargs
    ) -> BaseModel | None:
        return text_format(**self.kwargs)


