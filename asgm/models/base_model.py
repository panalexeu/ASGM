from typing import Any


class BaseChatModel:
    """
    An interface to implement calls to LLMs.
    """

    def create_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def acreate_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def create_tool_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def acreate_tool_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def create_structured_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def acreate_structured_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError
