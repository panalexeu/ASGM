import json
from typing import Any, Type

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from openai.types.responses.response_output_item import ResponseFunctionToolCall
from openai.types.shared.chat_model import ChatModel

from .base_model import BaseChatModel
from .types import Tool, Message


class OpenAIModel(BaseChatModel):
    """
    A wrapper around the OpenAI client that implements ``BaseModel``.

    Consider this class an example of how to implement ``BaseModel`` for an LLM provider.
    """

    def __init__(
            self,
            client: OpenAI | AsyncOpenAI,
            model: ChatModel,
            timeout: float = 180
    ):
        self.client = client
        self.model = model
        self.timeout = timeout

    # ==== Completions ====

    def create_completion(
            self,
            input: list[Message],
            **kwargs
    ) -> str:
        return self.client.responses.create(
            input=input,
            model=self.model,
            timeout=self.timeout,
            **kwargs
        ).output[0].content[0].text

    async def acreate_completion(
            self,
            input: list[Message],
            **kwargs
    ) -> str:
        res = await self.client.responses.create(
            input=input,
            model=self.model,
            timeout=self.timeout,
            **kwargs
        )

        return res.output[0].content[0].text

    # ==== Tool Completions ====

    def create_tool_completion(
            self,
            input: list[Message],
            tools: list[Tool],
            **kwargs
    ) -> list[Any]:
        res = self.client.responses.create(
            input=input,
            tools=[tool['schema'] for tool in tools],
            model=self.model,
            timeout=self.timeout,
            **kwargs
        )

        tool_calls = []
        for output in res.output:

            # if output is a function call
            if isinstance(output, ResponseFunctionToolCall):
                kwargs = json.loads(output.arguments)

                # find relevant tool name and call a tool
                for tool in tools:
                    if tool['name'] == output.name:
                        tool_calls.append(
                            tool['func'](**kwargs)
                        )

        return tool_calls

    async def acreate_tool_completion(
            self,
            input: list[Message],
            tools: list[Tool],
            **kwargs
    ) -> list[Any]:
        res = await self.client.responses.create(
            input=input,
            tools=[tool['schema'] for tool in tools],
            model=self.model,
            timeout=self.timeout,
            **kwargs
        )

        tool_calls = []
        for output in res.output:

            # if output is a function call
            if isinstance(output, ResponseFunctionToolCall):
                kwargs = json.loads(output.arguments)

                # find relevant tool name and call a tool
                for tool in tools:
                    if tool['name'] == output.name:
                        tool_calls.append(
                            tool['func'](**kwargs)
                        )

        return tool_calls

    # ==== Structured Completions ====

    def create_structured_completion(
            self,
            input: list[Message],
            text_format: Type[BaseModel],
            **kwargs
    ) -> BaseModel | None:
        try:
            return self.client.responses.parse(
                input=input,
                model=self.model,
                text_format=text_format,
                timeout=self.timeout,
                **kwargs
            ).output_parsed
        except:
            return

    async def acreate_structured_completion(
            self,
            input: list[Message],
            text_format: Type[BaseModel],
            **kwargs
    ) -> BaseModel | None:
        try:
            res = await self.client.responses.parse(
                input=input,
                model=self.model,
                text_format=text_format,
                timeout=self.timeout,
                **kwargs
            )

            return res.output_parsed
        except:
            return
