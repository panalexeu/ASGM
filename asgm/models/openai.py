import json
from typing import Any, Callable, TypedDict

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from openai.types.responses.response_input_param import Message
from openai.types.responses.response_output_item import ResponseFunctionToolCall

from .base_model import BaseChatModel


class Tool(TypedDict):
    name: str
    schema: dict
    func: Callable


class OpenAIModel(BaseChatModel):
    """
    A wrapper around the OpenAI client that implements ``BaseModel``.

    Consider this class an example of how to implement ``BaseModel`` for an LLM provider.
    """

    def __init__(
            self,
            client: OpenAI,
            model: str
    ):
        self.client = client
        self.model = model

    def create_completion(
            self,
            input: list[Message],
            **kwargs
    ) -> str:
        """
        :param input: List of objects of the OpenAI package type ``Message``.
        :param kwargs: Keyword arguments such as ``temperature``, ``top-p``, etc.
        :return:
        """
        return self.client.responses.create(
            input=input,
            model=self.model,
            **kwargs
        ).output[0].content[0].text

    def create_tool_completion(
            self,
            input: list[Message],
            tools: list[Tool],
            **kwargs
    ) -> list[Any]:
        res = self.client.responses.create(
            input=input,
            tools=[tool.schema for tool in tools],
            model=self.model,
            **kwargs
        )

        tool_calls = []
        for output in res.output:

            # if output is a function call
            if isinstance(output, ResponseFunctionToolCall):
                kwargs = json.loads(output.arguments)

                # find relevant tool name and call a tool
                for tool in tools:
                    if tool.name == output.name:
                        tool_calls.append(
                            tool.func(**kwargs)
                        )

        return tool_calls

    def create_structured_completion(
            self,
            input: list[Message],
            text_format: BaseModel,
            **kwargs
    ) -> Any:
        return self.client.responses.parse(
            input=input,
            model=self.model,
            text_format=text_format,
            **kwargs
        ).output_parsed


class AsyncOpenAIModel(BaseChatModel):
    """
    An asynchronous wrapper around the async OpenAI client that implements ``BaseModel``.
    """

    def __init__(
            self,
            client: AsyncOpenAI,
            model: str
    ):
        self.client = client
        self.model = model
