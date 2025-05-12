import json
from typing import Callable

from openai import AsyncOpenAI
from pydantic import BaseModel


class BaseABSNode:
    # implement system prompt
    sys_prompt = None

    # implement model response structure output format
    class OutputFormat(BaseModel):
        pass

    async def eval(self, content: str, client: AsyncOpenAI, model: str) -> OutputFormat:
        raise NotImplementedError


class AsyncBinaryNode(BaseABSNode):
    """
    Defines criteria when output is binary (pass/no pass).
    """

    sys_prompt = """You are a helpful judging assistant.
Evaluate whether the provided content passes the criteria.
Your output is boolean and should be provided in the `pass_` field.
Use True if it passes, False if it does not.
Additionally, provide reasoning behind your answer in the `reason` field."""

    class OutputFormat(BaseModel):
        pass_: bool
        reason: str

    def __init__(self, criteria: str):
        self.criteria = criteria

    async def eval(self, content: str, client: AsyncOpenAI, model: str) -> OutputFormat:
        res = await client.responses.parse(
            input=[
                {'role': 'system', 'content': self.sys_prompt},
                {'role': 'developer', 'content': f'Evaluation criteria: {self.criteria}'},
                {'role': 'user', 'content': content}
            ],
            model=model,
            text_format=self.OutputFormat,
            temperature=0
        )

        return res.output_parsed


class AsyncNonBinaryNode(BaseABSNode):
    """
    Defines criteria when the output is not binary but numeric (some form of scoring).
    Unlike ``AsyncBinaryNode`` in scoring, criteria and verdicts are used.
    Verdicts define what the score should be depending on the criteria results.
    """
    sys_prompt = """You are a helpful judging assistant.
Evaluate which verdict the provided criteria results in.
Your output is numeric and should be provided in the `score field`.
Additionally, provide reasoning behind your answer in the `reason` field."""

    class OutputFormat(BaseModel):
        score: int | float
        reason: str

    def __init__(
            self,
            criteria: str,
            verdicts: list[str]
    ):
        self.criteria = criteria
        self.verdicts = verdicts

    async def eval(self, content: str, client: AsyncOpenAI, model: str) -> OutputFormat:
        res = await client.responses.parse(
            input=[
                {'role': 'system', 'content': self.sys_prompt},
                {'role': 'developer', 'content': f'Evaluation criteria: {self.criteria}'},
                {'role': 'developer', 'content': f'Possible verdicts: {self.verdicts}'},
                {'role': 'user', 'content': content}
            ],
            model=model,
            text_format=self.OutputFormat,
            temperature=0
        )

        return res.output_parsed


class AsyncNonBinaryToolCallNode(BaseABSNode):
    """
    Defines criteria when the output is not binary and numeric score is retrieved by function calling.
    """
    sys_prompt = """You are a helpful judging assistant.
Evaluate whether the provided content passes the criteria determined by function calling."""

    def __init__(
            self,
            criteria,
            tool: Callable,
            tool_schema: dict
    ):
        self.criteria = criteria
        self.tool = tool
        self.tool_schema = tool_schema

    @staticmethod
    def _unpack_tool_call(tool_call) -> dict:
        res = json.loads(tool_call.output[0].arguments)
        return res

    async def eval(self, content: str, client: AsyncOpenAI, model: str) -> AsyncNonBinaryNode.OutputFormat:
        res = await client.responses.create(
            input=[
                {'role': 'system', 'content': self.sys_prompt},
                {'role': 'developer', 'content': self.criteria},
                {'role': 'user', 'content': content}
            ],
            tools=[self.tool_schema],
            model=model,
            temperature=0
        )

        kwargs = self._unpack_tool_call(res)
        score = self.tool(**kwargs)

        return AsyncNonBinaryNode.OutputFormat(
            score=score,
            reason='Tool call'
        )
