from pydantic import BaseModel

from .models.base_model import BaseChatModel
from .models.types import Message, Tool


class BaseABSNode:
    # implement system prompt
    sys_prompt = None

    # implement model response structure output format
    class OutputFormat(BaseModel):
        pass

    async def eval(self, content: str, model: BaseChatModel) -> OutputFormat:
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

    async def eval(self, content: str, model: BaseChatModel) -> OutputFormat:
        res = await model.acreate_structured_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=f'Evaluation critieria: {self.criteria}'),
                Message(role='user', content=content)
            ],
            text_format=self.OutputFormat,
            temperature=0
        )

        return res


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

    async def eval(self, content: str, model: BaseChatModel) -> OutputFormat:
        res = await model.acreate_structured_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=f'Evaluation criteria: {self.criteria}'),
                Message(role='developer', content=f'Possible verdicts: {self.verdicts}'),
                Message(role='user', content=content)
            ],
            text_format=self.OutputFormat,
            temperature=0
        )

        return res


class AsyncNonBinaryToolCallNode(BaseABSNode):
    """
    Defines criteria when the output is not binary and numeric score is retrieved by function calling.

    A possible criterion could be a description for the model specifying the cases in which the tool should be called.
    """
    sys_prompt = """You are a helpful judging assistant.
Evaluate whether the provided content passes the criteria determined by function calling."""

    def __init__(
            self,
            criteria,
            tools: list[Tool]
    ):
        self.criteria = criteria
        self.tools = tools

    async def eval(self, content: str, model: BaseChatModel) -> AsyncNonBinaryNode.OutputFormat:
        res = await model.acreate_tool_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=self.criteria),
                Message(role='user', content=content),
            ],
            tools=self.tools,
            temperature=0
        )

        return AsyncNonBinaryNode.OutputFormat(
            score=res[0],  # for now, retrieve the first entry in tool calls results
            reason='Tool call'
        )
