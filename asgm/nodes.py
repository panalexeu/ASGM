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
    Defines criterion when output is binary (pass/no pass).
    """

    sys_prompt = """You are a helpful judging assistant.
Evaluate whether the provided content passes the criterion.
Your output is boolean and should be provided in the `pass_` field.
Use True if it passes, False if it does not.
Additionally, provide reasoning behind your answer in the `reason` field."""

    class OutputFormat(BaseModel):
        pass_: bool
        reason: str

    def __init__(self, criterion: str):
        self.criterion = criterion

    async def eval(self, content: str, model: BaseChatModel) -> OutputFormat:
        res = await model.acreate_structured_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=f'Evaluation critieria: {self.criterion}'),
                Message(role='user', content=content)
            ],
            text_format=self.OutputFormat,
            temperature=0
        )

        if not res:
            return self.OutputFormat(
                pass_=False,
                reason='Unable to parse model response.'
            )

        return res


class AsyncNonBinaryNode(BaseABSNode):
    """
    Defines criterion when the output is not binary but numeric (some form of scoring).
    Unlike ``AsyncBinaryNode`` in scoring, criterion and verdicts are used.
    Verdicts define what the score should be depending on the criterion results.
    """
    sys_prompt = """You are a helpful judging assistant.
Evaluate which verdict the provided criterion results in.
Your output is numeric and should be provided in the `score field`.
Additionally, provide reasoning behind your answer in the `reason` field."""

    class OutputFormat(BaseModel):
        score: int | float
        reason: str

    def __init__(
            self,
            criterion: str,
            verdicts: list[str],
            weight: float = 1
    ):
        self.criterion = criterion
        self.verdicts = verdicts
        self.weight = weight

    async def eval(self, content: str, model: BaseChatModel) -> OutputFormat:
        res = await model.acreate_structured_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=f'Evaluation criterion: {self.criterion}'),
                Message(role='developer', content=f'Possible verdicts: {self.verdicts}'),
                Message(role='user', content=content)
            ],
            text_format=self.OutputFormat,
            temperature=0
        )

        if not res:
            return self.OutputFormat(
                score=0,
                reason='Unable to parse model response.'
            )

        # apply weight to the score
        res.score *= self.weight
        return res


class AsyncNonBinaryToolCallNode(BaseABSNode):
    """
    Defines criterion when the output is not binary and numeric score is retrieved by function calling.

    A possible criterion could be a description for the model specifying the cases in which the tool should be called.
    """
    sys_prompt = """You are a helpful judging assistant.
Evaluate whether the provided content passes the criterion determined by function calling."""

    def __init__(
            self,
            criterion,
            tools: list[Tool],
            weight: float = 1
    ):
        self.criterion = criterion
        self.tools = tools
        self.weight = weight

    async def eval(self, content: str, model: BaseChatModel) -> AsyncNonBinaryNode.OutputFormat:
        res = await model.acreate_tool_completion(
            input=[
                Message(role='system', content=self.sys_prompt),
                Message(role='developer', content=self.criterion),
                Message(role='user', content=content),
            ],
            tools=self.tools,
            temperature=0
        )

        if not res:
            return AsyncNonBinaryNode.OutputFormat(
                score=0,
                reason='Unable to parse model response.'
            )

        return AsyncNonBinaryNode.OutputFormat(
            score=res[0] * self.weight,  # for now, retrieve the first entry in tool calls results
            reason='Tool call'
        )
