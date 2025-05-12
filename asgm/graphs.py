"""ASGM (Async Star Graph Metric) directly depends on openai package."""
import asyncio

from openai import AsyncOpenAI

from .nodes import (
    AsyncBinaryNode,
    AsyncNonBinaryNode,
    BaseABSNode
)


class AsyncBaseStarGraph:
    def __init__(
            self,
            root_content: str,
            children: list[BaseABSNode],
            client: AsyncOpenAI,
            model: str
    ):
        self.root_content = root_content
        self.children = children
        self.client = client
        self.evaluation: list[BaseABSNode.OutputFormat] | None = None
        self.model = model

    # implement evaluation
    async def eval(self) -> list[BaseABSNode.OutputFormat]:
        """Returns evaluation result over children."""
        self.evaluation = await asyncio.gather(
            *[
                child.eval(
                    content=self.root_content,
                    client=self.client,
                    model=self.model
                )
                for child in self.children
            ]
        )

        return self.evaluation

    # implement scoring
    def score(self, *args, **kwargs) -> float:
        raise NotImplementedError


class AsyncBinaryStarGraph(AsyncBaseStarGraph):
    """
    ABSGraph (Async Binary Star Graph Metric) is a metric where the root node defines the content to be evaluated,
    and the evaluation criteria are represented by a set of independent nodes that return a binary output
    indicating whether each criterion is passed. The resulting graph consists of a root node with a set of leaf nodes
    at depth level 1, which is where the name "Star" comes from.

    This metric allows asynchronous evaluation of all criteria and returns a binary output indicating whether
    the content fits the defined criteria.
    """

    def __init__(
            self,
            root_content: str,
            children: list[AsyncBinaryNode],
            client: AsyncOpenAI,
            model: str
    ):
        super().__init__(root_content=root_content, children=children, client=client, model=model)

    async def eval(self) -> list[AsyncBinaryNode.OutputFormat]:
        return await super().eval()

    @property
    def _evaluation_bool(self) -> list[bool]:
        """Maps ``OutputFormat`` objects to boolean value of ``pass_``."""
        if self.evaluation:
            return [item.pass_ for item in self.evaluation]

        return None

    def binary_score(self) -> bool:
        """
        Returns the binary score of the evaluation.
        Returns ``True`` if all criteria are passed.
        Returns ``False`` if any of the criteria fail.
        """
        return sum(self._evaluation_bool) == len(self.evaluation)

    def score(self, norm: bool = True) -> float:
        """
        Returns the score by counting the number of criteria that passed.
        :param norm: If True, normalizes the score to a value between 0 and 1
        """
        score = sum(self._evaluation_bool)
        if norm:
            return score / len(self.evaluation)

        return score


class AsyncNonBinaryStarGraph(AsyncBaseStarGraph):
    def __init__(
            self,
            root_content: str,
            children: list[AsyncNonBinaryNode],
            client: AsyncOpenAI,
            model: str
    ):
        super().__init__(root_content=root_content, children=children, client=client, model=model)

    async def eval(self) -> list[AsyncNonBinaryNode.OutputFormat]:
        return await super().eval()

    @property
    def _eval_scores(self) -> list[float]:
        return [item.score for item in self.evaluation]

    def score(self, max_score: float | None = None) -> float:
        """
        Return the score by summing ``AsyncNonBinaryNode`` score values.
        :param max_score: If provided returns normalized score, by dividing ``score / max_score``.
        """

        score = sum(self._eval_scores)
        if max_score:
            return score / max_score

        return score
