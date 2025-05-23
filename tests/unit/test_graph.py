from asgm.nodes import (
    AsyncBinaryNode,
    AsyncNonBinaryNode,
    AsyncNonBinaryToolCallNode
)
from asgm.graphs import (
    AsyncBinaryStarGraph,
    AsyncNonBinaryStarGraph
)
from asgm.models.fake import FakeChatModel
from asgm.models.types import Tool


async def test_async_binary_graph():
    fake_model = FakeChatModel(
        # resembles OutputFormat of AsyncBinaryNode
        pass_=False,
        reason='fake'
    )
    children = [
        AsyncBinaryNode(criterion='fake criterion'),
        AsyncBinaryNode(criterion='fake criterion')
    ]
    binary_graph = AsyncBinaryStarGraph(
        children=children,
        model=fake_model
    )

    res = await binary_graph.eval('fake content')

    assert res == [
        AsyncBinaryNode.OutputFormat(pass_=False, reason='fake'),
        AsyncBinaryNode.OutputFormat(pass_=False, reason='fake')
    ]
    assert binary_graph.binary_score() is False
    assert binary_graph.score() == 0


async def test_async_non_binary_graph():
    fake_model = FakeChatModel(
        # resembles Output format of AsyncNonBinaryNode
        score=0,
        reason='fake'
    )
    children = [
        AsyncNonBinaryNode(criterion='fake criterion', verdicts=['fake verdict', 'fake verdict']),
        AsyncNonBinaryToolCallNode(
            criterion='fake criterion',
            tools=[Tool(name='fake', schema=dict(), func=lambda score, reason: 0)]
        )
    ]
    graph = AsyncNonBinaryStarGraph(
        children=children,
        model=fake_model
    )

    res = await graph.eval('fake content')

    assert res == [
        AsyncNonBinaryNode.OutputFormat(score=0, reason='fake'),
        AsyncNonBinaryNode.OutputFormat(score=0, reason='Tool call'),
    ]
    assert graph.score() == 0


async def test_async_non_binary_weighted_graph():
    fake_model = FakeChatModel(
        # resembles Output format of AsyncNonBinaryNode
        score=1,
        reason='fake'
    )
    # both nodes will output 1. the weighted result for one is 1 * 2 = 2, and for the other, 1 * 3 = 3. The sum is 5.
    children = [
        AsyncNonBinaryNode(
            criterion='fake criterion',
            verdicts=['fake verdict', 'fake verdict'],
            weight=2
        ),
        AsyncNonBinaryToolCallNode(
            criterion='fake criterion',
            tools=[Tool(name='fake', schema=dict(), func=lambda score, reason: 1)],
            weight=3
        )
    ]

    graph = AsyncNonBinaryStarGraph(
        children=children,
        model=fake_model
    )

    res = await graph.eval('fake content')

    assert res == [
        AsyncNonBinaryNode.OutputFormat(score=2, reason='fake'),
        AsyncNonBinaryNode.OutputFormat(score=3, reason='Tool call'),
    ]
    assert graph.score() == 5
