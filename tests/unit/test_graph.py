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
