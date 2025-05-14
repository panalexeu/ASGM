from typing import TypedDict, Literal, Callable


class Message(TypedDict):
    role: Literal['system', 'developer', 'user', 'assistant']
    content: str


class Tool(TypedDict):
    name: str
    schema: dict
    func: Callable
