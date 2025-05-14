import pytest
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from rich import print

from asgm.models.openai import OpenAIModel, Tool


# ==== Fixtures ====

@pytest.fixture(scope='module')
def model():
    # check that OPENAI_API_KEY is defined in .env
    load_dotenv()
    client = OpenAI()
    return OpenAIModel(client=client, model='gpt-4.1-mini')


@pytest.fixture(scope='module')
def async_model():
    load_dotenv()
    client = AsyncOpenAI()
    return OpenAIModel(client=client, model='gpt-4.1-mini')


@pytest.fixture(scope='module')
def input():
    return [
        {'role': 'user', 'content': 'What is Odessa? How many years have passed since foundation (cur. year 2025)?'}]


@pytest.fixture(scope='module')
def tools():
    return [
        Tool(
            name='web_search',
            func=lambda query: 'Odessa was officially founded on September 2, 1794.',
            schema={
                "type": "function",
                "name": "web_search",
                "description": "Use it to extract information from the web.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up information on the web."
                        }
                    },
                    "required": ["query"],
                    'additionalProperties': False
                },
            }
        ),
        Tool(
            name='calculator',
            func=lambda a, b: a - b,
            schema={
                "type": "function",
                "name": "calculator",
                "description": "Use it to determine how many years have passed since some event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The minuend (the number from which another number is subtracted)."
                        },
                        "b": {
                            "type": "number",
                            "description": "The subtrahend (the number to subtract from 'a')."
                        },
                    },
                    "required": ["a", "b"],
                    'additionalProperties': False
                },
            }
        )
    ]


# ==== Tests ====

def test_openai_model_returns_text_completion(model, input):
    res = model.create_completion(input=input, temperature=0)
    print(res)

    assert 'Odessa' in res


def test_openai_model_returns_structured_completion(model, input):
    class Response(BaseModel):
        headline: str
        text: str

    res = model.create_structured_completion(
        input=input,
        text_format=Response
    )
    print(res)

    assert isinstance(res, dict)


def test_openai_model_returns_tool_completion(model, input, tools):
    res = model.create_tool_completion(
        input=input,
        tools=tools
    )
    print(res)

    # model called web_search tool and calculator tool
    assert len(res) > 0


# ==== Async Tests ====

async def test_async_openai_model_returns_text_completion(async_model, input):
    res = await async_model.acreate_completion(input=input, temperature=0)
    print(res)

    assert 'Odessa' in res


async def test_async_openai_model_returns_structured_completion(async_model, input):
    class Response(BaseModel):
        headline: str
        text: str

    res = await async_model.acreate_structured_completion(
        input=input,
        text_format=Response
    )
    print(res)

    assert isinstance(res, dict)


async def test_async_openai_model_returns_tool_completion(async_model, input, tools):
    res = await async_model.acreate_tool_completion(
        input=input,
        tools=tools
    )
    print(res)

    # model called some tool
    assert len(res) > 0
