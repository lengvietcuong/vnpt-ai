from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict



class QuestionType(Enum):
    MATH = "math"
    READING = "reading"
    KNOWLEDGE = "knowledge"


class Question(TypedDict):
    qid: str
    question: str
    choices: List[str]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]


ContentPart = TextContent | ImageContentPart


class FunctionDescription(TypedDict, total=False):
    description: Optional[str]
    name: str
    parameters: Dict[str, Any]


class Tool(TypedDict):
    type: Literal["function"]
    function: FunctionDescription


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: str | List[ContentPart] | None
    name: str
    tool_calls: List[ToolCall]
    tool_call_id: str


@dataclass
class APICredentials:
    vnpt_ai_authorization_token: str
    llm_large_token_id: str
    llm_large_token_key: str
    llm_small_token_id: str
    llm_small_token_key: str
    embedding_token_id: str
    embedding_token_key: str


@dataclass
class LLMSettings:
    llm_size: Literal["small", "large"]
    system_prompt: str | None = None
    tools: List[Tool] = field(default_factory=list)
    prefill: str | None = None