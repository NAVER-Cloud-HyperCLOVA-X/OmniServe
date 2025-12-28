# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from:
# - vllm-project/vllm: https://github.com/vllm-project/vllm (Apache-2.0)
# - openai/openai-python: https://github.com/openai/openai-python (Apache-2.0)

import time
from dataclasses import field
import uuid
import urllib.parse
from typing import Annotated, Literal, TypeAlias, Any, Union, Optional
from collections import namedtuple

import pydantic
import pydantic_core
from pydantic import BaseModel, Field, ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import Required, TypedDict

from openai.types.chat import (
  ChatCompletionMessageToolCallParam,
  ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
  ChatCompletionAudioParam as OpenAIChatCompletionAudioParam,
  ChatCompletionModality as OpenAIChatCompletionModality,
)
from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation


# _LONG_INFO = torch.iinfo(torch.long)
_LONG_INFO = namedtuple('LongInfo', ['min', 'max'])
_LONG_INFO.min = -9_223_372_036_854_775_808
_LONG_INFO.max = 9_223_372_036_854_775_807


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
  if id_type == "kimi_k2":
    return f"functions.{func_name}:{idx}"
  else:
    # by default return random
    return f"chatcmpl-tool-{random_uuid()}"


class OpenAIBaseModel(BaseModel):
  pass


# NOTE(junhee.yoo): copy from https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/sampling_params.py#L34
#                   exclude methods
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""


# extra="forbid" is a workaround to have kwargs as a field,
# see https://github.com/pydantic/pydantic/issues/3125
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


LogitsProcessors = list[str | LogitsProcessorConstructor]


class FileFile(OpenAIBaseModel):
  file_data: str
  """
  The base64 encoded file data, used when passing the file to the model as a
  string.
  """

  file_id: str
  """The ID of an uploaded file to use as input."""

  filename: str
  """The name of the file, used when passing the file to the model as a string."""


class File(OpenAIBaseModel):
  file: FileFile

  type: Literal["file"]
  """The type of the content part. Always `file`."""


class CustomChatCompletionContentPartTextParam(OpenAIBaseModel):
  text: str
  type: Literal["text"]


class CustomImageURL(TypedDict, total=False):
  url: Required[str]
  detail: Literal["auto", "low", "high"] | None = None


class CustomChatCompletionContentPartImageParam(OpenAIBaseModel):
  type: Literal["image_url"]
  image_url: CustomImageURL

  @pydantic.model_validator(mode="after")
  def validate_url_scheme(self) -> "CustomChatCompletionContentPartImageParam":
    parsed_url = urllib.parse.urlparse(self.image_url["url"])
    if not parsed_url.scheme:
      raise pydantic_core.PydanticCustomError("invalid_url", "Invalid URL format", {"url": self.image_url["url"]})

    return self


class CustomInputAudio(OpenAIBaseModel):
  data: str
  format: Literal["wav", "mp3", "webm", "ogg", "opus"]


class CustomChatCompletionContentInputAudioParam(OpenAIBaseModel):
  type: Literal["input_audio"]
  input_audio: CustomInputAudio


ChatCompletionContentPartParam: TypeAlias = Union[ 
  CustomChatCompletionContentPartTextParam,
  CustomChatCompletionContentPartImageParam,
  CustomChatCompletionContentInputAudioParam,
  File,
]


class CustomChatCompletionMessageParam(TypedDict, total=False):
  """Enables custom roles in the Chat Completion API."""

  role: Required[str]
  """The role of the message's author."""

  content: str | list[ChatCompletionContentPartParam]
  """The contents of the message."""

  name: str
  """An optional name for the participant.

  Provides the model information to differentiate between participants of the
  same role.
  """

  tool_call_id: str | None
  """Tool call that this message is responding to."""

  tool_calls: list[ChatCompletionMessageToolCallParam] | None
  """The tool calls generated by the model, such as function calls."""


class JsonSchemaResponseFormat(OpenAIBaseModel):
  name: str
  description: str | None = None
  # schema is the field in openai but that causes conflicts with pydantic so
  # instead use json_schema with an alias
  json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
  strict: bool | None = None


class ResponseFormat(OpenAIBaseModel):
  # type must be "json_schema", "json_object", or "text"
  type: Literal["text", "json_object", "json_schema"]
  json_schema: JsonSchemaResponseFormat | None = None


ChatCompletionMessageParam: TypeAlias = (
  CustomChatCompletionMessageParam |
  OpenAIChatCompletionMessageParam
)


class LegacyStructuralTag(OpenAIBaseModel):
  begin: str
  # schema is the field, but that causes conflicts with pydantic so
  # instead use structural_tag_schema with an alias
  structural_tag_schema: dict[str, Any] | None = Field(default=None, alias="schema")
  end: str


class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
  type: Literal["structural_tag"]
  structures: list[LegacyStructuralTag]
  triggers: list[str]


class StructuralTagResponseFormat(OpenAIBaseModel):
  type: Literal["structural_tag"]
  format: Any


AnyResponseFormat: TypeAlias = (
  ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)


class StreamOptions(OpenAIBaseModel):
  include_usage: bool | None = True
  continuous_usage_stats: bool | None = False


class FunctionDefinition(OpenAIBaseModel):
  name: str
  description: str | None = None
  parameters: dict[str, Any] | None = None


class ChatCompletionToolsParam(OpenAIBaseModel):
  type: Literal["function"] = "function"
  function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
  name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
  function: ChatCompletionNamedFunction
  type: Literal["function"] = "function"


class ChatCompletionAudioParam(TypedDict, total=False):
  format: Literal["wav", "mp3", "flac", "ogg", "aac", "pcm"]
  voice: Optional[str] = None


class ChatCompletionRequest(OpenAIBaseModel):
  # Ordered by official OpenAI API documentation
  # https://platform.openai.com/docs/api-reference/chat/create
  messages: list[ChatCompletionMessageParam]
  model: str | None = None
  audio: ChatCompletionAudioParam | None = None
  modalities: list[OpenAIChatCompletionModality] | None = None
  frequency_penalty: float | None = 0.0
  logit_bias: dict[str, float] | None = None

  logprobs: bool | None = False
  top_logprobs: int | None = 0
  max_tokens: int | None = Field(
    default=None,
    deprecated="max_tokens is deprecated in favor of "
    "the max_completion_tokens field",
  )
  max_completion_tokens: int | None = None
  n: int | None = 1
  presence_penalty: float | None = 0.0
  response_format: AnyResponseFormat | None = None
  seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
  stop: str | list[str] | None = []
  stream: bool | None = False
  stream_options: StreamOptions | None = None
  temperature: float | None = None
  top_p: float | None = None
  tools: list[ChatCompletionToolsParam] | None = None
  tool_choice: (
    Literal["none"]
    | Literal["auto"]
    | Literal["required"]
    | ChatCompletionNamedToolChoiceParam
    | None
  ) = None  # Changed from "none" to None to allow tool_calls when tools are provided
  reasoning_effort: Literal["low", "medium", "high"] | None = None
  thinking_token_budget: int | None = None
  include_reasoning: bool = True

  # NOTE this will be ignored by vLLM -- the model determines the behavior
  parallel_tool_calls: bool | None = False
  user: str | None = None

  # --8<-- [start:chat-completion-sampling-params]
  best_of: int | None = None
  use_beam_search: bool = False
  top_k: int | None = None
  min_p: float | None = None
  repetition_penalty: float | None = None
  length_penalty: float = 1.0
  stop_token_ids: list[int] | None = []
  include_stop_str_in_output: bool = False
  ignore_eos: bool = False
  min_tokens: int = 0
  skip_special_tokens: bool = True
  spaces_between_special_tokens: bool = True
  truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None
  prompt_logprobs: int | None = None
  allowed_token_ids: list[int] | None = None
  bad_words: list[str] = Field(default_factory=list)
  # --8<-- [end:chat-completion-sampling-params]

  # --8<-- [start:chat-completion-extra-params]
  echo: bool = Field(
      default=False,
      description=(
          "If true, the new message will be prepended with the last message "
          "if they belong to the same role."
      ),
  )
  add_generation_prompt: bool = Field(
      default=True,
      description=(
          "If true, the generation prompt will be added to the chat template. "
          "This is a parameter used by chat template in tokenizer config of the "
          "model."
      ),
  )
  continue_final_message: bool = Field(
      default=False,
      description=(
          "If this is set, the chat will be formatted so that the final "
          "message in the chat is open-ended, without any EOS tokens. The "
          "model will continue this message rather than starting a new one. "
          'This allows you to "prefill" part of the model\'s response for it. '
          "Cannot be used at the same time as `add_generation_prompt`."
      ),
  )
  add_special_tokens: bool = Field(
      default=False,
      description=(
          "If true, special tokens (e.g. BOS) will be added to the prompt "
          "on top of what is added by the chat template. "
          "For most models, the chat template takes care of adding the "
          "special tokens so this should be set to false (as is the "
          "default)."
      ),
  )
  documents: list[dict[str, str]] | None = Field(
      default=None,
      description=(
          "A list of dicts representing documents that will be accessible to "
          "the model if it is performing RAG (retrieval-augmented generation)."
          " If the template does not support RAG, this argument will have no "
          "effect. We recommend that each document should be a dict containing "
          '"title" and "text" keys.'
      ),
  )
  chat_template: str | None = Field(
      default=None,
      description=(
          "A Jinja template to use for this conversion. "
          "As of transformers v4.44, default chat template is no longer "
          "allowed, so you must provide a chat template if the tokenizer "
          "does not define one."
      ),
  )
  chat_template_kwargs: dict[str, Any] | None = Field(
      default=None,
      description=(
          "Additional keyword args to pass to the template renderer. "
          "Will be accessible by the chat template."
      ),
  )
  mm_processor_kwargs: dict[str, Any] | None = Field(
      default=None,
      description=("Additional kwargs to pass to the HF processor."),
  )
  structured_outputs: StructuredOutputsParams | None = Field(
      default=None,
      description="Additional kwargs for structured outputs",
  )
  guided_json: str | dict | BaseModel | None = Field(
      default=None,
      description=(
          "`guided_json` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `json` to `structured_outputs` instead."
      ),
  )
  guided_regex: str | None = Field(
      default=None,
      description=(
          "`guided_regex` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `regex` to `structured_outputs` instead."
      ),
  )
  guided_choice: list[str] | None = Field(
      default=None,
      description=(
          "`guided_choice` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `choice` to `structured_outputs` instead."
      ),
  )
  guided_grammar: str | None = Field(
      default=None,
      description=(
          "`guided_grammar` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `grammar` to `structured_outputs` instead."
      ),
  )
  structural_tag: str | None = Field(
      default=None,
      description=(
          "`structural_tag` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `structural_tag` to `structured_outputs` instead."
      ),
  )
  guided_decoding_backend: str | None = Field(
      default=None,
      description=(
          "`guided_decoding_backend` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please remove it from your request."
      ),
  )
  guided_whitespace_pattern: str | None = Field(
      default=None,
      description=(
          "`guided_whitespace_pattern` is deprecated. "
          "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
          "Please pass `whitespace_pattern` to `structured_outputs` instead."
      ),
  )
  priority: int = Field(
      default=0,
      description=(
          "The priority of the request (lower means earlier handling; "
          "default: 0). Any priority other than 0 will raise an error "
          "if the served model does not use priority scheduling."
      ),
  )
  request_id: str = Field(
      default_factory=lambda: f"{random_uuid()}",
      description=(
          "The request_id related to this request. If the caller does "
          "not set it, a random_uuid will be generated. This id is used "
          "through out the inference process and return in response."
      ),
  )
  logits_processors: LogitsProcessors | None = Field(
      default=None,
      description=(
          "A list of either qualified names of logits processors, or "
          "constructor objects, to apply when sampling. A constructor is "
          "a JSON object with a required 'qualname' field specifying the "
          "qualified name of the processor class/factory, and optional "
          "'args' and 'kwargs' fields containing positional and keyword "
          "arguments. For example: {'qualname': "
          "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
          "{'param': 'value'}}."
      ),
  )
  return_tokens_as_token_ids: bool | None = Field(
      default=None,
      description=(
          "If specified with 'logprobs', tokens are represented "
          " as strings of the form 'token_id:{token_id}' so that tokens "
          "that are not JSON-encodable can be identified."
      ),
  )
  return_token_ids: bool | None = Field(
      default=None,
      description=(
          "If specified, the result will include token IDs alongside the "
          "generated text. In streaming mode, prompt_token_ids is included "
          "only in the first chunk, and token_ids contains the delta tokens "
          "for each chunk. This is useful for debugging or when you "
          "need to map generated text back to input tokens."
      ),
  )
  cache_salt: str | None = Field(
      default=None,
      description=(
          "If specified, the prefix cache will be salted with the provided "
          "string to prevent an attacker to guess prompts in multi-user "
          "environments. The salt should be random, protected from "
          "access by 3rd parties, and long enough to be "
          "unpredictable (e.g., 43 characters base64-encoded, corresponding "
          "to 256 bit). Not supported by vLLM engine V0."
      ),
  )
  kv_transfer_params: dict[str, Any] | None = Field(
      default=None,
      description="KVTransfer parameters used for disaggregated serving.",
  )

  vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
      default=None,
      description=(
          "Additional request parameters with (list of) string or "
          "numeric values, used by custom extensions."
      ),
  )
  # --8<-- [end:chat-completion-extra-params]

  # --8<-- extra argument for wbl
  extra_body: dict[str, Any] | None = Field(
    default=None,
    description="Additional request parameters for vllm. will be flattened into the request body"
  )

  is_ref_audio: bool = Field(
    default=False,
    description="If true, the audio decoding will be forced to use the audio of the last message as the reference audio"
  )

  # NOTE(junhee.yoo): this is used in flatten_extra_body to remove WBL specific fields from the request body
  _need_to_filter_out_field = {
    "extra_body",
    "is_ref_audio",
  }
  # --8<-- end of extra argument for wbl


  @pydantic.model_validator(mode="after")
  def validate_messages(self) -> "ChatCompletionRequest":
    if len(self.messages) == 0:
      raise ValueError("messages must be a non-empty list")
    
    return self


class ChatCompletionLogProb(OpenAIBaseModel):
  token: str
  logprob: float = -9999.0
  bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
  top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
  content: list[ChatCompletionLogProbsContent] | None = None
  refusal: list[ChatCompletionLogProbsContent] | None = None


class FunctionCall(OpenAIBaseModel):
  name: str
  arguments: str


class ToolCall(OpenAIBaseModel):
  id: str = Field(default_factory=make_tool_call_id)
  type: Literal["function"] = "function"
  function: FunctionCall


class ContentPart(OpenAIBaseModel):
  """content_parts의 개별 요소 (vLLM extension)"""
  type: Literal["text", "tool_call_ref", "image_url"]
  text: str | None = None
  tool_call_id: str | None = None
  image_url: dict[str, str] | None = None


class ChatMessage(OpenAIBaseModel):
  role: str
  content: str | None = None
  refusal: str | None = None
  annotations: OpenAIAnnotation | None = None
  audio: OpenAIChatCompletionAudioParam | None = None
  function_call: FunctionCall | None = None
  tool_calls: list[ToolCall] = Field(default_factory=list)
  # vLLM extension: 텍스트와 tool_call의 순서 보존 (내부 처리용, 유저 응답에서는 제거)
  content_parts: list[ContentPart] | None = None

  # vLLM-specific fields that are not in OpenAI spec
  reasoning: str | None = None
  reasoning_content: str | None = None
  """Deprecated: use `reasoning` instead."""

  @pydantic.model_validator(mode="after")
  def handle_deprecated_reasoning_content(self):
    """Copy reasoning to reasoning_content for backward compatibility."""
    self.reasoning_content = self.reasoning
    return self


class ChatCompletionResponseChoice(OpenAIBaseModel):
  index: int
  message: ChatMessage
  logprobs: ChatCompletionLogProbs | None = None
  # per OpenAI spec this is the default
  finish_reason: str | None = "stop"
  # not part of the OpenAI spec but included in vLLM for legacy reasons
  stop_reason: int | str | None = None
  # not part of the OpenAI spec but is useful for tracing the tokens
  # in agent scenarios
  token_ids: list[int] | None = None


class PromptTokenUsageInfo(OpenAIBaseModel):
  cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
  prompt_tokens: int = 0
  total_tokens: int = 0
  completion_tokens: int | None = 0
  prompt_tokens_details: PromptTokenUsageInfo | None = None


class Logprob(OpenAIBaseModel):
  logprob: float
  rank: int | None = None
  decoded_token: str | None = None


class ChatCompletionResponse(OpenAIBaseModel):
  id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
  object: Literal["chat.completion"] = "chat.completion"
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: list[ChatCompletionResponseChoice]
  service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
  system_fingerprint: str | None = None
  usage: UsageInfo


class DeltaFunctionCall(OpenAIBaseModel):
  name: str | None = None
  arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
  id: str | None = None
  type: Literal["function"] | None = None
  index: int
  function: DeltaFunctionCall | None = None


class DeltaMessage(OpenAIBaseModel):
  role: str | None = None
  content: str | None = None
  reasoning: str | None = None
  reasoning_content: str | None = None
  """Deprecated: use `reasoning` instead."""
  tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
  index: int
  delta: DeltaMessage
  logprobs: ChatCompletionLogProbs | None = None
  finish_reason: str | None = None
  stop_reason: int | str | None = None
  # not part of the OpenAI spec but for tracing the tokens
  token_ids: list[int] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
  id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
  object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: list[ChatCompletionResponseStreamChoice]
  usage: UsageInfo | None = Field(default=None)


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


class TokenizeCompletionRequest(OpenAIBaseModel):
  model: str | None = None
  prompt: str

  add_special_tokens: bool = Field(
    default=True,
    description=(
      "If true (the default), special tokens (e.g. BOS) will be added to "
      "the prompt."
    ),
  )
  return_token_strs: bool | None = Field(
    default=False,
    description=(
      "If true, also return the token strings corresponding to the token ids."
    ),
  )


class TokenizeResponse(OpenAIBaseModel):
  count: int
  max_model_len: int
  tokens: list[int]
  token_strs: list[str] | None = None


class TokenizeChatRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    tools: list[ChatCompletionToolsParam] | None = Field(
        default=None,
        description=("A list of tools the model may call."),
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data


TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest
