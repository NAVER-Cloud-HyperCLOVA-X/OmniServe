# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from vllm-project/vllm
# Source: https://github.com/vllm-project/vllm
# License: Apache-2.0

import json
from collections.abc import Sequence
from typing import Union

import re

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .hcx_parser_mixin import HcxStreamingParserFunctionsMixin

logger = init_logger(__name__)

class HcxToolParser(ToolParser, HcxStreamingParserFunctionsMixin):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.tool_call_start_token: str = " -> tool/function_call\n"
        self.tool_call_end_token: str = "<|im_end|>"
        # case 1. tool call is between other contents; case 2. tool call is at the end of the response
        self.tool_call_regex = re.compile(r"-> tool/function_call\n(.*?)<\|im_end\|>|-> tool/function_call\n(.*)]", re.DOTALL)
            
        # for streaming
        self.tool_call_offset = 0
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool: list[str] = []
        self.is_reasoning_ended = False

        # attributes for streaming parser mixin
        self.buffer_string = ''
        self.special_strings = ['<|im_end|>\n', '<|im_start|>assistant', '-> tool/function_call\n']
        self.escaped_special_strings = [re.escape(ss) for ss in self.special_strings]


    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token in model_output:
            try:
                tool_call_match = self.tool_call_regex.search(model_output)
                if tool_call_match:
                    if tool_call_match.group(1) is not None:
                        raw_function_calls = json.loads(tool_call_match.group(1))
                    else:
                        raw_function_calls = json.loads(tool_call_match.group(2) + ']')

                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(function_call["arguments"],
                                                 ensure_ascii=False)))
                    for function_call in raw_function_calls
                ]
                
                # check if there is other content before tool calls
                if '<|im_end|>\n<|im_start|>assistant -> tool/function_call\n' in model_output:
                    content = model_output.split('<|im_end|>\n<|im_start|>assistant -> tool/function_call\n')[0]

                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=content if content else None)
                else:
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=None)

            except Exception:
                logger.exception("Error in extracting tool call from response.")

                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)
        else:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if self.tool_call_start_token in current_text:
            function_call_text = current_text.split(self.tool_call_start_token)[-1]
            function_call_text = function_call_text[self.tool_call_offset:]
            opening_brace_index = None
            for idx, c in enumerate(function_call_text):
                if c == '{':
                    opening_brace_index = idx
                    break
            
            closing_brace_indices = [_idx for _idx, c in enumerate(function_call_text) if c == '}']
            
            if opening_brace_index is None:
                return None
            
            if len(closing_brace_indices) == 0:
                return None

            for closing_brace_index in closing_brace_indices:
                try:                        
                    _function_call = json.loads(function_call_text[opening_brace_index: closing_brace_index + 1])
                    self.current_tool_id += 1
                    self.tool_call_offset = closing_brace_index
                    self.prev_tool_call_arr.append(_function_call)
                    self.streamed_args_for_tool.append(function_call_text[opening_brace_index:closing_brace_index + 1])

                    return DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                            type="function",
                                            id=f'hcx_tool_call_{self.current_tool_id}',
                                            function=DeltaFunctionCall(
                                                name=_function_call.get('name', ''), 
                                                arguments=json.dumps(_function_call.get('arguments', ''))).model_dump(
                                                    exclude_none=True))])

                except json.JSONDecodeError:
                    logger.debug('Decode error:', function_call_text[opening_brace_index: closing_brace_index + 1])
                
            return None
        else:
            # check if reasoning ended with three conditions
            if len(current_token_ids) == 2 and len(current_text) == 0:
                # there is no reasoning content
                self.is_reasoning_ended = True

            if current_text.startswith(' -> tool/function_call\n'):
                self.is_reasoning_ended = True

            if '<|im_end|>\n<|im_start|>' in current_text:
                self.is_reasoning_ended = True

            # set up buffer for special string processing
            self.buffer_string += delta_text
            buffered_content = ''

            if self.check_is_special_string():
                buffered_content, delta_text = self.remove_special_string()
                self.buffer_string = delta_text

                if self.is_reasoning_ended:
                    return DeltaMessage(content=buffered_content)
                else:
                    return DeltaMessage(reasoning_content=buffered_content)

            if self.check_is_part_of_special_string():
                return None
            else:
                delta_text = self.buffer_string
                self.buffer_string = ''

            if self.is_reasoning_ended:
                return DeltaMessage(content=delta_text)
            else:
                return DeltaMessage(reasoning_content=delta_text)
