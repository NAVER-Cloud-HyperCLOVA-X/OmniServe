# HyperCLOVAX vLLM Plugin
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import re

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

from .hcx_parser_mixin import HcxStreamingParserFunctionsMixin

logger = init_logger(__name__)


class HcxReasoningParser(ReasoningParser, HcxStreamingParserFunctionsMixin):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "/think\n"
        self.think_end_token = "<|im_end|>\n<|im_start|>assistant"

        # for streaming
        self.end_token_id = self.vocab.get("<|im_end|>")
        self.non_reasoning_mode_start_token = tokenizer.encode("\n")[0]
        self.function_call_role = ' -> tool/function_call\n'
        self.no_reasoning_content = False

        # attributes for streaming parser mixin
        self.buffer_string = ''
        self.special_strings = ['/think\n', '<|im_end|>\n<|im_start|>assistant', self.function_call_role]
        self.escaped_special_strings = [re.escape(ss) for ss in self.special_strings]


    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        chat_template_kwargs = request.chat_template_kwargs or {}
        if chat_template_kwargs.get('skip_reasoning', False):
            return None, model_output

        is_reasoning = False

        if chat_template_kwargs.get('force_reasoning', False):
            is_reasoning = True

        if model_output.startswith(self.think_start_token):
            is_reasoning = True
            model_output_parts = model_output.partition(self.think_start_token)
            model_output = model_output_parts[2]
        
        if self.think_end_token not in model_output:
            if is_reasoning:
                return model_output, None
            else:
                return None, model_output

        reasoning_content, _, content = model_output.partition(self.think_end_token)

        final_content = content or None

        return reasoning_content, final_content


    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        if current_token_ids[0] == self.non_reasoning_mode_start_token:
            self.no_reasoning_content = True
            
        if len(current_text) == 0:
            return None

        if self.no_reasoning_content:
            return DeltaMessage(content=delta_text)

        self.buffer_string += delta_text
        buffered_content = ''

        if self.check_is_special_string():
            if current_text.startswith(self.function_call_role):
                self.no_reasoning_content = True
                delta_text = self.buffer_string
                self.buffer_string = ''
                return DeltaMessage(content=delta_text)

            buffered_content, delta_text = self.remove_special_string()
            self.buffer_string = delta_text
            
            if buffered_content:
                # if buffered_content is not empty, the special string must be '<|im_end|>\n<|im_start|>assistant'
                # which serves as the separator between reasoning content and other content
                if self.check_is_part_of_special_string():
                    return DeltaMessage(reasoning_content=buffered_content)
                else:
                    self.buffer_string = ''
                    return DeltaMessage(reasoning_content=buffered_content, content=delta_text)
                
        if self.check_is_part_of_special_string():
            if self.is_reasoning_end(delta_token_ids):
                return DeltaMessage(reasoning_content=self.buffer_string)
            else:
                return None
        else:
            delta_text = self.buffer_string
            self.buffer_string = ''

        if self.think_end_token in current_text:
            return DeltaMessage(content=delta_text)
        else:
            return DeltaMessage(reasoning_content=delta_text)


    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if len(input_ids) > 1:
            return False
        return self.no_reasoning_content or self.end_token_id in input_ids


    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.end_token_id) + 1:]
