# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                # end token in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                # Remove newlines after </think> token
                content = content.lstrip('\n')
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # end token in previous, thinking content ends
                # Only strip newlines on the first delta right after </think>
                if previous_token_ids[-1] == self.end_token_id:
                    content = delta_text.lstrip('\n')
                else:
                    content = delta_text
                return DeltaMessage(content=content if content else None)
            else:
                # no end token in previous or delta, reasoning content continues
                return DeltaMessage(reasoning=delta_text)

        return ret

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        Removes newlines after </think> token if present.
        """
        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # For models that may not generate start token,
        # assume the reasoning content is always at the start.
        if self.end_token not in model_output:
            return model_output, None
        else:
            reasoning, _, content = model_output.partition(self.end_token)
            # Remove newlines after </think> token
            content = content.lstrip('\n')
            # If generation stops right after end-of-think or only had newlines, return null content
            final_content = content or None
            return reasoning, final_content
