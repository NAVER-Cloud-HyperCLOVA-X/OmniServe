# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, cast

import jinja2
import torch
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.embedding_processor import (
    EmbeddingProcessor,
    MultiModalEmbeddings,
    get_embedding_processor,
    load_multimodal_embeddings,
    load_vision_embeddings,
)
from vllm.entrypoints.openai.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.renderer import RenderConfig
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs.data import EmbedsPrompt, TokensPrompt, is_embeds_prompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters

# Default placeholder token for vision embedding injection
DEFAULT_PLACEHOLDER_TOKEN = "<|ptoken|>"

logger = init_logger(__name__)


class OpenAIServingCompletion(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        # set up logits processors
        self.logits_processors = self.model_config.logits_processors

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        self.enable_force_include_usage = enable_force_include_usage
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default completion sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )
        
        # Embedding processor for vision/audio embedding injection
        # Lazily initialized on first use
        self._embedding_processor: Optional[EmbeddingProcessor] = None

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response("suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        # Check for vision_embeddings - cannot use with prompt_embeds
        has_vision_embeddings = (
            request.vision_embeddings is not None
            or request.vision_embeddings_s3_path is not None
            or request.vision_embeddings_url is not None
        )
        
        # Check for multi-modal embeddings (discrete + continuous)
        has_multimodal_embeddings = (
            request.multimodal_embeddings_s3_path is not None
            or request.multimodal_embeddings_url is not None
        )
        
        if has_vision_embeddings and request.prompt_embeds is not None:
            return self.create_error_response(
                "Cannot use vision_embeddings together with prompt_embeds."
            )
        
        if has_multimodal_embeddings and request.prompt_embeds is not None:
            return self.create_error_response(
                "Cannot use multimodal_embeddings together with prompt_embeds."
            )
        
        if has_vision_embeddings and has_multimodal_embeddings:
            return self.create_error_response(
                "Cannot use both vision_embeddings and multimodal_embeddings. "
                "Choose one format."
            )

        request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request)

            if self.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = await self.engine_client.get_tokenizer()
            renderer = self._get_renderer(tokenizer)

            # Handle multi-modal embeddings (discrete + continuous with expansion)
            if has_multimodal_embeddings:
                engine_prompts = await self._process_multimodal_embeddings(
                    request=request,
                    tokenizer=tokenizer,
                )
            # Handle vision_embeddings: process placeholder tokens and inject embeddings
            elif has_vision_embeddings:
                engine_prompts = await self._process_vision_embeddings(
                    request=request,
                    tokenizer=tokenizer,
                )
            else:
                engine_prompts = await renderer.render_prompt_and_embeds(
                    prompt_or_prompts=request.prompt,
                    prompt_embeds=request.prompt_embeds,
                    config=self._build_render_config(request),
                )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text, prompt_token_ids, prompt_embeds = (
                    self._get_prompt_components(engine_prompt)
                )

                input_length = None
                if prompt_token_ids is not None:
                    input_length = len(prompt_token_ids)
                elif prompt_embeds is not None:
                    input_length = len(prompt_embeds)
                else:
                    raise NotImplementedError

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=input_length,
                    default_sampling_params=self.default_sampling_params,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )
                    validate_logits_processors_parameters(
                        self.logits_processors,
                        sampling_params,
                    )

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                # Mypy inconsistently requires this second cast in different
                # environments. It shouldn't be necessary (redundant from above)
                # but pre-commit in CI fails without it.
                engine_prompt = cast(EmbedsPrompt | TokensPrompt, engine_prompt)
                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    engine_request, tokenization_kwargs = await self._process_inputs(
                        request_id_item,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                    generator = self.engine_client.generate(
                        engine_request,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        prompt_text=prompt_text,
                        tokenization_kwargs=tokenization_kwargs,
                        data_parallel_rank=data_parallel_rank,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self.models.model_name(lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (
            request.stream
            and (request.best_of is None or request.n == request.best_of)
            and not request.use_beam_search
        )

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                engine_prompts,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
            )

        # Non-streaming response
        final_res_batch: list[RequestOutput | None] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    engine_prompt = engine_prompts[i]
                    final_res.prompt = (
                        None
                        if is_embeds_prompt(engine_prompt)
                        else engine_prompt.get("prompt")
                    )

            final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        engine_prompts: list[TokensPrompt | EmbedsPrompt],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts
        num_cached_tokens = None
        first_iteration = True

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs

                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False

                prompt_text = res.prompt
                if prompt_text is None:
                    engine_prompt = engine_prompts[prompt_idx]
                    prompt_text = (
                        None
                        if is_embeds_prompt(engine_prompt)
                        else engine_prompt.get("prompt")
                    )

                # Prompt details are excluded from later streamed outputs
                if prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    # Useful when request.return_token_ids is True
                    # Returning prompt token IDs shares the same logic
                    # with the echo implementation.
                    prompt_token_ids_to_return: list[int] | None = None

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        if request.return_token_ids:
                            prompt_text = ""
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids,
                                *output.token_ids,
                            ]
                            out_logprobs = [
                                *(prompt_logprobs or []),
                                *(output.logprobs or []),
                            ]
                        prompt_token_ids_to_return = prompt_token_ids
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        # has_echoed[i] is reused here to indicate whether
                        # we have already returned the prompt token IDs.
                        if not has_echoed[i] and request.return_token_ids:
                            prompt_token_ids_to_return = prompt_token_ids
                            has_echoed[i] = True

                        if (
                            not delta_text
                            and not delta_token_ids
                            and not previous_num_tokens[i]
                        ):
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                                prompt_token_ids=prompt_token_ids_to_return,
                                token_ids=(
                                    as_list(output.token_ids)
                                    if request.return_token_ids
                                    else None
                                ),
                            )
                        ],
                    )
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        kv_transfer_params = None
        last_final_res = None
        for final_res in final_res_batch:
            last_final_res = final_res
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: GenericSequence[dict[int, Logprob] | None] | None

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo:
                    if request.return_token_ids:
                        prompt_text = ""
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

                        output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                    prompt_token_ids=(
                        prompt_token_ids if request.return_token_ids else None
                    ),
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        if (
            self.enable_prompt_tokens_details
            and last_final_res
            and last_final_res.num_cached_tokens
        ):
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=last_final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage
        if final_res_batch:
            kv_transfer_params = final_res_batch[0].kv_transfer_params
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            kv_transfer_params=kv_transfer_params,
        )

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int,
        tokenizer: AnyTokenizer,
        initial_text_offset: int = 0,
        return_as_token_id: bool | None = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[float | None] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[dict[str, float] | None] = []

        last_token_len = 0

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append(
                    {
                        # Convert float("-inf") to the
                        # JSON-serializable float that OpenAI uses
                        self._get_decoded_token(
                            top_lp[1],
                            top_lp[0],
                            tokenizer,
                            return_as_token_id=should_return_as_token_id,
                        ): max(top_lp[1].logprob, -9999.0)
                        for i, top_lp in enumerate(step_top_logprobs.items())
                        if num_output_top_logprobs >= i
                    }
                )

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )

    def _build_render_config(
        self,
        request: CompletionRequest,
        max_input_length: int | None = None,
    ) -> RenderConfig:
        max_input_tokens_len = self.max_model_len - (request.max_tokens or 0)
        return RenderConfig(
            max_length=max_input_tokens_len,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
            cache_salt=request.cache_salt,
            needs_detokenization=bool(request.echo and not request.return_token_ids),
        )

    def _get_embedding_processor(self) -> EmbeddingProcessor:
        """Get or create the embedding processor for vision embedding injection.
        
        This method lazily initializes the embedding processor on first use,
        loading the model's embedding weights into CPU memory.
        
        Returns:
            The EmbeddingProcessor instance
        """
        if self._embedding_processor is None:
            # Get model path from model config
            model_path = self.model_config.model
            
            # Determine dtype from model config
            dtype = self.model_config.dtype
            if dtype is None:
                dtype = torch.bfloat16
            
            logger.info(
                "Initializing embedding processor for model: %s (dtype=%s)",
                model_path, dtype
            )
            
            self._embedding_processor = get_embedding_processor(
                model_path=model_path,
                dtype=dtype,
                device="cpu",
                cache_size=1000,
            )
        
        return self._embedding_processor

    async def _process_vision_embeddings(
        self,
        request: CompletionRequest,
        tokenizer: AnyTokenizer | None,
    ) -> list[EmbedsPrompt]:
        """Process vision embeddings by replacing placeholder tokens.
        
        This method:
        1. Tokenizes the prompt
        2. Uses cached embedding table to convert tokens to embeddings
        3. Replaces placeholder positions with vision embeddings
        4. Returns the result as EmbedsPrompt
        
        All processing is done on CPU using the cached embedding table,
        avoiding any modification to vllm/v1/ core code.
        
        Args:
            request: The completion request containing prompt and vision_embeddings
            tokenizer: The tokenizer to use
            
        Returns:
            List of EmbedsPrompt with combined embeddings
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for vision_embeddings processing. "
                "Cannot use skip_tokenizer_init with vision_embeddings."
            )
        
        if request.prompt is None:
            raise ValueError("prompt is required when using vision_embeddings")
        
        # Load vision embeddings from one of the available sources
        # Supports: direct embeddings, S3 path, or presigned URL
        vision_embeddings = load_vision_embeddings(
            vision_embeddings=request.vision_embeddings,
            vision_embeddings_s3_path=request.vision_embeddings_s3_path,
            vision_embeddings_url=request.vision_embeddings_url,
        )
        
        # Get the placeholder token ID
        placeholder_token = DEFAULT_PLACEHOLDER_TOKEN
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        
        # Check if placeholder token exists in vocabulary
        if placeholder_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Placeholder token '{placeholder_token}' not found in vocabulary. "
                "Please ensure the tokenizer has this token registered as a special token."
            )
        
        # Handle single or multiple prompts
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        # For now, only support single prompt with vision_embeddings
        if len(prompts) > 1:
            raise ValueError(
                "Multiple prompts are not supported with vision_embeddings. "
                "Please send one prompt at a time."
            )
        
        prompt = prompts[0]
        
        # Tokenize the prompt
        if isinstance(prompt, str):
            token_ids = tokenizer.encode(
                prompt, 
                add_special_tokens=request.add_special_tokens
            )
        elif isinstance(prompt, list):
            # Already token IDs
            token_ids = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        # Count placeholder tokens
        num_placeholders = sum(1 for t in token_ids if t == placeholder_token_id)
        num_vision_embeddings = len(vision_embeddings)
        
        if num_placeholders != num_vision_embeddings:
            raise ValueError(
                f"Mismatch: found {num_placeholders} placeholder tokens "
                f"({placeholder_token}) but got {num_vision_embeddings} "
                "vision embeddings"
            )
        
        if num_placeholders == 0:
            raise ValueError(
                f"No placeholder tokens ({placeholder_token}) found in prompt. "
                "Vision embeddings require placeholder tokens in the prompt."
            )
        
        # Get the embedding processor (cached, loads embedding table on first use)
        processor = self._get_embedding_processor()
        
        # Process embeddings using the embedding processor
        # This runs on CPU using the cached embedding table
        try:
            combined_embeddings = processor.process_with_embeddings(
                token_ids=token_ids,
                placeholder_token_id=placeholder_token_id,
                external_embeddings=vision_embeddings,
                use_cache=True,  # Enable caching for repeated requests
            )
        except Exception as e:
            logger.exception("Error processing vision embeddings")
            raise RuntimeError(f"Failed to process vision embeddings: {e}") from e
        
        # Convert to EmbedsPrompt format
        embeds_prompt: EmbedsPrompt = {
            "prompt_embeds": combined_embeddings,
        }
        
        if request.cache_salt is not None:
            embeds_prompt["cache_salt"] = request.cache_salt
        
        return [embeds_prompt]

    async def _process_multimodal_embeddings(
        self,
        request: CompletionRequest,
        tokenizer: AnyTokenizer | None,
    ) -> list[EmbedsPrompt]:
        """Process multi-modal embeddings with discrete and continuous components.
        
        This handles the new format where:
        - <|DISCRETE_IMAGE_PAD|> is expanded to discrete token embeddings (lookup from embedding table)
        - <|IMAGE_PAD|> is expanded to continuous embeddings (direct injection)
        
        Args:
            request: The completion request
            tokenizer: The tokenizer to use
            
        Returns:
            List of EmbedsPrompt with expanded embeddings
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for multimodal_embeddings processing."
            )
        
        if request.prompt is None:
            raise ValueError("prompt is required when using multimodal_embeddings")
        
        # Load multi-modal embeddings from S3 or URL
        multimodal_data = load_multimodal_embeddings(
            multimodal_embeddings_s3_path=request.multimodal_embeddings_s3_path,
            multimodal_embeddings_url=request.multimodal_embeddings_url,
        )
        
        # Define token names for discrete and continuous placeholders
        discrete_placeholder_token = "<|DISCRETE_IMAGE_PAD|>"
        continuous_placeholder_token = "<|IMAGE_PAD|>"
        discrete_start_token = "<|discrete_image_start|>"
        discrete_end_token = "<|discrete_image_end|>"
        continuous_start_token = "<|image_start|>"
        continuous_end_token = "<|image_end|>"
        
        # Get token IDs
        discrete_placeholder_id = tokenizer.convert_tokens_to_ids(discrete_placeholder_token)
        continuous_placeholder_id = tokenizer.convert_tokens_to_ids(continuous_placeholder_token)
        discrete_start_id = tokenizer.convert_tokens_to_ids(discrete_start_token)
        discrete_end_id = tokenizer.convert_tokens_to_ids(discrete_end_token)
        continuous_start_id = tokenizer.convert_tokens_to_ids(continuous_start_token)
        continuous_end_id = tokenizer.convert_tokens_to_ids(continuous_end_token)
        
        # Validate placeholder tokens exist
        if discrete_placeholder_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Placeholder token '{discrete_placeholder_token}' not found in vocabulary."
            )
        if continuous_placeholder_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Placeholder token '{continuous_placeholder_token}' not found in vocabulary."
            )
        
        # Handle prompt
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        if len(prompts) > 1:
            raise ValueError(
                "Multiple prompts are not supported with multimodal_embeddings."
            )
        
        prompt = prompts[0]
        
        # Tokenize the prompt
        if isinstance(prompt, str):
            token_ids = tokenizer.encode(
                prompt, 
                add_special_tokens=request.add_special_tokens
            )
        elif isinstance(prompt, list):
            token_ids = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        # Validate placeholders exist in prompt
        num_discrete = sum(1 for t in token_ids if t == discrete_placeholder_id)
        num_continuous = sum(1 for t in token_ids if t == continuous_placeholder_id)
        
        logger.info(
            "Multi-modal prompt: %d discrete placeholders, %d continuous placeholders, "
            "total tokens: %d",
            num_discrete, num_continuous, len(token_ids)
        )
        
        if num_discrete == 0 and num_continuous == 0:
            raise ValueError(
                f"No placeholder tokens found in prompt. "
                f"Expected {discrete_placeholder_token} and/or {continuous_placeholder_token}"
            )
        
        # Validate that placeholders are within correct start/end tokens
        def validate_placeholder_position(
            token_ids: list[int],
            placeholder_id: int,
            start_id: int,
            end_id: int,
            placeholder_name: str,
            start_name: str,
            end_name: str,
        ) -> None:
            """Validate that placeholder tokens are between start and end tokens."""
            in_region = False
            for i, token_id in enumerate(token_ids):
                if token_id == start_id:
                    in_region = True
                elif token_id == end_id:
                    in_region = False
                elif token_id == placeholder_id:
                    if not in_region:
                        raise ValueError(
                            f"Invalid prompt structure: {placeholder_name} found at position {i} "
                            f"but it must be between {start_name} and {end_name}. "
                            f"Expected format: {start_name}{placeholder_name}{end_name}"
                        )
        
        # Validate discrete placeholder position
        if num_discrete > 0:
            validate_placeholder_position(
                token_ids=token_ids,
                placeholder_id=discrete_placeholder_id,
                start_id=discrete_start_id,
                end_id=discrete_end_id,
                placeholder_name=discrete_placeholder_token,
                start_name=discrete_start_token,
                end_name=discrete_end_token,
            )
            logger.info(
                "Validated: %s is correctly placed between %s and %s",
                discrete_placeholder_token, discrete_start_token, discrete_end_token
            )
        
        # Validate continuous placeholder position
        if num_continuous > 0:
            validate_placeholder_position(
                token_ids=token_ids,
                placeholder_id=continuous_placeholder_id,
                start_id=continuous_start_id,
                end_id=continuous_end_id,
                placeholder_name=continuous_placeholder_token,
                start_name=continuous_start_token,
                end_name=continuous_end_token,
            )
            logger.info(
                "Validated: %s is correctly placed between %s and %s",
                continuous_placeholder_token, continuous_start_token, continuous_end_token
            )
        
        # Get the embedding processor
        processor = self._get_embedding_processor()
        
        # Process with expansion
        try:
            combined_embeddings = processor.process_multimodal_with_expansion(
                token_ids=token_ids,
                multimodal_data=multimodal_data,
                discrete_placeholder_id=discrete_placeholder_id,
                continuous_placeholder_id=continuous_placeholder_id,
            )
        except Exception as e:
            logger.exception("Error processing multi-modal embeddings")
            raise RuntimeError(f"Failed to process multi-modal embeddings: {e}") from e
        
        # Convert to EmbedsPrompt format
        embeds_prompt: EmbedsPrompt = {
            "prompt_embeds": combined_embeddings,
        }
        
        if request.cache_salt is not None:
            embeds_prompt["cache_salt"] = request.cache_salt
        
        return [embeds_prompt]
