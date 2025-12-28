# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Final

import jinja2
import partial_json_parser
import regex as re
from fastapi import Request
from openai_harmony import Message as OpenAIMessage

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    make_tool_call_id,
)
from vllm.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    get_system_message,
    parse_chat_output,
    parse_input_to_harmony_message,
    render_for_completion,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.embedding_processor import (
    EmbeddingProcessor,
    ModalityEmbedding,
    download_embeddings_from_s3,
)
import hashlib
import threading
from collections import OrderedDict

# Cache for processed multimodal embeddings (prompt + s3_urls -> combined_embeddings)
_multimodal_embedding_cache: OrderedDict = OrderedDict()
_multimodal_cache_max_size = 100  # Max cached items
_multimodal_cache_lock = threading.Lock()


def _get_multimodal_cache_key(prompt: str, s3_urls: tuple) -> str:
    """Generate cache key from prompt and S3 URLs."""
    key_data = f"{prompt}|{s3_urls}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached_multimodal_embedding(cache_key: str):
    """Get cached embedding if available."""
    with _multimodal_cache_lock:
        if cache_key in _multimodal_embedding_cache:
            # Move to end (LRU)
            _multimodal_embedding_cache.move_to_end(cache_key)
            return _multimodal_embedding_cache[cache_key]
    return None


def _set_cached_multimodal_embedding(cache_key: str, data):
    """Cache the processed embedding."""
    with _multimodal_cache_lock:
        if len(_multimodal_embedding_cache) >= _multimodal_cache_max_size:
            _multimodal_embedding_cache.popitem(last=False)
        _multimodal_embedding_cache[cache_key] = data
from vllm.entrypoints.openai.serving_engine import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
from vllm.utils.collection_utils import as_list
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.enable_log_outputs = enable_log_outputs

        # set up logits processors
        self.logits_processors = self.model_config.logits_processors

        # set up reasoning parser
        self.reasoning_parser = self._get_reasoning_parser(
            reasoning_parser_name=reasoning_parser
        )
        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        self.tool_parser = self._get_tool_parser(
            tool_parser_name=tool_parser, enable_auto_tools=enable_auto_tools
        )
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default chat sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )
        if self.model_config.hf_config.model_type == "kimi_k2":
            self.tool_call_id_type = "kimi_k2"
        else:
            self.tool_call_id_type = "random"

        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

        # Initialize EmbeddingProcessor for multimodal embedding injection
        self._embedding_processor: EmbeddingProcessor | None = None
        self._embedding_processor_initialized = False

    def _get_embedding_processor(self) -> EmbeddingProcessor:
        """Get or initialize the embedding processor."""
        if not self._embedding_processor_initialized:
            try:
                self._embedding_processor = EmbeddingProcessor(
                    model_path=self.model_config.model,
                    device="cpu",
                )
                logger.info("EmbeddingProcessor initialized for chat completions")
            except Exception as e:
                logger.warning(f"Failed to initialize EmbeddingProcessor: {e}")
                self._embedding_processor = None
            self._embedding_processor_initialized = True
        
        if self._embedding_processor is None:
            raise RuntimeError(
                "EmbeddingProcessor not available. "
                "This is required for multimodal embedding injection."
            )
        return self._embedding_processor

    def _is_video_url(self, url: str) -> bool:
        """Check if URL points to a video file."""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
        url_lower = url.lower()
        return any(url_lower.endswith(ext) or f"{ext}." in url_lower for ext in video_extensions)

    def _extract_image_urls_from_messages(
        self,
        messages: list,
    ) -> tuple[list[str], list[str]]:
        """Extract image and video URLs from chat messages.
        
        Looks for content items with type='image_url' and extracts the URL.
        Separates images and videos based on file extension.
        
        Returns:
            Tuple of (image_urls, video_urls)
        """
        image_urls = []
        video_urls = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_url_obj = item.get("image_url", {})
                    url = image_url_obj.get("url", "")
                    if url:
                        if self._is_video_url(url):
                            video_urls.append(url)
                            logger.info("Detected video URL: %s", url[:50])
                        else:
                            image_urls.append(url)
        return image_urls, video_urls

    def _extract_audio_s3_paths_from_messages(
        self,
        messages: list,
    ) -> list[str]:
        """Extract audio S3 paths from chat messages.
        
        Looks for content items with type='input_audio' and extracts the S3 path
        from base64-encoded data field.
        
        The data field contains base64-encoded S3 path like:
        base64("s3://bucket/path/to/audio.pt")
        """
        import base64
        
        audio_s3_paths = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "input_audio":
                    input_audio = item.get("input_audio", {})
                    data = input_audio.get("data", "")
                    if data:
                        try:
                            # Decode base64 to get S3 path
                            decoded = base64.b64decode(data).decode("utf-8")
                            if decoded.startswith("s3://"):
                                audio_s3_paths.append(decoded)
                                logger.info("Extracted audio S3 path: %s", decoded[:50])
                        except Exception as e:
                            logger.warning("Failed to decode audio data: %s", e)
        return audio_s3_paths

    def _parse_s3_url(self, url: str) -> str | None:
        """Parse S3 URL to get bucket/path format.
        
        Supports:
        - s3://bucket/path/to/file.pt -> bucket/path/to/file.pt
        - Direct S3 paths like 'bucket/path/to/file.pt'
        """
        if url.startswith("s3://"):
            return url[5:]  # Remove 's3://' prefix
        return None

    async def _preprocess_s3_multimodal_chat(
        self,
        request: ChatCompletionRequest,
        tokenizer: AnyTokenizer,
        image_urls: list[str] | None = None,
        video_urls: list[str] | None = None,
        audio_s3_paths: list[str] | None = None,
    ) -> tuple[list, list, list]:
        """
        Preprocess chat with S3-based multimodal embeddings (image, video, and/or audio).
        
        This method bypasses the standard multimodal model check and applies
        the chat template directly, then injects embeddings from S3.
        
        Includes caching for processed embeddings to avoid recomputation.
        """
        import time as _time
        _start_time = _time.perf_counter()
        
        image_urls = image_urls or []
        video_urls = video_urls or []
        audio_s3_paths = audio_s3_paths or []
        
        # Generate cache key from messages and S3 URLs
        all_s3_urls = tuple(sorted(image_urls + video_urls + audio_s3_paths))
        messages_str = str([m.model_dump() if hasattr(m, 'model_dump') else m for m in request.messages])
        cache_key = _get_multimodal_cache_key(messages_str, all_s3_urls)
        
        # Check cache
        cached = _get_cached_multimodal_embedding(cache_key)
        if cached is not None:
            _elapsed = (_time.perf_counter() - _start_time) * 1000
            logger.info("Multimodal embedding CACHE HIT (%.1fms)", _elapsed)
            return cached
        
        # Apply chat template directly
        chat_template = request.chat_template or self.chat_template
        
        # Convert messages to the format expected by the chat template
        # Also fix video S3 URLs so chat template recognizes them as videos
        messages = []
        for msg in request.messages:
            if isinstance(msg, dict):
                msg_copy = dict(msg)
            else:
                msg_copy = msg.model_dump()
            
            # Fix video URLs: remove .pt suffix so chat template recognizes video extensions
            content = msg_copy.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url_obj = item.get("image_url", {})
                        url = image_url_obj.get("url", "")
                        # If this is a video S3 URL, fix the extension for chat template
                        if url in video_urls and url.endswith(".pt"):
                            # Remove .pt suffix to expose original video extension
                            fixed_url = url[:-3]  # Remove .pt
                            new_item = dict(item)
                            new_item["image_url"] = {"url": fixed_url}
                            new_content.append(new_item)
                            logger.info("Fixed video URL for chat template: %s -> %s", url[-50:], fixed_url[-50:])
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
            
            messages.append(msg_copy)

        # Build tool_dicts from request.tools (same as regular path)
        tool_dicts = None
        if request.tools:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        # Get chat_template_kwargs (e.g., skip_reasoning)
        chat_template_kwargs = request.chat_template_kwargs or {}

        # Apply chat template with tools and chat_template_kwargs
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            chat_template=chat_template,
            add_generation_prompt=request.add_generation_prompt,
            tokenize=False,
            tools=tool_dicts,
            **chat_template_kwargs,
        )
        
        logger.info("S3 multimodal chat: Applied chat template, prompt length: %d", len(prompt))
        logger.info("S3 multimodal chat: tools=%s, chat_template_kwargs=%s", tool_dicts is not None, chat_template_kwargs)
        logger.info("S3 multimodal chat: prompt ends with: %s", prompt[-300:])
        
        # Tokenize the prompt
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=request.add_special_tokens,
        )
        
        # Collect all multimodal data from S3
        all_multimodal_data = []
        
        # Process image URLs
        for s3_url in image_urls:
            if s3_url.startswith("s3://"):
                s3_path = self._parse_s3_url(s3_url)
                if s3_path:
                    data = download_embeddings_from_s3(
                        s3_path=s3_path, allow_dict=True, use_cache=True
                    )
                    if isinstance(data, dict):
                        all_multimodal_data.append(("image", data))
                        logger.info("Loaded image embeddings from S3: %s", s3_url[:50])
        
        # Process video URLs
        for s3_url in video_urls:
            if s3_url.startswith("s3://"):
                s3_path = self._parse_s3_url(s3_url)
                if s3_path:
                    data = download_embeddings_from_s3(
                        s3_path=s3_path, allow_dict=True, use_cache=True
                    )
                    if isinstance(data, dict):
                        all_multimodal_data.append(("video", data))
                        logger.info("Loaded video embeddings from S3: %s", s3_url[:50])
        
        # Process audio S3 paths
        for s3_url in audio_s3_paths:
            if s3_url.startswith("s3://"):
                s3_path = self._parse_s3_url(s3_url)
                if s3_path:
                    data = download_embeddings_from_s3(
                        s3_path=s3_path, allow_dict=True, use_cache=True
                    )
                    if isinstance(data, dict):
                        all_multimodal_data.append(("audio", data))
                        logger.info("Loaded audio embeddings from S3: %s", s3_url[:50])
        
        if not all_multimodal_data:
            raise ValueError("No valid multimodal data found from S3")
        
        # Define placeholder tokens for each modality
        modality_placeholders = {
            "image": {
                "discrete_placeholder": "<|DISCRETE_IMAGE_PAD|>",
                "continuous_placeholder": "<|IMAGE_PAD|>",
                "discrete_start": "<|discrete_image_start|>",
                "discrete_end": "<|discrete_image_end|>",
                "continuous_start": "<|image_start|>",
                "continuous_end": "<|image_end|>",
            },
            "video": {
                # Video only has continuous embeddings (no discrete)
                "discrete_placeholder": None,
                "continuous_placeholder": "<|VIDEO_PAD|>",
                "discrete_start": None,
                "discrete_end": None,
                "continuous_start": "<|video_start|>",
                "continuous_end": "<|video_end|>",
            },
            "audio": {
                "discrete_placeholder": "<|DISCRETE_AUDIO_PAD|>",
                "continuous_placeholder": "<|AUDIO_PAD|>",
                "discrete_start": "<|discrete_audio_start|>",
                "discrete_end": "<|discrete_audio_end|>",
                "continuous_start": "<|audio_start|>",
                "continuous_end": "<|audio_end|>",
            },
        }
        
        # Get token IDs for all placeholders (handle None tokens)
        placeholder_token_ids = {}
        for modality, placeholders in modality_placeholders.items():
            placeholder_token_ids[modality] = {
                key: tokenizer.convert_tokens_to_ids(token) if token else None
                for key, token in placeholders.items()
            }
        
        # Count and validate placeholders for each modality
        def validate_placeholder_position(
            token_ids: list[int],
            placeholder_id: int,
            start_id: int,
            end_id: int,
            placeholder_name: str,
            start_name: str,
            end_name: str,
        ) -> None:
            in_region = False
            for i, token_id in enumerate(token_ids):
                if token_id == start_id:
                    in_region = True
                elif token_id == end_id:
                    in_region = False
                elif token_id == placeholder_id:
                    if not in_region:
                        raise ValueError(
                            f"Invalid prompt structure: {placeholder_name} at position {i} "
                            f"must be between {start_name} and {end_name}."
                        )
        
        for modality, multimodal_data in all_multimodal_data:
            ids = placeholder_token_ids[modality]
            placeholders = modality_placeholders[modality]
            
            discrete_id = ids["discrete_placeholder"]
            continuous_id = ids["continuous_placeholder"]
            
            # Count placeholders (handle None for video discrete)
            num_discrete = sum(1 for t in token_ids if t == discrete_id) if discrete_id else 0
            num_continuous = sum(1 for t in token_ids if t == continuous_id) if continuous_id else 0
            
            logger.info(
                "S3 %s: %d discrete, %d continuous placeholders",
                modality, num_discrete, num_continuous
            )
            
            if num_discrete == 0 and num_continuous == 0:
                raise ValueError(
                    f"No placeholder tokens found for {modality}. "
                    f"Chat template may not have generated expected placeholders."
                )
            
            # Validate positions (skip if placeholder is None)
            if num_discrete > 0 and discrete_id is not None:
                validate_placeholder_position(
                    token_ids, discrete_id,
                    ids["discrete_start"], ids["discrete_end"],
                    placeholders["discrete_placeholder"],
                    placeholders["discrete_start"], placeholders["discrete_end"]
                )
            
            if num_continuous > 0 and continuous_id is not None:
                validate_placeholder_position(
                    token_ids, continuous_id,
                    ids["continuous_start"], ids["continuous_end"],
                    placeholders["continuous_placeholder"],
                    placeholders["continuous_start"], placeholders["continuous_end"]
                )
        
        # Get embedding processor
        processor = self._get_embedding_processor()
        
        import torch
        
        # Process multimodal embeddings with expansion for each modality
        # Start with token embeddings
        combined_embeddings = processor.embed_tokens(token_ids)
        current_token_ids = list(token_ids)
        
        for modality, multimodal_data in all_multimodal_data:
            ids = placeholder_token_ids[modality]
            discrete_id = ids["discrete_placeholder"]
            continuous_id = ids["continuous_placeholder"]
            
            # Get discrete and continuous data
            discrete_data = multimodal_data.get("discrete")
            continuous_data = multimodal_data.get("continuous")
            
            # Process discrete embeddings (expand single placeholder to multiple)
            if discrete_data is not None:
                # Find discrete placeholder position
                try:
                    pos = current_token_ids.index(discrete_id)
                    
                    # Get discrete token embeddings
                    discrete_token_ids = discrete_data.squeeze().to(torch.long).tolist()
                    discrete_embeddings = processor.embed_tokens(discrete_token_ids)
                    
                    # Replace single placeholder with multiple embeddings
                    combined_embeddings = torch.cat([
                        combined_embeddings[:pos],
                        discrete_embeddings,
                        combined_embeddings[pos+1:],
                    ], dim=0)
                    
                    # Update token ids for tracking
                    current_token_ids = current_token_ids[:pos] + [-1] * len(discrete_token_ids) + current_token_ids[pos+1:]
                    
                    logger.info(
                        "S3 %s: injected %d discrete embeddings at position %d",
                        modality, len(discrete_token_ids), pos
                    )
                except ValueError:
                    pass  # No discrete placeholder found
            
            # Process continuous embeddings (expand single placeholder to multiple)
            if continuous_data is not None:
                # Find continuous placeholder position
                try:
                    pos = current_token_ids.index(continuous_id)
                    
                    # Get continuous embeddings
                    continuous_embeddings = continuous_data.to(
                        dtype=processor.dtype, device=processor.device
                    )
                    
                    # Replace single placeholder with multiple embeddings
                    combined_embeddings = torch.cat([
                        combined_embeddings[:pos],
                        continuous_embeddings,
                        combined_embeddings[pos+1:],
                    ], dim=0)
                    
                    # Update token ids for tracking
                    current_token_ids = current_token_ids[:pos] + [-1] * continuous_embeddings.shape[0] + current_token_ids[pos+1:]
                    
                    logger.info(
                        "S3 %s: injected %d continuous embeddings at position %d",
                        modality, continuous_embeddings.shape[0], pos
                    )
                except ValueError:
                    pass  # No continuous placeholder found
        
        _elapsed = (_time.perf_counter() - _start_time) * 1000
        logger.info(
            "S3 multimodal chat: embedding injection complete, %d embeddings (%.1fms)",
            combined_embeddings.shape[0], _elapsed
        )
        
        # Create EmbedsPrompt
        from vllm.inputs.data import EmbedsPrompt
        embeds_prompt: EmbedsPrompt = {
            "prompt_embeds": combined_embeddings,
        }
        
        if hasattr(request, "cache_salt") and request.cache_salt is not None:
            embeds_prompt["cache_salt"] = request.cache_salt
        
        # Create conversation for response generation
        conversation = [{"role": m.get("role", "user"), "content": ""} for m in messages]
        
        # Cache the result for future requests
        result = ([embeds_prompt], conversation, [prompt])
        _set_cached_multimodal_embedding(cache_key, result)
        
        return result

    async def _process_multimodal_chat(
        self,
        request: ChatCompletionRequest,
        engine_prompt: EngineTokensPrompt,
        tokenizer: AnyTokenizer,
        image_urls: list[str],
    ) -> EngineTokensPrompt:
        """Process chat with multimodal embeddings.
        
        Downloads embeddings from S3 and injects them into the prompt.
        Supports multiple images - each image's embeddings are injected
        into corresponding placeholder regions in order.
        """
        if not image_urls:
            return engine_prompt
        
        # Filter valid S3 URLs
        valid_s3_paths = []
        for url in image_urls:
            s3_path = self._parse_s3_url(url)
            if s3_path is not None:
                valid_s3_paths.append((url, s3_path))
            else:
                logger.warning(
                    "Image URL '%s' is not an S3 path. Skipping.",
                    url[:50] if url else "None"
                )
        
        if not valid_s3_paths:
            logger.warning("No valid S3 image URLs found. Skipping embedding injection.")
            return engine_prompt
        
        logger.info(
            "Processing %d images for multimodal chat embedding injection",
            len(valid_s3_paths)
        )
        
        try:
            # Download all multimodal embeddings from S3
            all_multimodal_data = []
            for url, s3_path in valid_s3_paths:
                multimodal_data = download_embeddings_from_s3(
                    s3_path=s3_path,
                    allow_dict=True,
                    use_cache=True,
                )
                
                if not isinstance(multimodal_data, dict):
                    raise ValueError(f"Expected dict from S3 for {url[:50]}, got {type(multimodal_data)}")
                
                all_multimodal_data.append(multimodal_data)
                logger.debug("Downloaded embeddings for image: %s", url[:50])
            
            # Get token IDs from engine prompt
            token_ids = list(engine_prompt["prompt_token_ids"])
            
            # Get placeholder token IDs
            discrete_placeholder_token = "<|DISCRETE_IMAGE_PAD|>"
            continuous_placeholder_token = "<|IMAGE_PAD|>"
            discrete_start_token = "<|discrete_image_start|>"
            discrete_end_token = "<|discrete_image_end|>"
            continuous_start_token = "<|image_start|>"
            continuous_end_token = "<|image_end|>"
            
            discrete_placeholder_id = tokenizer.convert_tokens_to_ids(discrete_placeholder_token)
            continuous_placeholder_id = tokenizer.convert_tokens_to_ids(continuous_placeholder_token)
            discrete_start_id = tokenizer.convert_tokens_to_ids(discrete_start_token)
            discrete_end_id = tokenizer.convert_tokens_to_ids(discrete_end_token)
            continuous_start_id = tokenizer.convert_tokens_to_ids(continuous_start_token)
            continuous_end_id = tokenizer.convert_tokens_to_ids(continuous_end_token)
            
            # Find all image regions (each region is a pair of discrete + continuous blocks)
            # Each image has: <|discrete_image_start|>...<|discrete_image_end|>
            #                 <|image_start|>...<|image_end|>
            image_regions = []
            current_region = {}
            
            for i, token_id in enumerate(token_ids):
                if token_id == discrete_start_id:
                    current_region = {"discrete_start": i}
                elif token_id == discrete_end_id and "discrete_start" in current_region:
                    current_region["discrete_end"] = i
                elif token_id == continuous_start_id and "discrete_end" in current_region:
                    current_region["continuous_start"] = i
                elif token_id == continuous_end_id and "continuous_start" in current_region:
                    current_region["continuous_end"] = i
                    image_regions.append(current_region)
                    current_region = {}
            
            num_regions = len(image_regions)
            num_images = len(all_multimodal_data)
            
            logger.info(
                "Chat multimodal: found %d image regions, have %d image embeddings",
                num_regions, num_images
            )
            
            if num_regions == 0:
                logger.warning("No image regions found in chat prompt")
                return engine_prompt
            
            if num_regions != num_images:
                logger.warning(
                    "Mismatch: %d image regions but %d image embeddings. "
                    "Will process min(%d, %d) images.",
                    num_regions, num_images, num_regions, num_images
                )
            
            # Get embedding processor
            processor = self._get_embedding_processor()
            
            # Process each image region with its corresponding embeddings
            # Build combined embeddings by processing token_ids segment by segment
            import torch
            combined_parts = []
            last_end = 0
            
            for idx, region in enumerate(image_regions):
                if idx >= num_images:
                    # No more embeddings available, use placeholder embeddings
                    logger.warning("No embedding for image region %d, using placeholder", idx)
                    break
                
                multimodal_data = all_multimodal_data[idx]
                
                # Tokens before this region
                if region["discrete_start"] > last_end:
                    before_tokens = token_ids[last_end:region["discrete_start"]]
                    before_embeddings = processor.get_embeddings_for_tokens(before_tokens)
                    combined_parts.append(before_embeddings)
                
                # Process discrete region
                discrete_tokens = token_ids[region["discrete_start"]:region["discrete_end"] + 1]
                discrete_embeddings = processor.process_discrete_region(
                    discrete_tokens,
                    multimodal_data,
                    discrete_start_id,
                    discrete_end_id,
                    discrete_placeholder_id,
                )
                combined_parts.append(discrete_embeddings)
                
                # Tokens between discrete and continuous (should be minimal)
                between_start = region["discrete_end"] + 1
                between_end = region["continuous_start"]
                if between_end > between_start:
                    between_tokens = token_ids[between_start:between_end]
                    between_embeddings = processor.get_embeddings_for_tokens(between_tokens)
                    combined_parts.append(between_embeddings)
                
                # Process continuous region
                continuous_tokens = token_ids[region["continuous_start"]:region["continuous_end"] + 1]
                continuous_embeddings = processor.process_continuous_region(
                    continuous_tokens,
                    multimodal_data,
                    continuous_start_id,
                    continuous_end_id,
                    continuous_placeholder_id,
                )
                combined_parts.append(continuous_embeddings)
                
                last_end = region["continuous_end"] + 1
            
            # Remaining tokens after all image regions
            if last_end < len(token_ids):
                remaining_tokens = token_ids[last_end:]
                remaining_embeddings = processor.get_embeddings_for_tokens(remaining_tokens)
                combined_parts.append(remaining_embeddings)
            
            # Concatenate all parts
            combined_embeddings = torch.cat(combined_parts, dim=0)
            
            # Return as EmbedsPrompt
            from vllm.inputs.data import EmbedsPrompt
            embeds_prompt: EmbedsPrompt = {
                "prompt_embeds": combined_embeddings,
            }
            
            if hasattr(request, "cache_salt") and request.cache_salt is not None:
                embeds_prompt["cache_salt"] = request.cache_salt
            
            logger.info(
                "Chat multimodal embedding injection complete: %d images -> %d embeddings",
                min(num_regions, num_images), combined_embeddings.shape[0]
            )
            
            return embeds_prompt
            
        except Exception as e:
            logger.exception("Error processing multimodal chat embeddings")
            raise RuntimeError(f"Failed to process multimodal embeddings: {e}") from e

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True
            )

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            if request.tools is None or (
                request.tool_choice == "none"
                and self.exclude_tools_when_tool_choice_none
            ):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Check for S3 image/video URLs and audio S3 paths
                image_urls, video_urls = self._extract_image_urls_from_messages(request.messages)
                audio_s3_paths = self._extract_audio_s3_paths_from_messages(request.messages)
                
                has_s3_images = any(url.startswith("s3://") for url in image_urls)
                has_s3_videos = any(url.startswith("s3://") for url in video_urls)
                has_s3_audio = len(audio_s3_paths) > 0
                
                if has_s3_images or has_s3_videos or has_s3_audio:
                    # Handle S3-based multimodal directly
                    # Apply chat template manually to bypass multimodal model check
                    engine_prompts, conversation, request_prompts = await self._preprocess_s3_multimodal_chat(
                        request=request,
                        tokenizer=tokenizer,
                        image_urls=image_urls if has_s3_images else None,
                        video_urls=video_urls if has_s3_videos else None,
                        audio_s3_paths=audio_s3_paths if has_s3_audio else None,
                    )
                else:
                    # Common case - no S3 images
                    error_check_ret = self._validate_chat_template(
                        request_chat_template=request.chat_template,
                        chat_template_kwargs=request.chat_template_kwargs,
                        trust_request_chat_template=self.trust_request_chat_template,
                    )
                    if error_check_ret is not None:
                        return error_check_ret
                    (
                        conversation,
                        request_prompts,
                        engine_prompts,
                    ) = await self._preprocess_chat(
                        request,
                        tokenizer,
                        request.messages,
                        chat_template=request.chat_template or self.chat_template,
                        chat_template_content_format=self.chat_template_content_format,
                        add_generation_prompt=request.add_generation_prompt,
                        continue_final_message=request.continue_final_message,
                        tool_dicts=tool_dicts,
                        documents=request.documents,
                        chat_template_kwargs=request.chat_template_kwargs,
                        tool_parser=tool_parser,
                        add_special_tokens=request.add_special_tokens,
                    )
            else:
                # For GPT-OSS.
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = self._make_request_with_harmony(request)
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text, _, _ = self._get_prompt_components(request_prompts[i])

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                # Calculate input length based on prompt type
                if "prompt_token_ids" in engine_prompt:
                    input_length = len(engine_prompt["prompt_token_ids"])
                elif "prompt_embeds" in engine_prompt:
                    input_length = engine_prompt["prompt_embeds"].shape[0]
                else:
                    input_length = 0

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

                self._log_inputs(
                    request_id,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    engine_request, tokenization_kwargs = await self._process_inputs(
                        request_id,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                    generator = self.engine_client.generate(
                        engine_request,
                        sampling_params,
                        request_id,
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

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    @staticmethod
    def _bracket_level(s: str, opening="{", closing="}") -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str, previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == "{":
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == "}":
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ",":
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str | None,
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: int | None = None,
    ) -> tuple[DeltaMessage | None, bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            obj = partial_json_parser.loads(current_text)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug("not enough tokens to parse into JSON yet")
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text
            )
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and (
                "name" not in current_tool_call or "parameters" not in current_tool_call
            ):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(
                        r'.*"parameters":\s*(.*)', current_text, re.DOTALL
                    )
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text
                    )

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if finishes_previous_tool and "parameters" not in current_tool_call:
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx,
                    )
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                id=tool_call_id,
                                function=DeltaFunctionCall(
                                    name=current_tool_call["name"], arguments=arguments
                                ),
                                index=len(obj) - 1,
                                type="function",
                            )
                        ]
                    )

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text
                    )

                    if delta_text != "":
                        delta_message = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        # OpenAI API returns None
                                        # instead of name every time
                                        name=None,
                                        arguments=delta_text,
                                    ),
                                    index=len(obj) - 1,
                                )
                            ]
                        )
                    else:
                        delta_message = None

        return delta_message, function_name_returned

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        all_previous_token_ids: list[list[int]] | None
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or self.reasoning_parser:
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        else:
            all_previous_token_ids = None

        try:
            if self.reasoning_parser:
                reasoning_parser = self.reasoning_parser(
                    tokenizer,
                    chat_template_kwargs=request.chat_template_kwargs,  # type: ignore
                )
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[ToolParser | None] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient
                        delta_text = ""
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                            delta_text += harmony_parser.last_content_delta or ""
                        cur_channel = harmony_parser.current_channel
                        cur_recipient = harmony_parser.current_recipient
                    else:
                        delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or self.reasoning_parser:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids
                            )
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        if cur_channel == "final":
                            delta_message = DeltaMessage(content=delta_text)
                        elif cur_channel == "analysis":
                            if request.include_reasoning:
                                delta_message = DeltaMessage(reasoning=delta_text)
                            else:
                                delta_message = None
                        elif (
                            cur_channel == "commentary"
                            and cur_recipient
                            and cur_recipient.startswith("functions.")
                        ):
                            # Count completed tool calls to determine index
                            base_index = 0
                            for msg in harmony_parser.messages:
                                if (
                                    msg.channel == "commentary"
                                    and msg.recipient
                                    and msg.recipient.startswith("functions.")
                                ):
                                    base_index += 1

                            if prev_recipient != cur_recipient:
                                tool_name = cur_recipient.split("functions.", 1)[1]
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            id=make_tool_call_id(),
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=tool_name,
                                                arguments="",
                                            ),
                                            index=base_index,
                                        )
                                    ]
                                )
                            elif delta_text:
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=base_index,
                                            function=DeltaFunctionCall(
                                                arguments=delta_text
                                            ),
                                        )
                                    ]
                                )
                            else:
                                delta_message = None

                            if delta_message is not None:
                                harmony_tools_streamed[i] = True
                        else:
                            delta_message = None
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        if (
                            self.reasoning_parser
                            and not reasoning_end_arr[i]
                            and not reasoning_parser.is_reasoning_end(
                                previous_token_ids
                            )
                        ):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )
                            # When encountering think end id in delta_token_ids
                            # or think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning'.
                            if reasoning_parser.is_reasoning_end(
                                as_list(output.token_ids)
                            ) or (
                                res.prompt_token_ids
                                and reasoning_parser.is_reasoning_end(
                                    res.prompt_token_ids
                                )
                            ):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(arguments=delta_text),
                                    index=i,
                                )
                            else:
                                delta_tool_call = DeltaToolCall(
                                    id=make_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text,
                                    ),
                                    index=i,
                                )
                                function_name_returned[i] = True

                            delta_message = DeltaMessage(
                                tool_calls=[
                                    delta_tool_call,
                                ]
                            )
                            tools_streamed[i] = True

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]
                        output_token_ids = as_list(output.token_ids)

                        if (
                            self.reasoning_parser is not None
                            and not reasoning_end_arr[i]
                            and res.prompt_token_ids
                            and reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                        ):
                            reasoning_end_arr[i] = True

                        if self.reasoning_parser and not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                            )
                            if reasoning_parser.is_reasoning_end(output_token_ids):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    # reasoning ended
                                    current_text = ""

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text

                            delta_message, function_name_returned[i] = (
                                self.extract_tool_call_required_streaming(
                                    previous_text=previous_text,
                                    current_text=content,
                                    delta_text=delta_text,
                                    function_name_returned=fn_name_returned,
                                    tool_call_idx=history_tool_call_cnt,
                                )
                            )
                            if (
                                delta_message
                                and delta_message.tool_calls
                                and delta_message.tool_calls[0].id is not None
                            ):
                                history_tool_call_cnt += 1
                                tools_streamed[i] = True

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.reasoning_parser:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                            )
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning'.
                            if (
                                res.prompt_token_ids
                                and reasoning_parser.is_reasoning_end(
                                    res.prompt_token_ids
                                )
                            ):
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                # Reasoning already ended in prompt, so first output is content
                                # Keep delta_message.content for output and track in current_text
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                else:
                                    current_text = ""
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning'.
                            if reasoning_parser.is_reasoning_end(output_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids = (
                                    reasoning_parser.extract_content_ids(
                                        output_token_ids
                                    )
                                )
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=output.token_ids,
                            request=request,
                        )
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # when only reasoning
                    elif self.reasoning_parser:
                        delta_message = reasoning_parser.extract_reasoning_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output.token_ids,
                        )
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (
                        tool_choice_auto or self.reasoning_parser
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        if output.finish_reason is None:
                            continue
                        else:
                            delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content = ""
                        if delta_message.content:
                            delta_content = delta_message.content
                        elif delta_message.tool_calls:
                            delta_content = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )

                        if delta_content:
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        else:
                            index = 0

                        if (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                            and tool_parser
                        ):
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}
                                ),
                                ensure_ascii=False,
                            )

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=index,
                                        function=DeltaFunctionCall(
                                            arguments=remaining_call
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if (
                            auto_tools_called
                            or (tools_streamed[i] and not tool_choice_function_name)
                            or (self.use_harmony and harmony_tools_streamed[i])
                        ):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning = None

                if self.tool_parser is not None:
                    tool_parser = self.tool_parser(tokenizer)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                        content_parts=tool_call_info.content_parts,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (tool_call_info is not None and tool_call_info.tools_called)
                        else output.finish_reason
                        if output.finish_reason
                        else "stop"
                    ),
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)
                continue

            if self.reasoning_parser:
                try:
                    reasoning_parser = self.reasoning_parser(
                        tokenizer,
                        chat_template_kwargs=request.chat_template_kwargs,  # type: ignore
                    )
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            tool_call_class = (
                MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall
            )
            if (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # if the request uses tools and specified a tool choice
            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                assert tool_calls is not None and len(tool_calls) > 0
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content="",
                    tool_calls=[tool_call_class(function=tc) for tc in tool_calls],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class_items = []
                assert tool_calls is not None and len(tool_calls) > 0
                for tool_call in tool_calls:
                    tool_call_class_items.append(
                        tool_call_class(
                            id=make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tool_call.name,
                                idx=history_tool_call_cnt,
                            ),
                            function=tool_call,
                        )
                    )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=tool_call_class_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    # Omni: Extract content_parts for multimodal ordering
                    content_parts = None
                    try:
                        tool_parser_instance = self.tool_parser(tokenizer)
                        # Re-parse to get content_parts
                        tool_call_info = tool_parser_instance.extract_tool_calls(
                            output.text, request=request
                        )
                        if tool_call_info and tool_call_info.content_parts:
                            content_parts = tool_call_info.content_parts
                    except Exception:
                        pass  # content_parts is optional
                    
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=[
                            ToolCall(
                                function=tc,
                                type="function",
                            )
                            for tc in tool_calls
                        ],
                        content_parts=content_parts,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if content and len(content) > 0:
                        ret_content = content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
            )

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:
                        if hasattr(tc.function, "name") and hasattr(
                            tc.function, "arguments"
                        ):
                            tool_call_descriptions.append(
                                f"{tc.function.name}({tc.function.arguments})"
                            )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: AnyTokenizer,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.tool_parser
            and delta_message
            and delta_message.tool_calls
            and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
    ):
        messages: list[OpenAIMessage] = []

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None,
            with_custom_tools=request.tools is not None,
        )
        messages.append(sys_msg)

        # Add developer message.
        dev_msg = get_developer_message(tools=request.tools)
        messages.append(dev_msg)

        # Add user message.
        for chat_msg in request.messages:
            messages.extend(parse_input_to_harmony_message(chat_msg))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [prompt_token_ids], [engine_prompt]
