# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from huggingface/transformers
# Source: https://github.com/huggingface/transformers
# License: Apache-2.0

import math
from typing import List, Optional, Union

import numpy as np
from transformers import Qwen2_5_VLProcessor
from transformers.image_processing_utils import (
    BatchFeature,
)
from transformers.image_transforms import (
    resize,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessorKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput
from typing_extensions import Unpack

from .qwen_vision_process import process_vision_info

logger = logging.get_logger(__name__)


def determine_possible_resolutions(anyres: bool, max_num_grids: int, grid_size: int, use_1x1_grid: bool = False):
    """Determine possible resolution combinations for anyres processing."""
    possible_resolutions = []
    if anyres:
        assert max_num_grids > 0
        for i in range(1, max_num_grids + 1):
            for j in range(1, max_num_grids + 1):
                if i == 1 and j == 1 and not use_1x1_grid:
                    continue
                if i * j <= max_num_grids:
                    possible_resolutions.append([i, j])

        possible_resolutions = [[ys * grid_size, xs * grid_size] for ys, xs in possible_resolutions]

    return possible_resolutions


def divide_to_grids(image: np.array, grid_size: int, input_data_format=None) -> List[np.array]:
    """Divide image into (grid_size x grid_size) grids."""
    grids = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            if input_data_format == ChannelDimension.LAST:
                grid = image[i : i + grid_size, j : j + grid_size]
            else:
                grid = image[:, i : i + grid_size, j : j + grid_size]
            grids.append(grid)

    return grids


def pad(image: np.array, target_size: tuple, background_color=(127, 127, 127), input_data_format=None) -> np.array:
    """Pad image to target size with background color."""
    target_height, target_width = target_size
    height, width = get_image_size(image, channel_dim=input_data_format)

    # result = np.ones((target_height, target_width, image.shape[2]), dtype=image.dtype) * background_color
    result = np.empty((target_height, target_width, image.shape[2]), dtype=image.dtype)
    for i in range(image.shape[2]):
        result[..., i].fill(background_color[i])

    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2

    result[paste_y : paste_y + height, paste_x : paste_x + width, :] = image

    return result


def expand2square(
    image: np.array, bboxes_dict=None, background_color=(127, 127, 127), input_data_format=None
) -> np.array:
    """Expand image to square by adding padding, centering the original image."""
    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image, bboxes_dict
    elif width > height:
        # result = np.ones((width, width, image.shape[2]), dtype=image.dtype) * background_color
        result = np.empty((width, width, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 1] += (width - height) // 2
        return result, bboxes_dict
    else:
        # result = np.ones((height, height, image.shape[2]), dtype=image.dtype) * background_color
        result = np.empty((height, height, image.shape[2]), dtype=image.dtype)
        for i in range(image.shape[2]):
            result[..., i].fill(background_color[i])

        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        if bboxes_dict is not None:
            for key in bboxes_dict:
                bboxes_dict[key][:, :, 0] += (height - width) // 2
        return result, bboxes_dict


def resize_longside(
    image: np.array,
    size: int,
    resample: PILImageResampling = PILImageResampling.BICUBIC,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """Resize image so that the longer side matches the target size."""
    height, width = get_image_size(image, channel_dim=input_data_format)

    if width == height:
        target_height, target_width = size, size
    elif width > height:
        target_width = size
        target_height = math.ceil(height / width * size)
    else:
        target_width = math.ceil(width / height * size)
        target_height = size

    return resize(
        image,
        size=(target_height, target_width),
        resample=resample,
        data_format=data_format,
        input_data_format=input_data_format,
    )


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """Select the best resolution from possible resolutions based on original size.

    Maximizes effective resolution and minimizes wasted resolution.
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


class HCXVisionV2Processor(Qwen2_5_VLProcessor):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast", "PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.tokenizer = tokenizer
        # tokenizer가 없어도 동작하도록 처리
        if tokenizer is not None:
            super().__init__(image_processor, tokenizer, video_processor, chat_template=self.tokenizer.chat_template)
        else:
            # tokenizer 없이 초기화 (이미지/비디오만 처리)
            # ProcessorMixin의 초기화를 우회하고 필요한 속성만 직접 설정
            self.image_token = "<|image_pad|>"
            self.video_token = "<|video_pad|>"
            self.image_token_id = None
            self.video_token_id = None
            # image_processor와 video_processor 속성 직접 설정
            self.image_processor = image_processor
            self.video_processor = video_processor
            # ProcessorMixin의 기본 속성 설정 (초기화 우회)
            # ProcessorMixin은 object를 상속받으므로 직접 초기화하지 않음
            # 필요한 속성만 설정
            if hasattr(super(), "__init__"):
                # Qwen2_5_VLProcessor의 부모 클래스 초기화는 tokenizer 필요하므로 우회
                pass

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """Process images, videos, and text for the model. Works without text for image/video-only processing."""
        # tokenizer가 없으면 텍스트 처리 없이 이미지/비디오만 처리
        if self.tokenizer is None:
            return_tensors = kwargs.get("return_tensors", None)
            all_data = {}

            if images is not None:
                # Qwen2VLImageProcessor에 return_tensors를 직접 전달
                # images_kwargs가 있으면 병합, 없으면 빈 dict 사용
                image_kwargs = kwargs.get("images_kwargs", {})
                image_kwargs = {**image_kwargs, "return_tensors": return_tensors}
                image_inputs = self.image_processor(images=images, **image_kwargs)
                # BatchFeature는 dict를 상속받으므로 직접 update 가능
                if isinstance(image_inputs, BatchFeature):
                    all_data.update(dict(image_inputs))
                elif isinstance(image_inputs, dict):
                    all_data.update(image_inputs)
                else:
                    all_data.update(image_inputs)

            if videos is not None:
                video_kwargs = kwargs.get("videos_kwargs", {})
                video_kwargs = {**video_kwargs, "return_tensors": return_tensors}
                videos_inputs = self.video_processor(videos=videos, **video_kwargs)
                # BatchFeature는 dict를 상속받으므로 직접 update 가능
                if isinstance(videos_inputs, BatchFeature):
                    all_data.update(dict(videos_inputs))
                elif isinstance(videos_inputs, dict):
                    all_data.update(videos_inputs)
                else:
                    all_data.update(videos_inputs)

            return BatchFeature(data=all_data, tensor_type=return_tensors)

        # tokenizer가 있으면 기존 로직 사용
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

        # 텍스트가 없으면 이미지/비디오만 반환
        if text is None:
            return_tensors = output_kwargs.get("text_kwargs", {}).get("return_tensors", None)
            return BatchFeature(data={**image_inputs, **videos_inputs}, tensor_type=return_tensors)

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    text[i] = text[i].replace(
                        '{"resolution": [w, h]}', '{"resolution": ' + str(list(images[i].size)) + "}"
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def process_vision_info(self, *args, **kwargs):
        return process_vision_info(*args, **kwargs)
