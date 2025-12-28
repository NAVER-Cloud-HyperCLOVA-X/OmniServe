# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Omni Model Tool Parser - Vision/Audio 토큰을 Tool Call로 변환

실제 모델 출력 형식:

1. Audio 출력:
    <user_transcript>사용자 발화 텍스트</user_transcript>
    <assistant_transcript>어시스턴트 응답 텍스트</assistant_transcript>
    <audio_decoder_call>
    <speaker_info>{"speaker": "mltm", "gender": "male", "emotion": "neutral"}</speaker_info>
    <discrete_audio><|DISCRETE_AUDIO_PAD|>...<|DISCRETE_AUDIO_PAD|></discrete_audio>
    </audio_decoder_call>

2. Vision 출력:
    텍스트 응답
    <tool_call>t2i_model_generation
    <arg_key>discrete_image_token</arg_key>
    <arg_value>
    <|discrete_image_start|><|DISCRETE_IMAGE_PAD|>...<|DISCRETE_IMAGE_PAD|><|discrete_image_end|>
    </arg_value>
    </tool_call>

사용법:
    python -m vllm.entrypoints.openai.api_server \
        --model <model_path> \
        --enable-auto-tool-choice \
        --tool-call-parser omni
"""

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module(["omni", "omni_multimodal"])
class OmniToolParser(ToolParser):
    """
    Omni 모델용 Tool Parser
    
    Vision과 Audio 출력을 파싱하여 OpenAI 호환 tool_calls 형태로 변환합니다.
    
    지원하는 출력 패턴:
    
    1. Audio (음성 합성):
       - <audio_decoder_call> 블록 전체를 audio_synthesis tool call로 변환
       - <assistant_transcript>를 content로 반환
       - <speaker_info>와 <discrete_audio>를 arguments에 포함
    
    2. Vision (이미지 생성):
       - <tool_call>t2i_model_generation... 블록을 t2i_model_generation tool call로 변환
       - <arg_value> 내의 discrete_image_token을 arguments에 포함
    """
    
    # Audio decoder call 패턴
    AUDIO_DECODER_PATTERN = re.compile(
        r'<audio_decoder_call>\s*'
        r'<speaker_info>(.*?)</speaker_info>\s*'
        r'<discrete_audio>(.*?)</discrete_audio>\s*'
        r'</audio_decoder_call>',
        re.DOTALL
    )
    
    # User/Assistant transcript 패턴
    USER_TRANSCRIPT_PATTERN = re.compile(
        r'<user_transcript>(.*?)</user_transcript>',
        re.DOTALL
    )
    ASSISTANT_TRANSCRIPT_PATTERN = re.compile(
        r'<assistant_transcript>(.*?)</assistant_transcript>',
        re.DOTALL
    )
    
    # Vision tool call 패턴
    VISION_TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*t2i_model_generation\s*'
        r'<arg_key>\s*discrete_image_token\s*</arg_key>\s*'
        r'<arg_value>\s*(.*?)\s*</arg_value>\s*'
        r'</tool_call>',
        re.DOTALL
    )
    
    # 일반 tool call XML 패턴 (다른 tool call용)
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>(.*?)</tool_call>',
        re.DOTALL
    )
    
    # Discrete image 블록 패턴 (단독으로 있는 경우)
    DISCRETE_IMAGE_PATTERN = re.compile(
        r'<\|discrete_image_start\|>.*?<\|discrete_image_end\|>',
        re.DOTALL
    )
    
    # 레거시: Audio 토큰 패턴 (이전 형식 지원)
    LEGACY_AUDIO_TOKEN_PATTERN = re.compile(r'<\|audio(\d+)\|>')
    
    # 불완전한 Vision tool call 텍스트 패턴 (태그 없이 출력된 경우)
    # 예: "t2i_model_generation\ndiscrete_image_token\n\n<|discrete_image_start|>..."
    INCOMPLETE_VISION_TOOL_TEXT_PATTERN = re.compile(
        r't2i_model_generation\s*\n\s*discrete_image_token\s*\n*',
        re.IGNORECASE
    )
    
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_id = 0
        self.streamed_args_for_tool: list[str] = []
        
        # 스트리밍 상태
        self.audio_buffer: str = ""
        self.vision_buffer: str = ""
        self.in_audio_block = False
        self.in_vision_block = False
    
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """특수 토큰을 스킵하지 않도록 설정"""
        request = super().adjust_request(request)
        # Omni 모델의 special token을 보존해야 파싱 가능
        request.skip_special_tokens = False
        return request
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        """
        모델 출력에서 Vision/Audio 블록을 파싱하여 tool calls로 변환

        Args:
            model_output: 모델이 생성한 텍스트 출력
            request: ChatCompletionRequest 객체
            token_ids: 생성된 토큰 ID 시퀀스 (선택적)

        Returns:
            ExtractedToolCallInformation: 파싱된 tool calls와 텍스트 content
        """
        tool_calls: list[ToolCall] = []
        text_content = model_output

        # ID 매핑을 위한 딕셔너리 (discrete_image_token hash -> tool_call_id)
        token_to_id_map = {}

        # 1. Audio decoder call 처리
        audio_matches = self.AUDIO_DECODER_PATTERN.findall(model_output)
        for speaker_info_str, discrete_audio in audio_matches:
            try:
                speaker_info = json.loads(speaker_info_str.strip())
            except json.JSONDecodeError:
                speaker_info = {"raw": speaker_info_str.strip()}

            tool_call_id = f"call_audio_{random_uuid()[:8]}"
            tool_calls.append(ToolCall(
                id=tool_call_id,
                type="function",
                function=FunctionCall(
                    name="audio_synthesis",
                    arguments=json.dumps({
                        "speaker_info": speaker_info,
                        "discrete_audio": discrete_audio.strip(),
                    }, ensure_ascii=False)
                )
            ))
            logger.debug(f"Extracted audio decoder call, discrete_audio length: {len(discrete_audio)}")

        # Audio decoder call 블록 제거
        text_content = self.AUDIO_DECODER_PATTERN.sub('', text_content)

        # 2. Vision tool call 처리 (정규 형식 - </arg_value> 포함)
        vision_matches = self.VISION_TOOL_CALL_PATTERN.findall(model_output)
        vision_tool_call_found = len(vision_matches) > 0
        for discrete_image_token in vision_matches:
            # discrete_image_token의 해시값으로 고유 ID 생성
            token_hash = str(hash(discrete_image_token.strip()))[:16]
            tool_call_id = f"call_vision_{token_hash}"

            tool_calls.append(ToolCall(
                id=tool_call_id,
                type="function",
                function=FunctionCall(
                    name="t2i_model_generation",
                    arguments=json.dumps({
                        "discrete_image_token": discrete_image_token.strip()
                    }, ensure_ascii=False)
                )
            ))

            # 매핑 저장
            token_to_id_map[discrete_image_token.strip()] = tool_call_id
            logger.debug(f"Extracted vision tool call (regular), token length: {len(discrete_image_token)}, id: {tool_call_id}")

        # Vision tool call 블록 제거
        text_content = self.VISION_TOOL_CALL_PATTERN.sub('', text_content)

        # 3. 일반 tool call XML 처리 (모든 function 호출 처리)
        remaining_tool_calls = self.TOOL_CALL_PATTERN.findall(text_content)
        for tool_call_content in remaining_tool_calls:
            parsed_call = self._parse_xml_tool_call(tool_call_content)
            if parsed_call:
                tool_calls.append(parsed_call)
                logger.debug(f"Extracted tool call: {parsed_call.function.name}")

        # 일반 tool call 블록 제거
        text_content = self.TOOL_CALL_PATTERN.sub('', text_content)

        # 4. 단독 discrete_image 블록 처리 (tool_call 태그 없이 출력된 경우)
        standalone_images = self.DISCRETE_IMAGE_PATTERN.findall(text_content)
        for discrete_image in standalone_images:
            # discrete_image_token의 해시값으로 고유 ID 생성
            token_hash = str(hash(discrete_image.strip()))[:16]
            tool_call_id = f"call_vision_{token_hash}"

            tool_calls.append(ToolCall(
                id=tool_call_id,
                type="function",
                function=FunctionCall(
                    name="t2i_model_generation",
                    arguments=json.dumps({
                        "discrete_image_token": discrete_image.strip()
                    }, ensure_ascii=False)
                )
            ))

            # 매핑 저장
            token_to_id_map[discrete_image.strip()] = tool_call_id
            logger.debug(f"Extracted standalone discrete image, length: {len(discrete_image)}, id: {tool_call_id}")

        # 단독 discrete_image 블록 제거
        text_content = self.DISCRETE_IMAGE_PATTERN.sub('', text_content)

        # 5. 레거시 Audio 토큰 처리 (<|audio0000|> 형식)
        legacy_audio_matches = self.LEGACY_AUDIO_TOKEN_PATTERN.findall(text_content)
        if legacy_audio_matches:
            units = [int(t) for t in legacy_audio_matches]
            tool_call_id = f"call_audio_{random_uuid()[:8]}"
            tool_calls.append(ToolCall(
                id=tool_call_id,
                type="function",
                function=FunctionCall(
                    name="audio_synthesis",
                    arguments=json.dumps({
                        "units": units,
                        "format": "wav",
                        "token_count": len(units)
                    }, ensure_ascii=False)
                )
            ))
            logger.debug(f"Extracted {len(units)} legacy audio tokens")

        # 레거시 Audio 토큰 제거
        text_content = self.LEGACY_AUDIO_TOKEN_PATTERN.sub('', text_content)
        
        # 6. Transcript 추출 (content로 사용)
        # assistant_transcript가 있으면 그것을 주요 content로 사용
        assistant_transcripts = self.ASSISTANT_TRANSCRIPT_PATTERN.findall(text_content)
        if assistant_transcripts:
            # 모든 assistant_transcript를 합침
            transcript_content = '\n'.join(t.strip() for t in assistant_transcripts)
            # transcript 태그 제거
            text_content = self.ASSISTANT_TRANSCRIPT_PATTERN.sub('', text_content)
            text_content = self.USER_TRANSCRIPT_PATTERN.sub('', text_content)
            # transcript를 주요 content로 사용, 나머지 텍스트가 있으면 추가
            remaining = text_content.strip()
            if remaining:
                text_content = f"{transcript_content}\n\n{remaining}"
            else:
                text_content = transcript_content
        else:
            # transcript가 없으면 user_transcript만 제거
            text_content = self.USER_TRANSCRIPT_PATTERN.sub('', text_content)
        
        # 7. 텍스트 정리
        # <think></think> 블록 제거
        text_content = re.sub(r'<think>\s*</think>', '', text_content)
        text_content = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL)
        
        # 불완전한 Vision tool call 텍스트 제거 (태그 없이 출력된 경우)
        # 예: "t2i_model_generation\ndiscrete_image_token\n\n" 형태
        text_content = self.INCOMPLETE_VISION_TOOL_TEXT_PATTERN.sub('', text_content)
        
        # 불필요한 연속 공백/줄바꿈 정리
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        text_content = re.sub(r' {2,}', ' ', text_content)
        text_content = text_content.strip()
        
        # 8. content_parts 생성 (순서 보존, token_to_id_map 전달)
        content_parts = self._extract_content_parts_with_order(
            model_output, tool_calls, token_to_id_map
        ) if tool_calls else None

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls if tool_calls else [],
            content=text_content if text_content else None,
            content_parts=content_parts
        )
    
    def _extract_content_parts_with_order(
        self,
        model_output: str,
        tool_calls: list[ToolCall],
        token_to_id_map: dict
    ) -> list[dict]:
        """
        원본 model_output에서 순서를 보존하면서 content_parts 생성

        Args:
            model_output: 원본 모델 출력
            tool_calls: 파싱된 tool calls 리스트
            token_to_id_map: discrete_image_token -> tool_call_id 매핑

        Returns:
            [
                {"type": "text", "text": "..."},
                {"type": "tool_call_ref", "tool_call_id": "call_vision_xxx"},
                {"type": "text", "text": "..."},
            ]
        """
        # 모든 multimodal 블록의 위치를 찾음
        blocks = []

        # Vision tool call 패턴 (정규 형식)
        for match in self.VISION_TOOL_CALL_PATTERN.finditer(model_output):
            discrete_token = match.group(1).strip()
            blocks.append({
                "start": match.start(),
                "end": match.end(),
                "type": "vision",
                "content": discrete_token,
                "tool_call_id": token_to_id_map.get(discrete_token)  # 매핑된 ID 사용
            })
        
        # 단독 discrete_image 패턴
        for match in self.DISCRETE_IMAGE_PATTERN.finditer(model_output):
            # 이미 VISION_TOOL_CALL_PATTERN에서 처리된 것인지 확인
            already_matched = any(
                b["start"] <= match.start() < b["end"]
                for b in blocks if b["type"] == "vision"
            )
            if not already_matched:
                discrete_token = match.group(0).strip()
                blocks.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": "vision_standalone",
                    "content": discrete_token,
                    "tool_call_id": token_to_id_map.get(discrete_token)  # 매핑된 ID 사용
                })
        
        # Audio decoder call 패턴
        for match in self.AUDIO_DECODER_PATTERN.finditer(model_output):
            blocks.append({
                "start": match.start(),
                "end": match.end(),
                "type": "audio",
            })
        
        # 레거시 Audio 토큰 패턴 - 연속된 토큰을 하나의 블록으로
        legacy_audio_matches = list(self.LEGACY_AUDIO_TOKEN_PATTERN.finditer(model_output))
        if legacy_audio_matches:
            # 첫 번째와 마지막 audio 토큰의 위치를 기준으로 블록 생성
            first_match = legacy_audio_matches[0]
            last_match = legacy_audio_matches[-1]
            blocks.append({
                "start": first_match.start(),
                "end": last_match.end(),
                "type": "audio_legacy",
            })
        
        # 일반 tool call 패턴
        for match in self.TOOL_CALL_PATTERN.finditer(model_output):
            # 이미 VISION_TOOL_CALL_PATTERN에서 처리된 것인지 확인
            already_matched = any(
                b["start"] <= match.start() < b["end"] 
                for b in blocks
            )
            if not already_matched:
                blocks.append({
                    "start": match.start(),
                    "end": match.end(),
                    "type": "tool_call",
                })
        
        # 위치 순으로 정렬
        blocks.sort(key=lambda x: x["start"])

        # tool_calls의 id를 type별로 매핑 (fallback용)
        vision_tool_ids = [tc.id for tc in tool_calls if tc.function.name == "t2i_model_generation"]
        audio_tool_ids = [tc.id for tc in tool_calls if tc.function.name == "audio_synthesis"]
        other_tool_ids = [tc.id for tc in tool_calls if tc.function.name not in ["t2i_model_generation", "audio_synthesis"]]

        vision_idx = 0
        audio_idx = 0
        other_idx = 0

        # content_parts 생성
        content_parts = []
        last_end = 0

        for block in blocks:
            # 블록 전 텍스트
            text_before = model_output[last_end:block["start"]]
            text_before = self._clean_text_for_content_parts(text_before)
            if text_before:
                content_parts.append({"type": "text", "text": text_before})

            # 블록 (tool_call_ref) - 매핑된 ID 우선 사용
            if block["type"] in ("vision", "vision_standalone"):
                tool_call_id = block.get("tool_call_id")
                if not tool_call_id and vision_idx < len(vision_tool_ids):
                    # fallback: 순서 기반 매칭
                    tool_call_id = vision_tool_ids[vision_idx]

                if tool_call_id:
                    content_parts.append({
                        "type": "tool_call_ref",
                        "tool_call_id": tool_call_id
                    })
                    vision_idx += 1
            elif block["type"] in ("audio", "audio_legacy") and audio_idx < len(audio_tool_ids):
                content_parts.append({
                    "type": "tool_call_ref",
                    "tool_call_id": audio_tool_ids[audio_idx]
                })
                audio_idx += 1
            elif block["type"] == "tool_call" and other_idx < len(other_tool_ids):
                content_parts.append({
                    "type": "tool_call_ref",
                    "tool_call_id": other_tool_ids[other_idx]
                })
                other_idx += 1

            last_end = block["end"]
        
        # 마지막 블록 이후 텍스트
        text_after = model_output[last_end:]
        text_after = self._clean_text_for_content_parts(text_after)
        if text_after:
            content_parts.append({"type": "text", "text": text_after})
        
        return content_parts if content_parts else None
    
    def _clean_text_for_content_parts(self, text: str) -> str:
        """content_parts용 텍스트 정리"""
        # <think> 블록 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>\s*</think>', '', text)
        # transcript 태그 제거
        text = self.USER_TRANSCRIPT_PATTERN.sub('', text)
        text = self.ASSISTANT_TRANSCRIPT_PATTERN.sub('', text)
        # 불완전한 vision tool call 텍스트 제거
        text = self.INCOMPLETE_VISION_TOOL_TEXT_PATTERN.sub('', text)
        # 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def _parse_xml_tool_call(self, tool_call_content: str) -> ToolCall | None:
        """
        XML 형태의 tool call을 파싱
        
        예시 1 (정상):
        <tool_call>function_name
        <arg_key>arg1</arg_key>
        <arg_value>value1</arg_value>
        </tool_call>
        
        예시 2 (불완전 - </arg_value> 누락):
        <tool_call>function_name
        <arg_key>arg1</arg_key>
        <arg_value>value1
        </tool_call>
        """
        try:
            lines = tool_call_content.strip().split('\n')
            if not lines:
                return None
            
            # 첫 줄에서 함수 이름 추출
            function_name = lines[0].strip()
            if not function_name:
                return None
            
            # arg_key/arg_value 쌍 추출
            arguments = {}
            
            # 방법 1: 정상 형식 (</arg_value> 있음)
            arg_key_pattern = re.compile(r'<arg_key>(.*?)</arg_key>')
            arg_value_pattern = re.compile(r'<arg_value>(.*?)</arg_value>', re.DOTALL)
            
            keys = arg_key_pattern.findall(tool_call_content)
            values = arg_value_pattern.findall(tool_call_content)
            
            if keys and values and len(keys) == len(values):
                for key, value in zip(keys, values):
                    arguments[key.strip()] = value.strip()
            elif keys:
                # 방법 2: 불완전 형식 (</arg_value> 누락)
                # <arg_value> 이후의 모든 내용을 value로 처리
                arg_value_open_pattern = re.compile(r'<arg_value>(.*)', re.DOTALL)
                values_incomplete = arg_value_open_pattern.findall(tool_call_content)
                
                for i, key in enumerate(keys):
                    if i < len(values_incomplete):
                        value = values_incomplete[i].strip()
                        # 다음 <arg_key>나 끝까지의 내용
                        if i + 1 < len(keys):
                            next_key_match = re.search(r'<arg_key>', value)
                            if next_key_match:
                                value = value[:next_key_match.start()].strip()
                        arguments[key.strip()] = value
            
            if not function_name:
                return None
            
            return ToolCall(
                id=f"call_{function_name}_{random_uuid()[:8]}",
                type="function",
                function=FunctionCall(
                    name=function_name,
                    arguments=json.dumps(arguments, ensure_ascii=False)
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse XML tool call: {e}")
        
        return None
    
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        스트리밍 모드에서 tool calls 추출
        
        스트리밍에서는 토큰이 하나씩 들어오므로, 버퍼링하여 처리합니다.
        """
        # Audio decoder call 시작 감지
        if "<audio_decoder_call>" in delta_text:
            self.in_audio_block = True
            self.audio_buffer = ""
        
        # Audio 블록 내용 수집
        if self.in_audio_block:
            self.audio_buffer += delta_text
            
            # Audio 블록 종료 감지
            if "</audio_decoder_call>" in self.audio_buffer:
                self.in_audio_block = False
                
                # Audio tool call 생성
                match = self.AUDIO_DECODER_PATTERN.search(self.audio_buffer)
                if match:
                    speaker_info_str, discrete_audio = match.groups()
                    try:
                        speaker_info = json.loads(speaker_info_str.strip())
                    except json.JSONDecodeError:
                        speaker_info = {"raw": speaker_info_str.strip()}
                    
                    self.current_tool_id += 1
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id - 1,
                                id=f"call_audio_{random_uuid()[:8]}",
                                type="function",
                                function=DeltaFunctionCall(
                                    name="audio_synthesis",
                                    arguments=json.dumps({
                                        "speaker_info": speaker_info,
                                        "discrete_audio": discrete_audio.strip(),
                                    }, ensure_ascii=False)
                                )
                            )
                        ]
                    )
            
            return None
        
        # Tool call 시작 감지
        if "<tool_call>" in delta_text:
            self.in_vision_block = True
            self.vision_buffer = ""
        
        # Vision tool call 내용 수집
        if self.in_vision_block:
            self.vision_buffer += delta_text
            
            # Tool call 종료 감지
            if "</tool_call>" in self.vision_buffer:
                self.in_vision_block = False
                
                # Vision tool call 확인
                match = self.VISION_TOOL_CALL_PATTERN.search(self.vision_buffer)
                if match:
                    discrete_image_token = match.group(1)
                    self.current_tool_id += 1
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id - 1,
                                id=f"call_vision_{random_uuid()[:8]}",
                                type="function",
                                function=DeltaFunctionCall(
                                    name="t2i_model_generation",
                                    arguments=json.dumps({
                                        "discrete_image_token": discrete_image_token.strip()
                                    }, ensure_ascii=False)
                                )
                            )
                        ]
                    )
                else:
                    # 다른 tool call 파싱
                    parsed = self._parse_xml_tool_call(
                        self.vision_buffer.replace("<tool_call>", "").replace("</tool_call>", "")
                    )
                    if parsed:
                        self.current_tool_id += 1
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id - 1,
                                    id=parsed.id,
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=parsed.function.name,
                                        arguments=parsed.function.arguments
                                    )
                                )
                            ]
                        )
            
            return None
        
        # 일반 텍스트는 content로 전달 (transcript 태그 등 제외)
        clean_delta = delta_text
        if '<' in clean_delta and '>' in clean_delta:
            # 태그가 포함된 경우 제거하지 않고 그대로 전달 (전체 파싱은 나중에)
            pass
        
        if clean_delta:
            return DeltaMessage(content=clean_delta)
        
        return None
