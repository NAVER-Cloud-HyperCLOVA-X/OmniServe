#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""
Chat Template 테스트 스크립트

chat_template.jinja를 로드해서 OpenAI Chat 메시지를 변환합니다.
"""

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import json


def load_chat_template(template_path="chat_template.jinja"):
    """chat_template.jinja 파일을 로드"""
    env = Environment(
        loader=FileSystemLoader("."),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path)


def apply_chat_template(messages, tools=None, add_generation_prompt=False):
    """
    Chat template을 적용하여 메시지를 변환
    
    Args:
        messages: OpenAI 형식의 메시지 리스트
        tools: 도구 정의 리스트 (선택)
        add_generation_prompt: 생성 프롬프트 추가 여부
    
    Returns:
        변환된 string
    """
    template = load_chat_template()
    
    # Jinja2 템플릿에 전달할 컨텍스트
    context = {
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
    }
    
    return template.render(**context)


def test_case1_image_text():
    """Case 1: Image + Text → Text"""
    print("=" * 80)
    print("Test Case 1: Image + Text → Text")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "s3://example-bucket/path/to/image.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "이 표정 어때?"
                }
            ]
        }
    ]
    
    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)
    print()


def test_case2_audio():
    """Case 2: Audio → Audio"""
    print("=" * 80)
    print("Test Case 2: Audio → Audio")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "<base64_encoded(s3://example-bucket/path/to/audio.wav)>",
                        "format": "wav"
                    }
                }
            ]
        }
    ]
    
    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)
    print()


def test_case3_with_tools():
    """Case 3: Image + Text → Image + Text (with tools)"""
    print("=" * 80)
    print("Test Case 3: Image + Text → Image + Text (with tools)")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "내 그림을 더 개선해 주고, 설명해봐."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "s3://example-bucket/images/sample1.png"
                    }
                }
            ]
        }
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "t2i_model_generation",
                "description": "Generates an RGB image based on the provided discrete image representation.",
                "parameters": {
                    "type": "object",
                    "required": ["discrete_image_token"],
                    "properties": {
                        "discrete_image_token": {
                            "type": "string",
                            "description": "A serialized string of discrete vision tokens.",
                            "minLength": 1
                        }
                    }
                }
            }
        }
    ]
    
    result = apply_chat_template(messages, tools=tools, add_generation_prompt=True)
    print(result)
    print()


def test_case2_1_2_text_audio():
    """Case 2.1.2: Text + Audio → Audio"""
    print("=" * 80)
    print("Test Case 2.1.2: Text + Audio → Audio")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "입력 음성을 40대 여성의 슬픈 말투로 바꿔줘."
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "<base64_encoded(s3://example-bucket/audio/original.wav)>",
                        "format": "wav"
                    }
                }
            ]
        }
    ]
    
    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)
    print()


def test_case5_video_audio():
    """Case 5: Video + Audio → Audio"""
    print("=" * 80)
    print("Test Case 5: Video + Audio → Audio")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "s3://example-bucket/video/sample.mp4"
                    }
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "<base64_encoded(s3://example-bucket/audio/sample.flac)>",
                        "format": "flac"
                    }
                }
            ]
        }
    ]
    
    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)
    print()


def test_multi_turn():
    """Multi-turn conversation"""
    print("=" * 80)
    print("Test Case: Multi-turn conversation")
    print("=" * 80)
    
    messages = [
        {
            "role": "system",
            "content": "System Prompt"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "s3://example-bucket/path/to/image.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "이 표정 어때?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "웃음이 맑고 좋네요."
        },
        {
            "role": "user",
            "content": "더 자세히 설명해줘."
        }
    ]
    
    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)
    print()


if __name__ == "__main__":
    try:
        # 모든 테스트 실행
        test_case1_image_text()
        test_case2_audio()
        test_case2_1_2_text_audio()
        test_case3_with_tools()
        test_case5_video_audio()
        test_multi_turn()
        
        print("All tests completed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
