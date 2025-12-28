#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""
Chat Template 테스트 스크립트 (Track A - Image Only)
chat_template.jinja를 로드해서 Chat 메시지를 변환합니다.
Track A는 이미지 처리만 지원합니다.
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
        messages: 메시지 리스트
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


def test_case5_video_text():
    """Case 5: Video + Text → Text"""
    print("=" * 80)
    print("Test Case 5: Video + Text → Text")
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
                    "type": "text",
                    "text": "영상의 배경은 무엇인가요?"
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


def test_empty_content_with_tool_calls():
    """Test empty content with tool calls"""

    print("=" * 80)
    print("Test Case 6: empty assistant content with tool_calls")
    print("=" * 80)

    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "s3://<BUCKET>/source/derived/embedding/vision/test/image_example.pt",
              "detail": "auto"
            }
          },
          {
            "text": "\uc774 \uc0ac\ub78c\uc774 \ub204\uad6c\uc57c?",
            "type": "text"
          }
        ]
      },
      {
        "role": "user",
        "content": "\uc11c\uc6b8 \ub0a0\uc528 \uc54c\ub824\uc918"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_abc123",
            "function": {
              "arguments": "{\"location\":\"Seoul, KR\",\"unit\":\"celsius\"}",
              "name": "get_weather"
            },
            "type": "function"
          }
        ]
      },
      {
        "role": "tool",
        "content": "{\"temp_c\": 3, \"condition\": \"clear\"}",
        "name": "get_weather",
        "tool_call_id": "call_abc123"
      },
      {
        "role": "user",
        "content": "\ud55c \uc904\ub85c \uc694\uc57d\ud574\uc918"
      }
    ]

    result = apply_chat_template(messages, add_generation_prompt=True)
    print(result)


if __name__ == "__main__":
    try:
        # 이미지/비디오 입력 관련 테스트 실행 (멀티모달 출력은 불가능)
        test_case1_image_text()
        test_case5_video_text()
        test_multi_turn()
        test_empty_content_with_tool_calls()
        
        print("All tests completed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
