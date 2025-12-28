#!/bin/bash
# =============================================================================
# 배포 검증 통합 스크립트
# =============================================================================
#
# Track A와 Track B의 모든 테스트를 실행하여 배포 상태를 검증합니다.
#
# 사용법:
#   ./deployment_verification.sh
#   OMNI_CHAINER_ENDPOINT="https://your-endpoint.com" ./deployment_verification.sh
#   VERBOSE=true ./deployment_verification.sh
#
# =============================================================================

set -o pipefail

# =============================================================================
# 설정
# =============================================================================
OMNI_CHAINER_ENDPOINT="${OMNI_CHAINER_ENDPOINT:-http://localhost:8000}"
TRACK_A_MODEL="${TRACK_A_MODEL:-track_a_model}"
TRACK_B_MODEL="${TRACK_B_MODEL:-track_b_model}"
VERBOSE="${VERBOSE:-false}"
MIN_RETRIES=5

# 테스트 결과 저장
declare -a TEST_RESULTS=()
declare -a FAILED_TESTS=()
declare -a FAILED_ERRORS=()
declare -a FAILED_COMMANDS=()

TOTAL_TESTS=0
PASSED_TESTS=0

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 유틸리티 함수
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Verbose 출력 함수 - JSON을 예쁘게 포맷팅
verbose_response() {
    local label="$1"
    local response="$2"

    if [ "$VERBOSE" != "true" ]; then
        return
    fi

    echo ""
    echo -e "   ${BLUE}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "   ${BLUE}│${NC} ${YELLOW}$label${NC}"
    echo -e "   ${BLUE}├─────────────────────────────────────────────────────────────┤${NC}"

    # JSON인지 확인하고 예쁘게 출력
    if echo "$response" | jq . > /dev/null 2>&1; then
        echo "$response" | jq -C '.' 2>/dev/null | sed 's/^/   │ /'
    else
        # JSON이 아니면 그대로 출력
        echo "$response" | sed 's/^/   │ /'
    fi

    echo -e "   ${BLUE}└─────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

# Verbose 에러 출력 함수
verbose_error() {
    local response="$1"

    if [ "$VERBOSE" != "true" ]; then
        return
    fi

    echo ""
    echo -e "   ${RED}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "   ${RED}│${NC} ${RED}ERROR RESPONSE${NC}"
    echo -e "   ${RED}├─────────────────────────────────────────────────────────────┤${NC}"

    # JSON인지 확인하고 예쁘게 출력
    if echo "$response" | jq . > /dev/null 2>&1; then
        echo "$response" | jq -C '.' 2>/dev/null | sed 's/^/   │ /'
    else
        echo "$response" | sed 's/^/   │ /'
    fi

    echo -e "   ${RED}└─────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

record_result() {
    local test_name="$1"
    local result="$2"
    local error_msg="$3"
    local curl_cmd="$4"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$result" == "PASS" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        TEST_RESULTS+=("$test_name: PASS")
    else
        TEST_RESULTS+=("$test_name: FAIL")
        FAILED_TESTS+=("$test_name")
        FAILED_ERRORS+=("$error_msg")
        FAILED_COMMANDS+=("$curl_cmd")
    fi
}

# =============================================================================
# 헬스체크
# =============================================================================
check_health() {
    log_info "서버 상태 확인 중..."
    if ! curl -s --max-time 10 "${OMNI_CHAINER_ENDPOINT}/health" > /dev/null 2>&1; then
        log_fail "omni-chainer 연결 실패: ${OMNI_CHAINER_ENDPOINT}"
        echo ""
        echo "omni-chainer가 실행 중인지 확인하세요."
        exit 1
    fi
    log_success "omni-chainer: OK"
    echo ""
}

# =============================================================================
# Track A 테스트 함수들
# =============================================================================

# Track A: 이미지 입력 테스트
test_track_a_image_input() {
    local test_name="Track A - Image Input"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local INPUT_IMAGE_URL="https://www.w3schools.com/css/img_5terre.jpg"
    local PROMPT="이 사진에 무엇이 있나요? 한국어로 간단히 설명해주세요."
    local MAX_TOKENS=512
    local TEMPERATURE=0.7

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_A_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"${INPUT_IMAGE_URL}\"
            }
          },
          {
            \"type\": \"text\",
            \"text\": \"${PROMPT}\"
          }
        ]
      }
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"extra_body\": {
      \"chat_template_kwargs\": {
        \"thinking\": false
      }
    }
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_A_MODEL"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {
                "role": "user",
                "content": [
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": "'"$INPUT_IMAGE_URL"'"
                    }
                  },
                  {
                    "type": "text",
                    "text": "'"$PROMPT"'"
                  }
                ]
              }
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "extra_body": {
              "chat_template_kwargs": {
                "thinking": false
              }
            }
          }')

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error=$(echo "$RESPONSE" | jq -r '.error.message // "응답 없음"' 2>/dev/null)
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track A: Non-Reasoning 테스트
test_track_a_non_reasoning() {
    local test_name="Track A - Non-Reasoning"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local PROMPT="다음 수학 문제를 풀어주세요: 3x + 7 = 22일 때, x의 값은?"
    local MAX_TOKENS=512
    local TEMPERATURE=0.7

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_A_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"${PROMPT}\"}
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"extra_body\": {
      \"chat_template_kwargs\": {
        \"thinking\": false
      }
    }
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_A_MODEL"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "extra_body": {
              "chat_template_kwargs": {
                "thinking": false
              }
            }
          }')

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error=$(echo "$RESPONSE" | jq -r '.error.message // "응답 없음"' 2>/dev/null)
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track A: Reasoning 테스트
test_track_a_reasoning() {
    local test_name="Track A - Reasoning"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local PROMPT="다음 수학 문제를 풀어주세요: 3x + 7 = 22일 때, x의 값은?"
    local MAX_TOKENS=1024
    local TEMPERATURE=0.7
    local THINKING_TOKEN_BUDGET=50

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_A_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"${PROMPT}\"}
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"extra_body\": {
      \"thinking_token_budget\": ${THINKING_TOKEN_BUDGET},
      \"chat_template_kwargs\": {
        \"thinking\": true
      }
    }
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/a/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_A_MODEL"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "extra_body": {
              "thinking_token_budget": '"$THINKING_TOKEN_BUDGET"',
              "chat_template_kwargs": {
                "thinking": true
              }
            }
          }')

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error=$(echo "$RESPONSE" | jq -r '.error.message // "응답 없음"' 2>/dev/null)
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# =============================================================================
# Track B 테스트 함수들
# =============================================================================

# Track B: 오디오 입력 테스트
test_track_b_audio_input() {
    local test_name="Track B - Audio Input"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local AUDIO_URL="${TEST_AUDIO_URL:-https://download.samplelib.com/mp3/sample-3s.mp3}"
    local PROMPT="이 오디오에서 무슨 말을 하나요?"
    local MAX_TOKENS=256
    local TEMPERATURE=0.5

    # 오디오 URL을 base64로 인코딩
    local AUDIO_DATA=$(echo -n "$AUDIO_URL" | base64 | tr -d '\n')

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"input_audio\",
            \"input_audio\": {
              \"data\": \"<base64_encoded_audio_url>\",
              \"format\": \"mp3\"
            }
          },
          {
            \"type\": \"text\",
            \"text\": \"${PROMPT}\"
          }
        ]
      }
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_B_MODEL"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {
                "role": "user",
                "content": [
                  {
                    "type": "input_audio",
                    "input_audio": {
                      "data": "'"$AUDIO_DATA"'",
                      "format": "mp3"
                    }
                  },
                  {
                    "type": "text",
                    "text": "'"$PROMPT"'"
                  }
                ]
              }
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "chat_template_kwargs": {"skip_reasoning": true}
          }')

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error=$(echo "$RESPONSE" | jq -r '.error.message // "응답 없음"' 2>/dev/null)
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 이미지 입력 테스트
test_track_b_image_input() {
    local test_name="Track B - Image Input"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local INPUT_IMAGE_URL="https://www.w3schools.com/css/img_5terre.jpg"
    local PROMPT="이 사진에 무엇이 있나요? 한국어로 간단히 설명해주세요."
    local MAX_TOKENS=256
    local TEMPERATURE=0.5

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"${INPUT_IMAGE_URL}\"
            }
          },
          {
            \"type\": \"text\",
            \"text\": \"${PROMPT}\"
          }
        ]
      }
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_B_MODEL"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {
                "role": "user",
                "content": [
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": "'"$INPUT_IMAGE_URL"'"
                    }
                  },
                  {
                    "type": "text",
                    "text": "'"$PROMPT"'"
                  }
                ]
              }
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "chat_template_kwargs": {"skip_reasoning": true}
          }')

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error=$(echo "$RESPONSE" | jq -r '.error.message // "응답 없음"' 2>/dev/null)
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 이미지 출력 테스트
test_track_b_image_output() {
    local test_name="Track B - Image Output"
    local max_retries=10  # 원본 스크립트의 MAX_RETRIES 유지
    local last_error=""

    local PROMPT="강아지 그림을 그려줘"
    local MAX_TOKENS=7000
    local TEMPERATURE=0.7

    local SYSTEM_PROMPT="You are an AI assistant that generates images. When asked to draw or create an image, you MUST use the t2i_model_generation tool to generate the image. Always respond by calling the tool."

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"${SYSTEM_PROMPT}\"},
      {\"role\": \"user\", \"content\": \"${PROMPT}\"}
    ],
    \"tools\": [
      {
        \"type\": \"function\",
        \"function\": {
          \"name\": \"t2i_model_generation\",
          \"description\": \"Generates an RGB image based on the provided discrete image representation.\",
          \"parameters\": {
            \"type\": \"object\",
            \"required\": [\"discrete_image_token\"],
            \"properties\": {
              \"discrete_image_token\": {
                \"type\": \"string\",
                \"description\": \"A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.\",
                \"minLength\": 1
              }
            }
          }
        }
      }
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_B_MODEL"'",
            "messages": [
              {"role": "system", "content": "'"$SYSTEM_PROMPT"'"},
              {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "tools": [
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
                        "description": "A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.",
                        "minLength": 1
                      }
                    }
                  }
                }
              }
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "chat_template_kwargs": {"skip_reasoning": true}
          }')

        # 에러 체크
        if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
            last_error=$(echo "$RESPONSE" | jq -r '.error.message' 2>/dev/null)
            verbose_error "$RESPONSE"
            continue
        fi

        # tool_calls 확인
        TOKEN=$(echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.arguments' 2>/dev/null | jq -r '.discrete_image_token' 2>/dev/null)

        # S3 URL이면 성공
        if [[ "$TOKEN" == s3://* ]] || [[ "$TOKEN" == http* ]]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (이미지 URL: ${TOKEN})" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        # Vision tokens 생성 확인 (디코더 처리 대기)
        if [[ "$TOKEN" == *"discrete_image_start"* ]]; then
            [ "$VERBOSE" = "true" ] && echo "   Vision tokens 생성됨, Decoder 처리 중..."
            continue
        fi

        last_error="tool_call 없음 또는 예상치 못한 응답"
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 이미지→이미지 테스트
test_track_b_image_to_image() {
    local test_name="Track B - Image to Image"
    local max_retries=10  # 원본 스크립트의 MAX_RETRIES 유지
    local last_error=""

    local INPUT_IMAGE_URL="https://www.w3schools.com/css/img_5terre.jpg"
    local PROMPT="Transform this image into cartoon style"
    local MAX_TOKENS=7000
    local TEMPERATURE=0.5

    local SYSTEM_PROMPT="You are an AI assistant that transforms images. When asked to transform, edit, or stylize an image, you MUST use the t2i_model_generation tool to generate the new image. Always respond by calling the tool."

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"${SYSTEM_PROMPT}\"},
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"${INPUT_IMAGE_URL}\"
            }
          },
          {
            \"type\": \"text\",
            \"text\": \"${PROMPT}\"
          }
        ]
      }
    ],
    \"tools\": [
      {
        \"type\": \"function\",
        \"function\": {
          \"name\": \"t2i_model_generation\",
          \"description\": \"Generates an RGB image based on the provided discrete image representation.\",
          \"parameters\": {
            \"type\": \"object\",
            \"required\": [\"discrete_image_token\"],
            \"properties\": {
              \"discrete_image_token\": {
                \"type\": \"string\",
                \"description\": \"A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.\",
                \"minLength\": 1
              }
            }
          }
        }
      }
    ],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"$TRACK_B_MODEL"'",
            "messages": [
              {"role": "system", "content": "'"$SYSTEM_PROMPT"'"},
              {
                "role": "user",
                "content": [
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": "'"$INPUT_IMAGE_URL"'"
                    }
                  },
                  {
                    "type": "text",
                    "text": "'"$PROMPT"'"
                  }
                ]
              }
            ],
            "tools": [
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
                        "description": "A serialized string of discrete vision tokens, encapsulated by special tokens. The format must be strictly followed: <|discrete_image_start|><|vision_ratio_4:3|><|vision_token|><|visionaaaaa|><|visionbbbbb|>... <|visionzzzzz|><|vision_eol|><|vision_eof|><|discrete_image_end|>.",
                        "minLength": 1
                      }
                    }
                  }
                }
              }
            ],
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"',
            "chat_template_kwargs": {"skip_reasoning": true}
          }')

        # 에러 체크
        if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
            last_error=$(echo "$RESPONSE" | jq -r '.error.message' 2>/dev/null)
            verbose_error "$RESPONSE"
            continue
        fi

        # tool_calls 확인
        TOKEN=$(echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.arguments' 2>/dev/null | jq -r '.discrete_image_token' 2>/dev/null)

        # S3 URL이면 성공
        if [[ "$TOKEN" == s3://* ]] || [[ "$TOKEN" == http* ]]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (이미지 URL: ${TOKEN})" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        # Vision tokens 생성 확인
        if [[ "$TOKEN" == *"discrete_image_start"* ]]; then
            [ "$VERBOSE" = "true" ] && echo "   Vision tokens 생성됨, Decoder 처리 중..."
            continue
        fi

        last_error="tool_call 없음 또는 예상치 못한 응답"
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 비디오 입력 테스트
test_track_b_video_input() {
    local test_name="Track B - Video Input"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local VIDEO_URL="https://download.samplelib.com/mp4/sample-5s.mp4"
    local PROMPT="이 영상에서 무엇이 보이나요? 자세히 설명해주세요."
    local MAX_TOKENS=512
    local TEMPERATURE=0.5

    local curl_cmd="curl -s --max-time 180 \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {
          \"type\": \"image_url\",
          \"image_url\": {
            \"url\": \"${VIDEO_URL}\"
          }
        },
        {
          \"type\": \"text\",
          \"text\": \"${PROMPT}\"
        }
      ]
    }],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작... (비디오 인코딩에 시간이 걸릴 수 있습니다)"

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        # jq를 사용하여 JSON을 안전하게 생성
        JSON_PAYLOAD=$(jq -n \
          --arg model "$TRACK_B_MODEL" \
          --arg video_url "$VIDEO_URL" \
          --arg prompt "$PROMPT" \
          --argjson max_tokens "$MAX_TOKENS" \
          --argjson temperature "$TEMPERATURE" \
          '{
            model: $model,
            messages: [{
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: $video_url
                  }
                },
                {
                  type: "text",
                  text: $prompt
                }
              ]
            }],
            max_tokens: $max_tokens,
            temperature: $temperature,
            chat_template_kwargs: {skip_reasoning: true}
          }')

        RESPONSE=$(curl -s --max-time 180 "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "$JSON_PAYLOAD" 2>&1)

        # 에러 체크
        if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
            last_error=$(echo "$RESPONSE" | jq -r '.error.message' 2>/dev/null)
            verbose_error "$RESPONSE"
            continue
        fi

        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ -n "$CONTENT" ] && [ "$CONTENT" != "null" ]; then
            log_success "[$test_name] 성공!"
            verbose_response "응답 (Response)" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error="응답 없음"
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 오디오 출력 테스트
test_track_b_audio_output() {
    local test_name="Track B - Audio Output"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local PROMPT="입력 텍스트를 40대 여성의 슬픈 말투로 말해줘.\n벌레를 갖다 주었다고 우쭐대다니, 참 이상해."
    local MAX_TOKENS=1000
    local TEMPERATURE=0.7

    local curl_cmd="curl -s \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": \"${PROMPT}\"
    }],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        # jq를 사용하여 JSON을 안전하게 생성
        JSON_PAYLOAD=$(jq -n \
          --arg model "$TRACK_B_MODEL" \
          --arg prompt "$PROMPT" \
          --argjson max_tokens "$MAX_TOKENS" \
          --argjson temperature "$TEMPERATURE" \
          '{
            model: $model,
            messages: [{
              role: "user",
              content: $prompt
            }],
            max_tokens: $max_tokens,
            temperature: $temperature,
            chat_template_kwargs: {skip_reasoning: true}
          }')

        RESPONSE=$(curl -s "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "$JSON_PAYLOAD" 2>&1)

        # 에러 체크
        if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
            last_error=$(echo "$RESPONSE" | jq -r '.error.message' 2>/dev/null)
            verbose_error "$RESPONSE"
            continue
        fi

        # audio.data 확인
        AUDIO_DATA=$(echo "$RESPONSE" | jq -r '.choices[0].message.audio.data // empty' 2>/dev/null)

        if [ -n "$AUDIO_DATA" ] && [ "$AUDIO_DATA" != "null" ]; then
            log_success "[$test_name] 성공!"
            AUDIO_URL=$(echo "$AUDIO_DATA" | base64 -d 2>/dev/null)
            verbose_response "응답 (Audio URL: ${AUDIO_URL})" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error="audio 필드 없음"
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# Track B: 오디오→오디오 테스트
test_track_b_audio_to_audio() {
    local test_name="Track B - Audio to Audio"
    local max_retries=$MIN_RETRIES
    local last_error=""

    local INPUT_AUDIO_URL="https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=ko&q=%EC%98%A4%EB%8A%98%20%EB%82%A0%EC%94%A8%EB%8A%94%20%EC%96%B4%EB%95%8C"
    local INPUT_AUDIO_FORMAT="mp3"
    local PROMPT="이 음성을 듣고 음성으로 대답해줘"
    local MAX_TOKENS=2000
    local TEMPERATURE=0.7

    local curl_cmd="curl -s --max-time 120 \"${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{
    \"model\": \"${TRACK_B_MODEL}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {
          \"type\": \"input_audio\",
          \"input_audio\": {
            \"data\": \"${INPUT_AUDIO_URL}\",
            \"format\": \"${INPUT_AUDIO_FORMAT}\"
          }
        },
        {
          \"type\": \"text\",
          \"text\": \"${PROMPT}\"
        }
      ]
    }],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": ${TEMPERATURE},
    \"chat_template_kwargs\": {\"skip_reasoning\": true}
  }'"

    log_info "[$test_name] 테스트 시작..."

    for attempt in $(seq 1 $max_retries); do
        [ "$VERBOSE" = "true" ] && echo "   시도 $attempt/$max_retries..."

        # jq를 사용하여 JSON을 안전하게 생성
        JSON_PAYLOAD=$(jq -n \
          --arg model "$TRACK_B_MODEL" \
          --arg audio_url "$INPUT_AUDIO_URL" \
          --arg audio_format "$INPUT_AUDIO_FORMAT" \
          --arg prompt "$PROMPT" \
          --argjson max_tokens "$MAX_TOKENS" \
          --argjson temperature "$TEMPERATURE" \
          '{
            model: $model,
            messages: [{
              role: "user",
              content: [
                {
                  type: "input_audio",
                  input_audio: {
                    data: $audio_url,
                    format: $audio_format
                  }
                },
                {
                  type: "text",
                  text: $prompt
                }
              ]
            }],
            max_tokens: $max_tokens,
            temperature: $temperature,
            chat_template_kwargs: {skip_reasoning: true}
          }')

        RESPONSE=$(curl -s --max-time 120 "${OMNI_CHAINER_ENDPOINT}/b/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "$JSON_PAYLOAD" 2>&1)

        # 에러 체크
        if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
            last_error=$(echo "$RESPONSE" | jq -r '.error.message' 2>/dev/null)
            verbose_error "$RESPONSE"
            continue
        fi

        # audio.data 확인
        AUDIO_DATA=$(echo "$RESPONSE" | jq -r '.choices[0].message.audio.data // empty' 2>/dev/null)

        if [ -n "$AUDIO_DATA" ] && [ "$AUDIO_DATA" != "null" ]; then
            log_success "[$test_name] 성공!"
            AUDIO_URL=$(echo "$AUDIO_DATA" | base64 -d 2>/dev/null)
            verbose_response "응답 (Audio URL: ${AUDIO_URL})" "$RESPONSE"
            record_result "$test_name" "PASS" "" ""
            return 0
        fi

        last_error="audio 필드 없음"
        verbose_error "$RESPONSE"
    done

    log_fail "[$test_name] $max_retries회 시도 후 실패"
    record_result "$test_name" "FAIL" "$last_error" "$curl_cmd"
    return 1
}

# =============================================================================
# 최종 요약 출력
# =============================================================================
print_summary() {
    echo ""
    echo "=============================================="
    echo "   배포 검증 결과 요약"
    echo "=============================================="
    echo ""
    echo "Endpoint: $OMNI_CHAINER_ENDPOINT"
    echo "Track A Model: $TRACK_A_MODEL"
    echo "Track B Model: $TRACK_B_MODEL"
    echo ""
    echo "----------------------------------------------"
    echo "테스트 결과: $PASSED_TESTS / $TOTAL_TESTS 성공"
    echo "----------------------------------------------"
    echo ""

    for result in "${TEST_RESULTS[@]}"; do
        if [[ "$result" == *"PASS"* ]]; then
            echo -e "  ${GREEN}[PASS]${NC} ${result%%:*}"
        else
            echo -e "  ${RED}[FAIL]${NC} ${result%%:*}"
        fi
    done

    echo ""

    if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
        echo "=============================================="
        echo -e "  ${GREEN}SUCCESS: 모든 배포 검증이 완료되었습니다!${NC}"
        echo "=============================================="
        exit 0
    else
        echo "=============================================="
        echo -e "  ${RED}FAILED: 일부 테스트가 실패했습니다${NC}"
        echo "=============================================="
        echo ""
        echo "실패한 테스트 상세 정보:"
        echo ""

        for i in "${!FAILED_TESTS[@]}"; do
            echo "----------------------------------------------"
            echo -e "${RED}[${FAILED_TESTS[$i]}]${NC}"
            echo ""
            echo "에러 원인:"
            echo "  ${FAILED_ERRORS[$i]}"
            echo ""
            echo "재현 방법:"
            echo "  ${FAILED_COMMANDS[$i]}"
            echo ""
        done

        echo "----------------------------------------------"
        echo ""
        echo "확인사항:"
        echo "  1. omni-chainer가 정상적으로 실행 중인지 확인"
        echo "  2. Track A/B vLLM 서버가 연결되어 있는지 확인"
        echo "  3. Encoder/Decoder 서비스가 정상인지 확인"
        echo "  4. 네트워크 연결 상태 확인"
        echo ""
        exit 1
    fi
}

# =============================================================================
# 메인 실행
# =============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "   배포 검증 통합 테스트"
    echo "=============================================="
    echo ""
    echo "Endpoint: $OMNI_CHAINER_ENDPOINT"
    echo "Track A Model: $TRACK_A_MODEL"
    echo "Track B Model: $TRACK_B_MODEL"
    echo "Verbose: $VERBOSE"
    echo ""

    # 헬스체크
    check_health

    # Track A 테스트
    echo "=============================================="
    echo "   Track A 테스트"
    echo "=============================================="
    echo ""

    test_track_a_image_input
    test_track_a_non_reasoning
    test_track_a_reasoning

    echo ""

    # Track B 테스트
    echo "=============================================="
    echo "   Track B 테스트"
    echo "=============================================="
    echo ""

    test_track_b_audio_input
    test_track_b_image_input
    test_track_b_image_output
    test_track_b_image_to_image
    test_track_b_video_input
    test_track_b_audio_output
    test_track_b_audio_to_audio

    # 최종 요약
    print_summary
}

# 스크립트 실행
main "$@"
