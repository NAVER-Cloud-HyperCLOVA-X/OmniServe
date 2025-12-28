#!/bin/bash

# ==========================================
# 설정: API 엔드포인트 및 공통 데이터
# ==========================================
API_URL="https://omni.ncloud.com/dev/b/audio/decoder/predictions"
UNITS="[1490, 1950, 1946, 1460, 753, 3184, 2384, 2855, 4880, 5186, 5410, 4644, 4397, 2330, 2408, 5078, 5080, 2714, 4863, 2031, 1959, 2112, 1761, 2187, 2932, 4571, 4428, 6453, 2004, 4671, 4528, 3228, 5988, 6311, 3265, 4832, 4068, 3832, 2347, 306, 29, 245, 4700, 2197, 3159, 1944, 1947, 4218, 4137]"

# 색상 코드
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "API Health Check Start: $(date)"
echo "Target: $API_URL"
echo "========================================"

# 오류 카운트
FAIL_COUNT=0

# ==========================================
# 테스트 함수 정의
# ==========================================
run_test() {
    local TEST_NAME="$1"
    local SPEAKER_JSON="$2"
    local OUTPUT_FILE="$3"

    # 전체 JSON 페이로드 조합
    local PAYLOAD="{\"speaker\": $SPEAKER_JSON, \"units\": $UNITS}"

    # CURL 실행
    # -s: 진행바 숨김
    # -w: HTTP 상태 코드만 별도 추출
    # -o: 결과 파일 저장
    HTTP_CODE=$(curl -s -o "$OUTPUT_FILE" -w "%{http_code}" -X POST "$API_URL" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD")

    # 검증 로직: HTTP 200 OK + 파일 존재 + 파일 크기 > 0
    if [ "$HTTP_CODE" -eq 200 ] && [ -s "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo -e "[ ${GREEN}PASS${NC} ] $TEST_NAME"
        echo "       L Status: $HTTP_CODE, Output: $OUTPUT_FILE ($FILE_SIZE)"
    else
        echo -e "[ ${RED}FAIL${NC} ] $TEST_NAME"
        echo "       L Status: $HTTP_CODE, Output: $OUTPUT_FILE"
        # 에러 발생 시 응답 내용 출력 (디버깅용)
        if [ -f "$OUTPUT_FILE" ]; then
            echo "       L Response Body: $(cat $OUTPUT_FILE)"
        fi
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
    echo "----------------------------------------"
}

# ==========================================
# 테스트 케이스 실행
# ==========================================

# 1. 기본 Speaker ID (fkms)
run_test "Case 1: Speaker ID (fkms)" \
    '{"id": "fkms"}' \
    "test_output.wav"

# 2. 성별 지정 (Male)
run_test "Case 2: Gender (Male)" \
    '{"gender": "m"}' \
    "test_male_output.wav"

# 3. 나이 및 성별 지정 (Age 10, Female)
run_test "Case 3: Age & Gender (10, Female)" \
    '{"age": "10", "gender": "f"}' \
    "test_10_f_output.wav"

# 4. Zero-shot (Reference Audio URL)
run_test "Case 4: Zero-shot (Ref Audio)" \
    '{"ref_audio_url": "https://kr.object.ncloudstorage.com/voice-change/4_standard_100/elevenlabs_c001_standard_female_jessica.mp3"}' \
    "test_zeroshot_output.wav"

# ==========================================
# 최종 리포트
# ==========================================
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}$FAIL_COUNT tests failed.${NC}"
    exit 1
fi
