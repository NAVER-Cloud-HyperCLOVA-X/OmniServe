#!/bin/bash

# process_image_or_video 엔드포인트 테스트 스크립트
# 서버 주소와 포트를 환경변수로 설정하거나 기본값 사용
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
ENDPOINT="${ENDPOINT:-/process_image_or_video}"

echo "=== process_image_or_video 엔드포인트 테스트 ==="
echo "서버 URL: ${SERVER_URL}${ENDPOINT}"
echo ""

# 1. 이미지 URL로 테스트
echo "1. 이미지 URL로 테스트"
curl -X POST "${SERVER_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/image.jpg",
    "anyres": false,
    "unpad": false,
    "num_queries_vis_abstractor": 0
  }' | jq .
echo -e "\n"

# 2. 비디오 URL로 테스트
echo "2. 비디오 URL로 테스트"
curl -X POST "${SERVER_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/video.mp4"
  }' | jq .
echo -e "\n"

# 3. Base64 이미지로 테스트 (data URL 형식)
echo "3. Base64 이미지로 테스트 (data URL 형식)"
# 작은 테스트 이미지를 Base64로 인코딩 (실제 사용 시에는 실제 이미지 파일을 인코딩)
# 예시: 1x1 픽셀 PNG 이미지
BASE64_IMAGE="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
curl -X POST "${SERVER_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d "{
    \"media_base64\": \"${BASE64_IMAGE}\",
    \"anyres\": true,
    \"unpad\": true,
    \"num_queries_vis_abstractor\": 10
  }" | jq .
echo -e "\n"

# 4. Base64 비디오로 테스트 (data URL 형식)
echo "4. Base64 비디오로 테스트 (data URL 형식)"
# 실제 비디오 파일을 Base64로 인코딩해야 함
# 예시 형식만 보여줌
curl -X POST "${SERVER_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{
    "media_base64": "data:video/mp4;base64,AAAAIGZ0eXBpc29t..."
  }' | jq .
echo -e "\n"

# 5. 최소 필수 파라미터만 사용 (이미지 URL)
echo "5. 최소 필수 파라미터만 사용 (이미지 URL)"
curl -X POST "${SERVER_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/image.png"
  }' | jq .
echo -e "\n"

# 6. 로컬 이미지 파일을 Base64로 변환하여 테스트
echo "6. 로컬 이미지 파일을 Base64로 변환하여 테스트"
if [ -f "test_image.jpg" ]; then
  BASE64_CONTENT=$(base64 -w 0 test_image.jpg)
  curl -X POST "${SERVER_URL}${ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "{
      \"media_base64\": \"data:image/jpeg;base64,${BASE64_CONTENT}\"
    }" | jq .
else
  echo "test_image.jpg 파일이 없습니다. 건너뜁니다."
fi
echo -e "\n"

echo "=== 테스트 완료 ==="

