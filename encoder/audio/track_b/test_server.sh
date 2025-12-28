#!/bin/bash
# 서버 테스트 스크립트

echo "=== Audio Encoder API 서버 테스트 ==="
echo ""

# 서버가 실행 중인지 확인
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "서버가 실행 중이 아닙니다. 서버를 먼저 시작해주세요."
    echo "실행 방법: python -m uvicorn app.server:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "1. Health Check"
echo "---------------"
curl -X GET http://localhost:8000/health
echo ""
echo ""

echo "2. Audio URL 테스트"
echo "-------------------"
curl -X POST http://localhost:8000/process_audio \
  -H "Content-Type: application/json" \
  -d '{"media_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"}' \
  -w "\nHTTP Status: %{http_code}\n"
echo ""
echo ""

echo "3. Video URL 테스트"
echo "-------------------"
curl -X POST http://localhost:8000/process_audio \
  -H "Content-Type: application/json" \
  -d '{"media_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"}' \
  -w "\nHTTP Status: %{http_code}\n"
echo ""
echo ""

echo "테스트 완료!"

