#!/bin/bash
# 서버 실행 및 테스트 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 기존 서버 종료
pkill -f "uvicorn app.server:app" || true
sleep 2

# 서버 시작
echo "Starting server..."
nohup python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 > /tmp/audio_encoder_server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."
sleep 60

# Health check
echo ""
echo "=== Testing /health ==="
curl -X GET http://localhost:8000/health
echo ""

# Audio URL test
echo ""
echo "=== Testing Audio URL ==="
curl -X POST http://localhost:8000/process_audio \
  -H "Content-Type: application/json" \
  -d '{"media_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"}' \
  -w "\nHTTP Status: %{http_code}\n"
echo ""

# Video URL test
echo ""
echo "=== Testing Video URL ==="
curl -X POST http://localhost:8000/process_audio \
  -H "Content-Type: application/json" \
  -d '{"media_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"}' \
  -w "\nHTTP Status: %{http_code}\n"
echo ""

# Server logs
echo ""
echo "=== Server logs (last 50 lines) ==="
tail -50 /tmp/audio_encoder_server.log

# 서버 종료
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true

