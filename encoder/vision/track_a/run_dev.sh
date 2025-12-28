#!/usr/bin/zsh
# 개발 모드로 서버 실행 (자동 리로드 활성화)

cd "$(dirname "$0")"

echo "Starting server in development mode with auto-reload..."
echo "Code changes will be automatically detected and reloaded."
echo "Press Ctrl+C to stop the server."
echo ""

uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir app \
    --reload-include "*.py" \
    --log-level info

