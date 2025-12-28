#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""Test server by starting it and making requests"""
import subprocess
import time
import requests
import json
import sys
import signal
import os


def test_server():
    # Start server
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(__file__)),  # Use script directory
    )

    try:
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(20)

        # Check if server is running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print(f"Server failed to start!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return 1

        # Test health endpoint
        print("\n=== Testing /health ===")
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return 1

        # Test image processing
        print("\n=== Testing /process_image_url ===")
        try:
            response = requests.post(
                "http://localhost:8000/process_image_url",
                json={"image_url": "https://picsum.photos/400/300"},
                timeout=60,
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Vision query length: {result.get('vision_query_length', 'N/A')}")
                print(f"Continuous features shape: {result.get('continuous_features_shape', 'N/A')}")
            else:
                print(f"Error: {response.text}")
                return 1
        except Exception as e:
            print(f"Image processing failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

        print("\n=== All tests passed! ===")
        return 0

    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()


if __name__ == "__main__":
    sys.exit(test_server())
