import json
import base64

# 제공해주신 JSON 응답 데이터
response_data = {
  "id": "chatcmpl-d62c494013ab438cb48e8125500737da",
  "object": "chat.completion",
  "created": 1766591800,
  "model": "vllm-omni",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<user_transcript>안녕하세요 테스트 중 입니다</user_transcript>\n<assistant_transcript>안녕하심니꺼</assistant_transcript>\n<audio_decoder_call>\n<speaker_info>{\"is_ref_audio\": true}</speaker_info>\n<discrete_audio>\n<|discrete_audio_start|>",
        "audio": {
          "id": "aHR0cHM6Ly9rci5vYmplY3QubmNsb3Vkc3RvcmFnZS5jb20vd2JsL3NvdXJjZS9kZXJpdmVkL2F1ZGlvL2F1ZGlvLWRlY29kZXIvMjhkOTM4YmQtZGQyZS00YzVmLWFlNGQtYWM2Mjg1MTg2YjJjLndhdj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPW5jcF9pYW1fQlBBU0tSamh2bXg5MWhISFU0dVElMkYyMDI1MTIyNCUyRmtyLXN0YW5kYXJkJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIyNFQxNTU2NDBaJlgtQW16LUV4cGlyZXM9MzYwMCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmWC1BbXotU2lnbmF0dXJlPTJlZjQ3Y2VlNTFkMjU5OTkyODI0NjQzODgwNThjYTljODMxZTRjYmNjNTNlNDAzNTkyNTA4NmFiZjY0ZmIyNTc=",
          "data": "aHR0cHM6Ly9rci5vYmplY3QubmNsb3Vkc3RvcmFnZS5jb20vd2JsL3NvdXJjZS9kZXJpdmVkL2F1ZGlvL2F1ZGlvLWRlY29kZXIvMjhkOTM4YmQtZGQyZS00YzVmLWFlNGQtYWM2Mjg1MTg2YjJjLndhdj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPW5jcF9pYW1fQlBBU0tSamh2bXg5MWhISFU0dVElMkYyMDI1MTIyNCUyRmtyLXN0YW5kYXJkJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIyNFQxNTU2NDBaJlgtQW16LUV4cGlyZXM9MzYwMCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmWC1BbXotU2lnbmF0dXJlPTJlZjQ3Y2VlNTFkMjU5OTkyODI0NjQzODgwNThjYTljODMxZTRjYmNjNTNlNDAzNTkyNTA4NmFiZjY0ZmIyNTc=",
          "expires_at": 1766595400,
          "transcript": "..."
        }
      },
      "finish_reason": "stop"
    }
  ]
}

def extract_audio_url(data):
    try:
        # 1. JSON 구조에서 encoded 데이터 추출
        encoded_str = data['choices'][0]['message']['audio']['data']
        
        # 2. Base64 디코딩
        decoded_bytes = base64.b64decode(encoded_str)
        decoded_url = decoded_bytes.decode('utf-8')
        
        return decoded_url
    except Exception as e:
        return f"Error: {e}"

# 실행
url = extract_audio_url(response_data)
print(f"변환된 오디오 URL:\n{url}")

# (선택사항) 파일 다운로드 기능이 필요하면 아래 주석을 해제하세요.
# import urllib.request
# urllib.request.urlretrieve(url, "output_audio.wav")
# print("파일이 output_audio.wav로 저장되었습니다.")
