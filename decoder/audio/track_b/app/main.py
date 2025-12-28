# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import Dict, Optional

import fastapi

# 추가 : S3 업로드와 임시 파일 작성을 위한 추가 import
import tempfile  # CHANGED
from pathlib import Path  # CHANGED
import uuid  # CHANGED
import logging  # CHANGED
from wbl_storage_utility.s3_util import S3Connection  # CHANGED

from . import configs, decoders, exceptions, references, server_types

# 추가 : 로거 설정 (필수는 아니지만 디버깅에 편함)
logger = logging.getLogger(__name__)  # CHANGED

# Load from environment variables
settings = configs.Settings()
app = fastapi.FastAPI()

# 추가 : S3Connection 인스턴스를 전역으로 하나만 생성해서 재사용
#   - AudioEncoder 쪽 코드와 동일하게 기본 생성자를 사용
#   - 환경변수(NCP_S3_*, WBL_S3_BUCKET_NAME 등)는 S3Connection 내부에서 읽어감
s3_conn = S3Connection()  # CHANGED

speakers = None
if settings.speaker_config_path:
    speakers = references.References(settings)


FORMAT_MIME_MAP: Dict[server_types.Format, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}


async def _load_speaker(
    speaker: Optional[server_types.Speaker],
) -> references.Reference:
    default_speaker = references.FinetunedReference(settings.default_speaker)
    if not speaker:
        return default_speaker

    if speaker.id:
        return references.FinetunedReference(speaker.id)

    if speaker.ref_audio_base64:
        return references.ZeroshotReference(speaker.ref_audio_base64)

    if speaker.ref_audio_url:
        data = await references.load_file_from_url(settings, speaker.ref_audio_url)
        if data is None:
            raise exceptions.BadRequestException(
                "ref_audio_url must start with http(s):// or s3://"
            )
        return references.ZeroshotReference(data)

    # age and gender mapping
    if not speakers:
        return default_speaker
    _speaker = await speakers.get(speaker.age, speaker.gender)
    if not _speaker:
        return default_speaker
    return _speaker


@app.post("/predictions")
async def completions(request: server_types.Request):
    # Input request logging (curl payload 형태)
    print(f"[Request] {request.model_dump_json()}")
    
    speaker = await _load_speaker(request.speaker)

    is_zeroshot = isinstance(speaker, references.ZeroshotReference)
    model_name = settings.zeroshot_model if is_zeroshot else settings.finetuned_model

    wav_bytes = await decoders.request(
        settings.endpoint, model_name, request.units, speaker
    )

    audio_bytes = decoders.convert_audio(wav_bytes, request.format)
    
    # 요청 포맷 기반으로 확장자/키 생성
    # request.format 은 "mp3", "wav" 같은 문자열(서버 타입 alias)이라고 가정
    file_ext = request.format  # e.g. "mp3"
    s3_key = f"source/derived/audio/audio-decoder/{uuid.uuid4()}.{file_ext}"  # CHANGED
    # s3_key = f"audio-decoder/output/{uuid.uuid4()}.{file_ext}"  # CHANGED

    # audio_bytes를 임시 파일로 저장 후 업로드
    tmp_path = None  # CHANGED
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:  # CHANGED
            tmp_file.write(audio_bytes)  # CHANGED
            tmp_path = Path(tmp_file.name)  # CHANGED

        print(f"Uploading audio to S3: key={s3_key}, size={len(audio_bytes)} bytes")  # CHANGED

        # CHANGED: AudioEncoder 예제에서 사용하던 방식과 동일하게 upload_wbl_asset 사용
        #   - 내부에서 WBL_S3_BUCKET_NAME, NCP_S3_* 환경변수 사용
        uploaded_key = s3_conn.upload_wbl_asset(file_path=tmp_path, key=s3_key)  # CHANGED

        print(f"S3 upload done: {uploaded_key}")  # CHANGED

    finally:
        # 임시 파일 정리
        if tmp_path and tmp_path.exists():  # CHANGED
            try:
                tmp_path.unlink()  # CHANGED
            except Exception as e:  # CHANGED
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")  # CHANGED

    # 테스트 편의상, 업로드된 S3 키와 메타 정보들을 JSON으로 리턴
    return {  # CHANGED
        # "s3_key": uploaded_key,  # 실제 S3 object key (upload_wbl_asset 반환값)
        "s3_key": s3_key, 
        "requested_format": request.format,
        "content_type": FORMAT_MIME_MAP[request.format],
        "size_bytes": len(audio_bytes),
    }

    # 기존 코드
    # return fastapi.Response(
    #     content=audio_bytes, media_type=FORMAT_MIME_MAP[request.format]
    # )
