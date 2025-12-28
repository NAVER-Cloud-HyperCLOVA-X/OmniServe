# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import asyncio
import functools
import logging
import os
import random
import tempfile
from typing import Dict, Literal, Mapping, NamedTuple, Optional, Sequence, Union

import aiofiles
import aiohttp
import boto3
import botocore.config
import pydantic

from . import configs, exceptions, server_types


def _to_str_list(item_or_items: Union[str, Sequence[str]]) -> list[str]:
    return [item_or_items] if isinstance(item_or_items, str) else list(item_or_items)


class ReferenceConfig(pydantic.BaseModel):
    # Map (age, gender) to either audio file path, URL or speaker name
    audio_paths_or_urls_or_ids: Mapping[
        Literal[server_types.Age, "default"],
        Mapping[server_types.Gender, Union[str, Sequence[str]]],
    ]


_CACHE: Dict[str, bytes] = {}


async def _get_file_from_url(url: str) -> bytes:
    if url in _CACHE:
        return _CACHE[url]
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            audio = await response.read()
            _CACHE[url] = audio
            return audio


@functools.cache
def _get_s3_client(endpoint: str, region: str, access_key: str, secret_key: str):
    config = botocore.config.Config(
        s3={"use_accelerate_endpoint": False},
        retries={"max_attempts": 3},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=config,
    )


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    bucket, path = s3_path.split("/", 1)
    return bucket, path


def _get_file_from_s3(settings: configs.Settings, s3_path: str) -> bytes:
    if not settings.s3_access_key or not settings.s3_secret_key:
        logging.error(
            "NCP_S3_ACCESS_KEY 및 NCP_S3_SECRET_KEY 환경 변수가 설정되지 않아 S3 자원을 내려받지 못했습니다."
        )
        raise exceptions.ServerError("Internal Server Error while handling S3 URL")
    client = _get_s3_client(
        settings.s3_endpoint,
        settings.s3_region,
        settings.s3_access_key,
        settings.s3_secret_key,
    )

    bucket, path = _parse_s3_path(s3_path)
    with tempfile.NamedTemporaryFile(delete_on_close=False) as file:
        file.close()
        client.download_file(bucket, path, file.name)

        with open(file.name, "rb") as downloaded:
            return downloaded.read()


async def get_file_from_s3(settings: configs.Settings, s3_path: str) -> bytes:
    return await asyncio.to_thread(_get_file_from_s3, settings, s3_path)


async def _get_file_from_filesystem(path: str) -> bytes:
    if path in _CACHE:
        return _CACHE[path]
    async with aiofiles.open(path, mode="rb") as f:
        audio = await f.read()
        _CACHE[path] = audio
        return audio


async def load_file_from_url(settings: configs.Settings, url: str):
    if url.startswith("http://") or url.startswith("https://"):
        return await _get_file_from_url(url)

    if url.startswith("s3://"):
        return await get_file_from_s3(settings, url)

    if url.startswith("file://"):
        return await _get_file_from_filesystem(url[7:])

    return None


class ZeroshotReference(NamedTuple):
    raw_audio: bytes


class FinetunedReference(NamedTuple):
    speaker_id: str


Reference = Union[ZeroshotReference, FinetunedReference]


async def _load(
    settings: configs.Settings, path_or_url_or_id: str, rel_dir: str
) -> Reference:
    data = await load_file_from_url(settings, path_or_url_or_id)
    if data is not None:
        return ZeroshotReference(data)

    maybe_path = (
        path_or_url_or_id
        if os.path.isabs(path_or_url_or_id)
        else os.path.join(rel_dir, path_or_url_or_id)
    )
    if os.path.exists(maybe_path):
        return ZeroshotReference(await _get_file_from_filesystem(maybe_path))

    return FinetunedReference(path_or_url_or_id)  # Assume it's speaker ID


class References:
    def __init__(self, settings: configs.Settings) -> None:
        self._default_speaker = settings.default_speaker

        assert settings.speaker_config_path
        self._global_settings = settings

        try:
            with open(
                settings.speaker_config_path, "r", encoding="utf-8"
            ) as config_file:
                config = config_file.read()
        except Exception as e:
            raise ValueError("Cannot read speaker configuration file.") from e

        try:
            self._config = ReferenceConfig.model_validate_json(config)
        except Exception as e:
            raise ValueError("Speaker configuration is invalid.") from e

        if set(server_types.AGES) - set(self._config.audio_paths_or_urls_or_ids.keys()):
            raise ValueError(
                f"Reference audio configuration must specify for all ages: {server_types.AGES}"
            )
        if "default" not in self._config.audio_paths_or_urls_or_ids:
            raise ValueError(
                'Reference audio configuration must also specify for "default" age.'
            )

        for age, audios in self._config.audio_paths_or_urls_or_ids.items():
            if set(audios.keys()) != set(server_types.GENDERS):
                raise ValueError(
                    f"Reference audio configuration did not specify for all genders in age {age}."
                )
        self._config_dir = os.path.dirname(settings.speaker_config_path)

    async def get(
        self, age: Optional[server_types.Age], gender: Optional[server_types.Gender]
    ) -> Optional[Reference]:
        if not self._config.audio_paths_or_urls_or_ids:
            logging.warning("No audio configured. Using default speaker...")
            return None

        if not age and not gender:
            audio = self._default_speaker
            return await _load(self._global_settings, audio, rel_dir=self._config_dir)

        _age = age or "default"
        audio_map = self._config.audio_paths_or_urls_or_ids[_age]
        if gender:
            audio_list = _to_str_list(audio_map[gender])
        else:
            audio_list = [
                audio for audios in audio_map.values() for audio in _to_str_list(audios)
            ]
        audio = random.choice(audio_list)
        return await _load(self._global_settings, audio, rel_dir=self._config_dir)
