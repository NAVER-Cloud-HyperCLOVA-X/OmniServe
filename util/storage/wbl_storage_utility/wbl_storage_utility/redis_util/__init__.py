# SPDX-License-Identifier: Apache-2.0

"""
Redis Storage Utilities

Redis를 파일 스토리지로 사용하기 위한 유틸리티를 제공합니다.
"""

from .redis_connection import (
    RedisConnection,
    get_global_redis_connection,
)

__all__ = [
    'RedisConnection',
    'get_global_redis_connection',
]

