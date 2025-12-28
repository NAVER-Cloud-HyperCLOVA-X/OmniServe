# SPDX-License-Identifier: Apache-2.0

"""
Redis Storage Connection Module

Redis를 파일 스토리지로 사용하는 클래스를 제공합니다.
파일 업로드, 다운로드, 목록 조회 및 메타데이터 관리를 지원합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import redis
from redis.cluster import RedisCluster, ClusterNode
from redis.exceptions import RedisError

from ..common.storage_interface import (
    BaseStorageConnection,
    StorageConnectionError,
    StorageUploadError,
    StorageDownloadError
)


class RedisConnection(BaseStorageConnection):
    """
    Redis 스토리지 연결 클래스
    
    환경변수를 통해 설정을 읽어와 Redis 클라이언트를 초기화하고,
    파일 업로드/다운로드/관리 기능을 제공합니다.
    """

    # Redis 키 패턴
    DATA_KEY_PREFIX = "storage:data:"
    METADATA_KEY_PREFIX = "storage:meta:"
    INDEX_KEY = "storage:index"

    def __init__(self, reuse_client: bool = True, cluster_mode: bool = False) -> None:
        """
        RedisConnection 초기화

        Args:
            reuse_client: 클라이언트 재사용 여부 (기본값: True)
        """
        self._client = None
        self.reuse_client = reuse_client

        # 환경 변수에서 설정 읽기
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.db = int(os.getenv('REDIS_DB', '0'))
        self.password = os.getenv('REDIS_PASSWORD')
        self.default_ttl = int(os.getenv('REDIS_DEFAULT_TTL', str(3600 * 24 * 7)))  # 기본 7일
        self.default_storage_name = os.getenv('WBL_REDIS_STORAGE_NAME', 'your-storage')
        self.cluster_mode = cluster_mode
        
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Redis 연결 검증"""
        try:
            client = self.get_client()
            client.ping()
        except RedisError as e:
            raise StorageConnectionError(
                f"Redis 연결 실패: {e}\n"
                f"설정 확인:\n"
                f"REDIS_HOST={self.host}\n"
                f"REDIS_PORT={self.port}\n"
                f"REDIS_DB={self.db}"
            )

    def get_client(self) -> redis.Redis:
        """
        Redis 클라이언트 인스턴스 반환

        Returns:
            redis.Redis 클라이언트
        """
        if self._client is None or not self.reuse_client:
            if self.cluster_mode:
                self._client = RedisCluster(
                    startup_nodes=[
                        ClusterNode(
                            host=self.host,
                            port=self.port,
                        )
                    ],
                    password=self.password,
                    decode_responses=False  # 바이너리 데이터 처리를 위해 False
                )
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False  # 바이너리 데이터 처리를 위해 False
                )
        return self._client

    def _make_data_key(self, storage_name: str, object_key: str) -> str:
        """데이터 키 생성"""
        return f"{self.DATA_KEY_PREFIX}{storage_name}:{object_key}"

    def _make_metadata_key(self, storage_name: str, object_key: str) -> str:
        """메타데이터 키 생성"""
        return f"{self.METADATA_KEY_PREFIX}{storage_name}:{object_key}"

    def _make_index_key(self, storage_name: str) -> str:
        """인덱스 키 생성"""
        return f"{self.INDEX_KEY}:{storage_name}"

    def upload_wbl_asset(
            self,
            file_path: Path,
            key: str,
            prefix: Optional[str] = None,
            ttl: Optional[int] = None,
            **kwargs
    ) -> str:
        """
        파일을 업로드하고 접근 키를 반환하는 통합 메서드

        Args:
            file_path: 업로드할 로컬 파일 경로
            key: Redis 객체 키
            prefix: 키 접두사 (선택사항)
            ttl: Time To Live (초), None이면 기본값 사용
            **kwargs: 추가 메타데이터

        Returns:
            Redis 키 문자열

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            StorageUploadError: 업로드 실패 시
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 키 구성
        full_key = f"{prefix}/{key}" if prefix else key

        # 파일 업로드
        success = self.upload_file(
            storage_name=self.default_storage_name,
            local_path=str(file_path),
            storage_path=full_key,
            ttl=ttl,
            **kwargs
        )

        if not success:
            raise StorageUploadError(f"Failed to upload {file_path} to Redis")

        return full_key

    def upload_file(
            self,
            storage_name: str,
            local_path: str,
            storage_path: str,
            ttl: Optional[int] = None,
            **kwargs
    ) -> bool:
        """
        로컬 파일을 Redis에 업로드

        Args:
            storage_name: Redis 스토리지 네임스페이스
            local_path: 로컬 파일 경로
            storage_path: Redis 키 경로
            ttl: Time To Live (초)
            **kwargs: 추가 메타데이터

        Returns:
            업로드 성공 여부
        """
        client = self.get_client()
        file_path = Path(local_path)

        try:
            # 파일 읽기
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # 메타데이터 생성
            metadata = {
                'filename': file_path.name,
                'size': len(file_data),
                'uploaded_at': datetime.utcnow().isoformat(),
                'content_type': self._guess_content_type(file_path),
                **kwargs
            }

            # 키 생성
            data_key = self._make_data_key(storage_name, storage_path)
            meta_key = self._make_metadata_key(storage_name, storage_path)
            index_key = self._make_index_key(storage_name)

            # 파이프라인으로 원자적 업로드
            pipe = client.pipeline()
            
            # 데이터 저장
            pipe.set(data_key, file_data)
            if ttl or self.default_ttl:
                pipe.expire(data_key, ttl or self.default_ttl)

            # 메타데이터 저장
            pipe.set(meta_key, json.dumps(metadata))
            if ttl or self.default_ttl:
                pipe.expire(meta_key, ttl or self.default_ttl)

            # 인덱스에 추가
            pipe.sadd(index_key, storage_path)

            pipe.execute()

            logging.info(f"Successfully uploaded {local_path} to Redis:{storage_name}/{storage_path}")
            return True

        except (IOError, RedisError) as e:
            logging.error(f"Upload failed: {e}")
            return False

    def download_file(
            self,
            storage_name: str,
            local_path: str,
            storage_path: str,
            **kwargs
    ) -> bool:
        """
        Redis에서 파일을 다운로드

        Args:
            storage_name: Redis 스토리지 네임스페이스
            local_path: 저장할 로컬 파일 경로
            storage_path: Redis 키 경로
            **kwargs: 추가 옵션

        Returns:
            다운로드 성공 여부
        """
        client = self.get_client()

        try:
            data_key = self._make_data_key(storage_name, storage_path)
            
            # 데이터 가져오기
            file_data = client.get(data_key)
            
            if file_data is None:
                logging.error(f"Object not found: {storage_name}/{storage_path}")
                return False

            # 파일 쓰기
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_file, 'wb') as f:
                f.write(file_data)

            logging.info(f"Successfully downloaded Redis:{storage_name}/{storage_path} to {local_path}")
            return True

        except (IOError, RedisError) as e:
            logging.error(f"Download failed: {e}")
            return False

    def delete_object(self, storage_name: str, object_key: str) -> bool:
        """
        객체 삭제

        Args:
            storage_name: Redis 스토리지 네임스페이스
            object_key: 삭제할 객체 키

        Returns:
            삭제 성공 여부
        """
        client = self.get_client()

        try:
            data_key = self._make_data_key(storage_name, object_key)
            meta_key = self._make_metadata_key(storage_name, object_key)
            index_key = self._make_index_key(storage_name)

            # 파이프라인으로 원자적 삭제
            pipe = client.pipeline()
            pipe.delete(data_key)
            pipe.delete(meta_key)
            pipe.srem(index_key, object_key)
            results = pipe.execute()

            if results[0] > 0:  # 데이터가 삭제되었는지 확인
                logging.info(f"Successfully deleted Redis:{storage_name}/{object_key}")
                return True
            else:
                logging.warning(f"Object not found: {storage_name}/{object_key}")
                return False

        except RedisError as e:
            logging.error(f"Failed to delete object: {e}")
            return False

    def object_exists(self, storage_name: str, object_key: str) -> bool:
        """
        객체 존재 여부 확인

        Args:
            storage_name: Redis 스토리지 네임스페이스
            object_key: 확인할 객체 키

        Returns:
            객체 존재 여부
        """
        client = self.get_client()

        try:
            data_key = self._make_data_key(storage_name, object_key)
            return client.exists(data_key) > 0

        except RedisError as e:
            logging.error(f"Failed to check object existence: {e}")
            return False

    def list_objects(
            self,
            storage_name: str,
            prefix: str = "",
            max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        객체 목록 조회

        Args:
            storage_name: Redis 스토리지 네임스페이스
            prefix: 객체 키 접두사
            max_keys: 최대 반환 객체 수

        Returns:
            객체 정보 목록
        """
        client = self.get_client()

        try:
            index_key = self._make_index_key(storage_name)
            all_keys = client.smembers(index_key)

            # 바이트를 문자열로 변환하고 필터링
            all_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys]
            
            if prefix:
                filtered_keys = [k for k in all_keys if k.startswith(prefix)]
            else:
                filtered_keys = list(all_keys)

            # 최대 개수 제한
            filtered_keys = filtered_keys[:max_keys]

            # 메타데이터 수집
            objects = []
            for key in filtered_keys:
                metadata = self.get_metadata(storage_name, key)
                if metadata:
                    objects.append({
                        'Key': key,
                        'Size': metadata.get('size', 0),
                        'LastModified': metadata.get('uploaded_at', ''),
                        'Metadata': metadata
                    })

            return objects

        except RedisError as e:
            logging.error(f"Failed to list objects: {e}")
            return []

    def get_metadata(self, storage_name: str, object_key: str) -> Optional[Dict[str, Any]]:
        """
        객체 메타데이터 조회

        Args:
            storage_name: Redis 스토리지 네임스페이스
            object_key: 객체 키

        Returns:
            메타데이터 딕셔너리, 실패 시 None
        """
        client = self.get_client()

        try:
            meta_key = self._make_metadata_key(storage_name, object_key)
            metadata_json = client.get(meta_key)

            if metadata_json is None:
                return None

            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')

            return json.loads(metadata_json)

        except (RedisError, json.JSONDecodeError) as e:
            logging.error(f"Failed to get metadata: {e}")
            return None

    def set_ttl(self, storage_name: str, object_key: str, ttl: int) -> bool:
        """
        객체의 TTL 설정

        Args:
            storage_name: Redis 스토리지 네임스페이스
            object_key: 객체 키
            ttl: Time To Live (초)

        Returns:
            설정 성공 여부
        """
        client = self.get_client()

        try:
            data_key = self._make_data_key(storage_name, object_key)
            meta_key = self._make_metadata_key(storage_name, object_key)

            pipe = client.pipeline()
            pipe.expire(data_key, ttl)
            pipe.expire(meta_key, ttl)
            results = pipe.execute()

            return all(results)

        except RedisError as e:
            logging.error(f"Failed to set TTL: {e}")
            return False

    def get_ttl(self, storage_name: str, object_key: str) -> Optional[int]:
        """
        객체의 남은 TTL 조회

        Args:
            storage_name: Redis 스토리지 네임스페이스
            object_key: 객체 키

        Returns:
            남은 시간(초), 없으면 None
        """
        client = self.get_client()

        try:
            data_key = self._make_data_key(storage_name, object_key)
            ttl = client.ttl(data_key)
            
            if ttl < 0:  # -1: 만료 없음, -2: 키 없음
                return None
            
            return ttl

        except RedisError as e:
            logging.error(f"Failed to get TTL: {e}")
            return None

    @staticmethod
    def _guess_content_type(file_path: Path) -> str:
        """파일 확장자로 Content-Type 추정"""
        extension_map = {
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.pdf': 'application/pdf',
            '.zip': 'application/zip',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
        }
        return extension_map.get(file_path.suffix.lower(), 'application/octet-stream')


def setup_environment_example() -> None:
    """환경 변수 설정 예시를 출력"""
    example_vars = [
        "export REDIS_HOST='localhost'",
        "export REDIS_PORT='6379'",
        "export REDIS_DB='0'",
        "export REDIS_PASSWORD='your_password'  # 선택사항",
        "export REDIS_DEFAULT_TTL='604800'  # 7일 (초)",
        "export WBL_REDIS_STORAGE_NAME='your-storage'"
    ]

    print("=== 환경 변수 설정 예시 ===")
    for var in example_vars:
        print(var)

    print("\nPython에서 설정:")
    print("import os")
    for var in example_vars:
        if '#' in var:
            var = var.split('#')[0].strip()
        if '=' in var:
            key_value = var.split('=')
            key = key_value[0].split()[1]
            value = key_value[1]
            print(f"os.environ['{key}'] = {value}")


# 전역 인스턴스 (필요한 경우에만 사용)
def get_global_redis_connection() -> RedisConnection:
    """전역 Redis 연결 인스턴스를 반환"""
    if not hasattr(get_global_redis_connection, '_instance'):
        get_global_redis_connection._instance = RedisConnection()
    return get_global_redis_connection._instance


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    try:
        # Redis 연결 초기화
        redis_conn = RedisConnection()

        # 파일 업로드 예시
        # file_path = Path("example.txt")
        # redis_key = redis_conn.upload_wbl_asset(
        #     file_path=file_path,
        #     key="test/example.txt",
        #     ttl=3600
        # )
        # print(f"Redis Key: {redis_key}")

        # 객체 목록 조회
        # objects = redis_conn.list_objects(storage_name="your-storage", prefix="test/")
        # for obj in objects:
        #     print(f"Key: {obj['Key']}, Size: {obj['Size']}")

    except StorageConnectionError as e:
        print(f"Redis 연결 오류: {e}")
        setup_environment_example()

