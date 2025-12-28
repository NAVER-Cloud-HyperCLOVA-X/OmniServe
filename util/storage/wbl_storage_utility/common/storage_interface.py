# SPDX-License-Identifier: Apache-2.0

"""
Storage Interface Module

스토리지 시스템을 위한 공통 인터페이스를 정의합니다.
S3, Redis 등 다양한 스토리지 백엔드가 이 인터페이스를 구현합니다.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any


class StorageConnectionError(Exception):
    """스토리지 연결 관련 예외"""
    pass


class StorageUploadError(Exception):
    """스토리지 업로드 관련 예외"""
    pass


class StorageDownloadError(Exception):
    """스토리지 다운로드 관련 예외"""
    pass


class BaseStorageConnection(ABC):
    """
    스토리지 연결을 위한 추상 베이스 클래스
    
    모든 스토리지 구현체는 이 인터페이스를 따라야 합니다.
    """

    @abstractmethod
    def upload_wbl_asset(
            self,
            file_path: Path,
            key: str,
            prefix: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        파일을 업로드하고 접근 URL/키를 반환하는 통합 메서드

        Args:
            file_path: 업로드할 로컬 파일 경로
            key: 스토리지 객체 키
            prefix: 키 접두사 (선택사항)
            **kwargs: 추가 옵션 (ttl, metadata 등)

        Returns:
            접근 URL 또는 키 문자열

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            StorageUploadError: 업로드 실패 시
        """
        pass

    @abstractmethod
    def upload_file(self, storage_name: str, local_path: str, storage_path: str, **kwargs) -> bool:
        """
        로컬 파일을 스토리지에 업로드

        Args:
            storage_name: 스토리지 이름 (버킷명, DB명 등)
            local_path: 로컬 파일 경로
            storage_path: 스토리지 경로/키
            **kwargs: 추가 옵션

        Returns:
            업로드 성공 여부
        """
        pass

    @abstractmethod
    def download_file(self, storage_name: str, local_path: str, storage_path: str, **kwargs) -> bool:
        """
        스토리지에서 파일을 다운로드

        Args:
            storage_name: 스토리지 이름
            local_path: 저장할 로컬 파일 경로
            storage_path: 스토리지 경로/키
            **kwargs: 추가 옵션

        Returns:
            다운로드 성공 여부
        """
        pass

    @abstractmethod
    def delete_object(self, storage_name: str, object_key: str) -> bool:
        """
        객체 삭제

        Args:
            storage_name: 스토리지 이름
            object_key: 삭제할 객체 키

        Returns:
            삭제 성공 여부
        """
        pass

    @abstractmethod
    def object_exists(self, storage_name: str, object_key: str) -> bool:
        """
        객체 존재 여부 확인

        Args:
            storage_name: 스토리지 이름
            object_key: 확인할 객체 키

        Returns:
            객체 존재 여부
        """
        pass

    @abstractmethod
    def list_objects(
            self,
            storage_name: str,
            prefix: str = "",
            max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        객체 목록 조회

        Args:
            storage_name: 스토리지 이름
            prefix: 객체 키 접두사
            max_keys: 최대 반환 객체 수

        Returns:
            객체 정보 목록
        """
        pass

    @abstractmethod
    def get_metadata(self, storage_name: str, object_key: str) -> Optional[Dict[str, Any]]:
        """
        객체 메타데이터 조회

        Args:
            storage_name: 스토리지 이름
            object_key: 객체 키

        Returns:
            메타데이터 딕셔너리, 실패 시 None
        """
        pass

