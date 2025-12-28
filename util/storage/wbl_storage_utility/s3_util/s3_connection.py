# SPDX-License-Identifier: Apache-2.0

"""
NCP S3 Connection Module

네이버 클라우드 플랫폼 Object Storage와 연동하는 클래스를 제공합니다.
파일 업로드, 다운로드, 목록 조회 및 presigned URL 생성을 지원합니다.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import boto3
from botocore.config import Config
import requests
from botocore.exceptions import ClientError

from ..common.storage_interface import (
    BaseStorageConnection,
    StorageConnectionError,
    StorageUploadError
)


# 하위 호환성을 위한 별칭
class S3ConnectionError(StorageConnectionError):
    """S3 연결 관련 예외 (StorageConnectionError의 별칭)"""
    pass


class S3UploadError(StorageUploadError):
    """S3 업로드 관련 예외 (StorageUploadError의 별칭)"""
    pass


class S3Connection(BaseStorageConnection):
    """
    NCP Object Storage S3 연결 클래스

    환경변수를 통해 설정을 읽어와 S3 클라이언트를 초기화하고,
    파일 업로드/다운로드/관리 기능을 제공합니다.
    """

    def __init__(self, reuse_client: bool = True) -> None:
        """
        S3Connection 초기화

        Args:
            reuse_client: 클라이언트 재사용 여부 (기본값: True)
        """
        self._sync_client = None
        self.reuse_client = reuse_client

        # 환경 변수에서 설정 읽기
        self.service_name = os.getenv('NCP_S3_SERVICE')
        self.endpoint_url = os.getenv('NCP_S3_ENDPOINT')
        self.region_name = os.getenv('NCP_S3_REGION')
        self.access_key = os.getenv('NCP_S3_ACCESS_KEY')
        self.secret_key = os.getenv('NCP_S3_SECRET_KEY')
        self.bucket_name = os.getenv('WBL_S3_BUCKET_NAME')

        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """필수 환경 변수 검증"""
        if not self.access_key or not self.secret_key:
            raise S3ConnectionError(
                "NCP_S3_ACCESS_KEY 및 NCP_S3_SECRET_KEY 환경 변수가 설정되어야 합니다.\n"
                "설정 예시:\n"
                "export NCP_S3_ACCESS_KEY='your_access_key'\n"
                "export NCP_S3_SECRET_KEY='your_secret_key'"
            )

    def get_sync_client(self) -> boto3.client:
        """
        S3 클라이언트 인스턴스 반환

        Returns:
            boto3 S3 클라이언트
        """
        if self._sync_client is None or not self.reuse_client:
            from botocore.config import Config
            config = Config(
    s3={'use_accelerate_endpoint': False},
    retries={'max_attempts': 3},
    request_checksum_calculation='when_required',
    response_checksum_validation='when_required'
           )
            self._sync_client = boto3.client(
                self.service_name,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                config = config
            )
        return self._sync_client

    def upload_wbl_asset(
            self,
            file_path: Path,
            key: str,
            prefix: Optional[str] = None,
            s3_key: Optional[str] = None,  # 하위 호환성
            s3_prefix: Optional[str] = None,  # 하위 호환성
            **kwargs
    ) -> str:
        """
        파일을 업로드하고 presigned URL을 반환하는 통합 메서드

        Args:
            file_path: 업로드할 로컬 파일 경로
            key: S3 객체 키
            prefix: S3 키 접두사 (선택사항)
            s3_key: (deprecated) key의 별칭
            s3_prefix: (deprecated) prefix의 별칭
            **kwargs: 추가 옵션

        Returns:
            presigned URL 문자열

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            S3UploadError: 업로드 실패 시
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 하위 호환성: 기존 파라미터명 지원
        actual_key = s3_key if s3_key is not None else key
        actual_prefix = s3_prefix if s3_prefix is not None else prefix

        # S3 키 구성
        full_s3_key = f"{actual_prefix}/{actual_key}" if actual_prefix else actual_key

        # 파일 업로드
        success = self.upload_file(
            storage_name=self.bucket_name,
            local_path=str(file_path),
            storage_path=full_s3_key
        )

        if not success:
            raise S3UploadError(f"Failed to upload {file_path} to S3")

        # presigned URL 생성
        presigned_url = self.create_presigned_get(
            bucket_name=self.bucket_name,
            object_name=full_s3_key
        )

        if presigned_url is None:
            raise S3UploadError(f"Failed to generate presigned URL for {full_s3_key}")

        return presigned_url

    def download_file(
            self,
            storage_name: str,
            local_path: str,
            storage_path: str,
            bucket_name: Optional[str] = None,  # 하위 호환성
            s3_path: Optional[str] = None,  # 하위 호환성
            **kwargs
    ) -> bool:
        """
        S3에서 파일을 다운로드

        Args:
            storage_name: S3 버킷 이름
            local_path: 저장할 로컬 파일 경로
            storage_path: S3 객체 경로
            bucket_name: (deprecated) storage_name의 별칭
            s3_path: (deprecated) storage_path의 별칭
            **kwargs: 추가 옵션

        Returns:
            다운로드 성공 여부
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name
        actual_path = s3_path if s3_path is not None else storage_path

        client = self.get_sync_client()
        try:
            client.download_file(actual_bucket, actual_path, local_path)
            logging.info(f"Successfully downloaded {actual_bucket}/{actual_path} to {local_path}")
            return True
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return False

    def upload_file(
            self,
            storage_name: str,
            local_path: str,
            storage_path: str,
            bucket_name: Optional[str] = None,  # 하위 호환성
            s3_path: Optional[str] = None,  # 하위 호환성
            **kwargs
    ) -> bool:
        """
        로컬 파일을 S3에 업로드

        Args:
            storage_name: S3 버킷 이름
            local_path: 로컬 파일 경로
            storage_path: S3 객체 경로
            bucket_name: (deprecated) storage_name의 별칭
            s3_path: (deprecated) storage_path의 별칭
            **kwargs: 추가 옵션

        Returns:
            업로드 성공 여부
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name
        actual_path = s3_path if s3_path is not None else storage_path

        client = self.get_sync_client()

        try:
            client.upload_file(local_path, actual_bucket, actual_path)
            logging.info(f"Successfully uploaded {local_path} to {actual_bucket}/{actual_path}")
            return True
        except Exception as e:
            logging.error(f"Upload failed: {e}")
            return False

    def upload_via_presigned_post(
            self,
            url: str,
            fields: Dict[str, Any],
            local_path: str,
            object_key: str
    ) -> None:
        """
        Presigned POST URL을 사용하여 파일 업로드

        Args:
            url: Presigned POST URL
            fields: 폼 필드
            local_path: 로컬 파일 경로
            object_key: S3 객체 키
        """
        file_path = Path(local_path)

        logging.info(f"업로드할 파일: {file_path}")
        logging.info(f"S3 키: {object_key}")

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (object_key, f)}
                http_response = requests.post(url, data=fields, files=files)

            logging.info(f'File upload HTTP status code: {http_response.status_code}')

            if http_response.status_code != 204:
                logging.error(f"Upload failed with status: {http_response.status_code}")
                logging.error(f"Response: {http_response.text}")

        except Exception as e:
            logging.error(f"Upload via presigned POST failed: {e}")
            raise S3UploadError(f"Failed to upload via presigned POST: {e}")

    def create_presigned_post(
            self,
            bucket_name: str,
            object_name: str,
            fields: Optional[Dict[str, Any]] = None,
            conditions: Optional[List[Any]] = None,
            expiration: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        파일 업로드용 Presigned POST URL 생성

        Args:
            bucket_name: S3 버킷 이름
            object_name: S3 객체 이름
            fields: 사전 정의된 폼 필드
            conditions: 정책 조건 목록
            expiration: 만료 시간 (초, 기본값: 3600)

        Returns:
            URL과 필드가 포함된 딕셔너리, 실패 시 None
        """
        client = self.get_sync_client()

        try:
            response = client.generate_presigned_post(
                bucket_name,
                object_name,
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=expiration,
            )
            return response
        except ClientError as e:
            logging.error(f"Failed to generate presigned POST URL: {e}")
            return None

    def create_presigned_get(
            self,
            bucket_name: str,
            object_name: str,
            expiration: int = 3600 * 24
    ) -> Optional[str]:
        """
        파일 다운로드용 Presigned GET URL 생성

        Args:
            bucket_name: S3 버킷 이름
            object_name: S3 객체 이름
            expiration: 만료 시간 (초, 기본값: 24시간)

        Returns:
            Presigned URL, 실패 시 None
        """
        client = self.get_sync_client()

        try:
            response = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            logging.error(f"Failed to generate presigned URL: {e}")
            return None

    def list_objects(
            self,
            storage_name: str,
            prefix: str = "",
            max_keys: int = 1000,
            bucket_name: Optional[str] = None  # 하위 호환성
    ) -> List[Dict[str, Any]]:
        """
        버킷의 객체 목록 조회

        Args:
            storage_name: S3 버킷 이름
            prefix: 객체 키 접두사
            max_keys: 최대 반환 객체 수
            bucket_name: (deprecated) storage_name의 별칭

        Returns:
            객체 정보 목록
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name

        client = self.get_sync_client()

        try:
            response = client.list_objects_v2(
                Bucket=actual_bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return response.get('Contents', [])
        except ClientError as e:
            logging.error(f"Failed to list objects: {e}")
            return []

    def delete_object(
            self,
            storage_name: str,
            object_key: str,
            bucket_name: Optional[str] = None  # 하위 호환성
    ) -> bool:
        """
        객체 삭제

        Args:
            storage_name: S3 버킷 이름
            object_key: 삭제할 객체 키
            bucket_name: (deprecated) storage_name의 별칭

        Returns:
            삭제 성공 여부
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name

        client = self.get_sync_client()

        try:
            client.delete_object(Bucket=actual_bucket, Key=object_key)
            logging.info(f"Successfully deleted {actual_bucket}/{object_key}")
            return True
        except ClientError as e:
            logging.error(f"Failed to delete object: {e}")
            return False

    def object_exists(
            self,
            storage_name: str,
            object_key: str,
            bucket_name: Optional[str] = None  # 하위 호환성
    ) -> bool:
        """
        객체 존재 여부 확인

        Args:
            storage_name: S3 버킷 이름
            object_key: 확인할 객체 키
            bucket_name: (deprecated) storage_name의 별칭

        Returns:
            객체 존재 여부
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name

        client = self.get_sync_client()

        try:
            client.head_object(Bucket=actual_bucket, Key=object_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logging.error(f"Failed to check object existence: {e}")
            return False

    def get_metadata(
            self,
            storage_name: str,
            object_key: str,
            bucket_name: Optional[str] = None  # 하위 호환성
    ) -> Optional[Dict[str, Any]]:
        """
        객체 메타데이터 조회

        Args:
            storage_name: S3 버킷 이름
            object_key: 객체 키
            bucket_name: (deprecated) storage_name의 별칭

        Returns:
            메타데이터 딕셔너리, 실패 시 None
        """
        # 하위 호환성
        actual_bucket = bucket_name if bucket_name is not None else storage_name

        client = self.get_sync_client()

        try:
            response = client.head_object(Bucket=actual_bucket, Key=object_key)
            
            metadata = {
                'ContentLength': response.get('ContentLength'),
                'ContentType': response.get('ContentType'),
                'ETag': response.get('ETag'),
                'LastModified': response.get('LastModified').isoformat() if response.get('LastModified') else None,
                'Metadata': response.get('Metadata', {}),
                'StorageClass': response.get('StorageClass'),
            }
            
            return metadata

        except ClientError as e:
            logging.error(f"Failed to get metadata: {e}")
            return None


def setup_environment_example() -> None:
    """환경 변수 설정 예시를 출력"""
    example_vars = [
        "export NCP_S3_ACCESS_KEY='your_access_key'",
        "export NCP_S3_SECRET_KEY='your_secret_key'",
        "export NCP_S3_ENDPOINT='https://your-s3-endpoint.com'",
        "export NCP_S3_REGION='your-region'",
        "export NCP_S3_SERVICE='s3'",
        "export WBL_S3_BUCKET_NAME='your-bucket'"
    ]

    print("=== 환경 변수 설정 예시 ===")
    for var in example_vars:
        print(var)

    print("\nPython에서 설정:")
    print("import os")
    for var in example_vars:
        key_value = var.split('=')
        key = key_value[0].split()[1]
        value = key_value[1]
        print(f"os.environ['{key}'] = {value}")


# 전역 인스턴스 (필요한 경우에만 사용)
def get_global_s3_connection() -> S3Connection:
    """전역 S3 연결 인스턴스를 반환"""
    if not hasattr(get_global_s3_connection, '_instance'):
        get_global_s3_connection._instance = S3Connection()
    return get_global_s3_connection._instance


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    try:
        # S3 연결 초기화
        s3_conn = S3Connection()

        # 파일 업로드 예시
        # file_path = Path("example.txt")
        # presigned_url = s3_conn.upload_wbl_asset(
        #     file_path=file_path,
        #     s3_key="test/example.txt"
        # )
        # print(f"Presigned URL: {presigned_url}")

    except S3ConnectionError as e:
        print(f"S3 연결 오류: {e}")
        setup_environment_example()
