# SPDX-License-Identifier: Apache-2.0

"""
Storage Utilities 사용 예시

S3Connection과 RedisConnection이 동일한 인터페이스를 구현하여
서로 호환되는 방식으로 사용할 수 있음을 보여줍니다.
"""

import logging
from pathlib import Path
from typing import Union

from wbl_storage_utility.common import BaseStorageConnection
from wbl_storage_utility.s3_util import S3Connection
from wbl_storage_utility.redis_util import RedisConnection


def demonstrate_storage_compatibility(storage: BaseStorageConnection, storage_name: str):
    """
    스토리지 호환성 데모
    
    Args:
        storage: S3Connection 또는 RedisConnection 인스턴스
        storage_name: 스토리지 이름 (버킷명 또는 네임스페이스)
    """
    print(f"\n{'='*60}")
    print(f"Testing {storage.__class__.__name__}")
    print(f"{'='*60}\n")

    # 테스트 파일 생성
    test_file = Path("test_file.txt")
    test_file.write_text("Hello, World! This is a test file.")

    try:
        # 1. 파일 업로드
        print("1. Uploading file...")
        success = storage.upload_file(
            storage_name=storage_name,
            local_path=str(test_file),
            storage_path="test/sample.txt"
        )
        print(f"   Upload success: {success}")

        # 2. 객체 존재 확인
        print("\n2. Checking if object exists...")
        exists = storage.object_exists(
            storage_name=storage_name,
            object_key="test/sample.txt"
        )
        print(f"   Object exists: {exists}")

        # 3. 메타데이터 조회
        print("\n3. Getting metadata...")
        metadata = storage.get_metadata(
            storage_name=storage_name,
            object_key="test/sample.txt"
        )
        if metadata:
            print(f"   Metadata: {metadata}")

        # 4. 객체 목록 조회
        print("\n4. Listing objects...")
        objects = storage.list_objects(
            storage_name=storage_name,
            prefix="test/",
            max_keys=10
        )
        print(f"   Found {len(objects)} object(s)")
        for obj in objects:
            print(f"   - {obj.get('Key', 'N/A')}")

        # 5. 파일 다운로드
        print("\n5. Downloading file...")
        download_path = Path("downloaded_file.txt")
        success = storage.download_file(
            storage_name=storage_name,
            local_path=str(download_path),
            storage_path="test/sample.txt"
        )
        print(f"   Download success: {success}")
        
        if download_path.exists():
            content = download_path.read_text()
            print(f"   Downloaded content: {content}...")
            download_path.unlink()  # 다운로드 파일 삭제

        # 6. upload_wbl_asset 사용 예시
        print("\n6. Using upload_wbl_asset...")
        result = storage.upload_wbl_asset(
            file_path=test_file,
            key="demo.txt",
            prefix="assets"
        )
        print(f"   Result: {result}...")

        # 7. 객체 삭제
        print("\n7. Deleting objects...")
        for key in ["test/sample.txt", "assets/demo.txt"]:
            success = storage.delete_object(
                storage_name=storage_name,
                object_key=key
            )
            print(f"   Deleted {key}: {success}")

    finally:
        # 테스트 파일 정리
        if test_file.exists():
            test_file.unlink()

    print(f"\n{'='*60}\n")


def main():
    """메인 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Storage Interface Compatibility Demo")
    print("=" * 60)
    print("\n이 예시는 S3Connection과 RedisConnection이")
    print("동일한 BaseStorageConnection 인터페이스를 구현하여")
    print("호환 가능한 방식으로 사용될 수 있음을 보여줍니다.\n")

    # S3 스토리지 테스트
    try:
        print("\n[1/2] Testing S3 Storage...")
        s3_storage = S3Connection()
        demonstrate_storage_compatibility(s3_storage, "your-bucket")
    except Exception as e:
        print(f"S3 테스트 실패: {e}")
        print("S3 환경 변수를 설정해주세요:")
        print("  - NCP_S3_ACCESS_KEY")
        print("  - NCP_S3_SECRET_KEY")
        print("  - WBL_S3_BUCKET_NAME (선택)")

    # Redis 스토리지 테스트
    try:
        print("\n[2/2] Testing Redis Storage...")
        redis_storage = RedisConnection()
        demonstrate_storage_compatibility(redis_storage, "your-bucket")
    except Exception as e:
        print(f"Redis 테스트 실패: {e}")
        print("Redis 환경 변수를 설정하거나 Redis 서버를 시작해주세요:")
        print("  - REDIS_HOST (기본값: localhost)")
        print("  - REDIS_PORT (기본값: 6379)")
        print("  - REDIS_DB (기본값: 0)")

    print("\n" + "=" * 60)
    print("Demo 완료!")
    print("=" * 60)


def show_polymorphic_usage():
    """
    다형성 사용 예시
    
    동일한 코드로 S3와 Redis를 전환할 수 있습니다.
    """
    def process_file_with_storage(
            storage: BaseStorageConnection,
            storage_name: str,
            file_path: Path,
            key: str
    ):
        """스토리지 백엔드에 관계없이 파일을 처리하는 함수"""
        # 업로드
        storage.upload_file(storage_name, str(file_path), key)
        
        # 존재 확인
        if storage.object_exists(storage_name, key):
            print(f"File {key} uploaded successfully")
        
        # 메타데이터 조회
        metadata = storage.get_metadata(storage_name, key)
        print(f"Metadata: {metadata}")
        
        # 삭제
        storage.delete_object(storage_name, key)

    # 같은 함수를 S3와 Redis 모두에서 사용 가능
    # s3_conn = S3Connection()
    # process_file_with_storage(s3_conn, "my-bucket", Path("file.txt"), "path/to/file.txt")
    
    # redis_conn = RedisConnection()
    # process_file_with_storage(redis_conn, "my-namespace", Path("file.txt"), "path/to/file.txt")


if __name__ == "__main__":
    main()

