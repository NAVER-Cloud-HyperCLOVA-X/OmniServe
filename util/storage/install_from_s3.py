#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
S3에서 wbl_storage_utility 패키지 설치 스크립트

사용법:
    python install_from_s3.py [버전]
    
예시:
    python install_from_s3.py 0.1.0
    python install_from_s3.py  # 최신 버전 자동 설치
"""

import sys
import subprocess
import re
import argparse
from pathlib import Path

try:
    from wbl_storage_utility.s3_util import S3Connection, S3ConnectionError
except ImportError:
    print("❌ wbl_storage_utility가 설치되어 있지 않습니다.")
    print("먼저 다음 명령으로 설치하세요:")
    print("  pip install boto3 botocore requests redis")
    print("  # 또는 소스에서 직접 설치")
    sys.exit(1)


def find_package_version(s3_conn, bucket_name, version=None):
    """S3에서 패키지 버전 찾기"""
    objects = s3_conn.list_objects(
        storage_name=bucket_name,
        prefix="wbl_storage_utility/",
        max_keys=100
    )
    
    # wheel 파일만 필터링
    wheel_files = [
        obj for obj in objects 
        if obj['Key'].endswith('.whl')
    ]
    
    if not wheel_files:
        return None
    
    if version:
        # 특정 버전 찾기
        for obj in wheel_files:
            if version in obj['Key']:
                return obj['Key']
        return None
    else:
        # 최신 버전 찾기
        def extract_version(key):
            match = re.search(r'(\d+\.\d+\.\d+)', key)
            return tuple(map(int, match.group(1).split('.'))) if match else (0, 0, 0)
        
        latest = max(wheel_files, key=lambda x: extract_version(x['Key']))
        return latest['Key']


def install_from_s3(version=None, upgrade=False):
    """S3에서 패키지 설치"""
    try:
        s3_conn = S3Connection()
        bucket_name = s3_conn.bucket_name
        
        print(f"📦 S3 버킷: {bucket_name}")
        print(f"🔍 패키지 검색 중...")
        
        package_key = find_package_version(s3_conn, bucket_name, version)
        
        if not package_key:
            if version:
                print(f"❌ 버전 {version}을 찾을 수 없습니다.")
            else:
                print("❌ S3에서 패키지를 찾을 수 없습니다.")
            return False
        
        print(f"✓ 패키지 발견: {package_key}")
        
        # Presigned URL 생성
        print("🔗 Presigned URL 생성 중...")
        presigned_url = s3_conn.create_presigned_get(
            bucket_name=bucket_name,
            object_name=package_key,
            expiration=3600 * 24 * 3650  # 10년
        )
        
        if not presigned_url:
            print("❌ Presigned URL 생성 실패")
            return False
        
        # pip install 실행
        print("⬇️  패키지 설치 중...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(presigned_url)
        
        subprocess.check_call(cmd)
        print("✓ 패키지 설치 완료!")
        return True
        
    except S3ConnectionError as e:
        print(f"❌ S3 연결 오류: {e}")
        print("\n환경 변수를 확인하세요:")
        print("  - NCP_S3_ACCESS_KEY")
        print("  - NCP_S3_SECRET_KEY")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ 설치 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def list_available_versions():
    """S3에 있는 모든 버전 목록 출력"""
    try:
        s3_conn = S3Connection()
        bucket_name = s3_conn.bucket_name
        
        objects = s3_conn.list_objects(
            storage_name=bucket_name,
            prefix="wbl_storage_utility/",
            max_keys=100
        )
        
        wheel_files = [
            obj for obj in objects 
            if obj['Key'].endswith('.whl')
        ]
        
        if not wheel_files:
            print("S3에 패키지가 없습니다.")
            return
        
        print(f"S3://{bucket_name}/wbl_storage_utility/ 에 있는 패키지:")
        print()
        
        # 버전별로 정렬
        def extract_version(key):
            match = re.search(r'(\d+\.\d+\.\d+)', key)
            return tuple(map(int, match.group(1).split('.'))) if match else (0, 0, 0)
        
        wheel_files.sort(key=lambda x: extract_version(x['Key']), reverse=True)
        
        for obj in wheel_files:
            version_match = re.search(r'(\d+\.\d+\.\d+)', obj['Key'])
            version = version_match.group(1) if version_match else "unknown"
            size_mb = obj['Size'] / (1024 * 1024)
            print(f"  {version:10s} - {obj['Key']:60s} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ 오류: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="S3에서 wbl_storage_utility 패키지 설치",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s                    # 최신 버전 설치
  %(prog)s 0.1.0              # 특정 버전 설치
  %(prog)s --upgrade          # 최신 버전으로 업그레이드
  %(prog)s --list              # 사용 가능한 버전 목록
        """
    )
    parser.add_argument(
        'version',
        nargs='?',
        help='설치할 패키지 버전 (예: 0.1.0). 생략 시 최신 버전 설치'
    )
    parser.add_argument(
        '--upgrade', '-U',
        action='store_true',
        help='기존 패키지를 업그레이드'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='사용 가능한 버전 목록 출력'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_versions()
        return
    
    success = install_from_s3(version=args.version, upgrade=args.upgrade)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

