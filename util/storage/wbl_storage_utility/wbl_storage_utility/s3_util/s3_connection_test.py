#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S3Connection 클래스 테스트 스크립트

이 스크립트는 다음 기능들을 테스트합니다:
1. S3Connection 클래스 초기화
2. 파일 업로드/다운로드 기능
3. Presigned URL 생성 기능 (GET/POST)
4. vLLM asset 업로드 워크플로우
5. 객체 존재 여부 확인
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time
import requests

from wbl_storage_utility.s3_util import S3Connection, S3ConnectionError, S3UploadError


class TestConfig:
    """테스트 설정 클래스"""

    def __init__(self):
        self.bucket_name = os.getenv('WBL_S3_BUCKET_NAME', "your-bucket")
        self.test_file_content = "This is a test file for S3 upload testing."
        self.test_file_name = "test_file.txt"
        self.test_file_download_name = "test_file_download.txt"

        # 테스트 경로 설정
        self.upload_paths = {
            'simple': 'simple-uploads/example_file.txt',
            'presigned_post': 'presigned-post-url-gen/post_test_basic.txt',
            'workflow': 'workflow_test.txt',
            'workflow_prefix': 'presigned-url-get'
        }


class S3TestRunner:
    """S3 테스트 실행 클래스"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.s3_conn = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _create_test_file(self) -> Path:
        """테스트용 파일 생성"""
        test_file = Path(self.config.test_file_name)
        test_file.write_text(self.config.test_file_content, encoding='utf-8')
        self.logger.info(f"테스트 파일 생성: {test_file}")
        return test_file

    def _cleanup_test_file(self) -> None:
        """테스트 파일 정리"""
        test_file = Path(self.config.test_file_name)
        test_file_download_file = Path(self.config.test_file_download_name)
        if test_file.exists():
            test_file.unlink()
            self.logger.info(f"테스트 파일 삭제: {test_file}")
        if test_file_download_file.exists():
            test_file_download_file.unlink()
            self.logger.info(f"테스트 파일 삭제: {test_file_download_file}")

    def _print_section_header(self, title: str) -> None:
        """섹션 헤더 출력"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def _print_test_result(self, test_name: str, success: bool, details: str = "") -> None:
        """테스트 결과 출력"""
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
        if details:
            print(f"  {details}")

    def initialize_s3_connection(self) -> bool:
        """S3 연결 초기화 테스트"""
        self._print_section_header("S3 Connection 초기화 테스트")

        try:
            self.s3_conn = S3Connection()
            self._print_test_result("S3 연결 초기화", True, f"버킷: {self.config.bucket_name}")
            return True

        except S3ConnectionError as e:
            self._print_test_result("S3 연결 초기화", False, str(e))
            return False
        except Exception as e:
            self._print_test_result("S3 연결 초기화", False, f"예상치 못한 오류: {e}")
            return False

    def test_file_upload(self) -> Optional[str]:
        """기본 파일 업로드 테스트"""
        self._print_section_header("기본 파일 업로드 테스트")

        try:
            test_file = self._create_test_file()
            object_name = self.config.upload_paths['simple']

            print(f"업로드할 파일: {test_file}")
            print(f"S3 키: {object_name}")
            print(f"버킷: {self.config.bucket_name}")

            success = self.s3_conn.upload_file(
                bucket_name=self.config.bucket_name,
                local_path=str(test_file),
                s3_path=object_name
            )

            if success:
                # 업로드된 객체 존재 확인
                exists = self.s3_conn.object_exists(
                    bucket_name=self.config.bucket_name,
                    object_key=object_name
                )

                if exists:
                    self._print_test_result("파일 업로드 및 존재 확인", True)
                    return object_name
                else:
                    self._print_test_result("파일 업로드", False, "업로드 후 객체를 찾을 수 없음")
                    return None
            else:
                self._print_test_result("파일 업로드", False)
                return None

        except Exception as e:
            self._print_test_result("파일 업로드", False, str(e))
            return None
    def test_file_download(self) -> Optional[str]:
        """기본 파일 다운로드 테스트"""
        self._print_section_header("기본 파일 다운로드 테스트")
        test_file = "test_file_download.txt"

        try:
            object_name = self.config.upload_paths['simple']

            print(f"S3 키: {object_name}")
            print(f"버킷: {self.config.bucket_name}")
            """
               def download_file(self, bucket_name: str, local_path: str, s3_path: str) -> bool:
        client = self.get_sync_client()
        try:
            client.download_file( bucket_name=bucket_name, local_path=local_path, s3_path=s3_path)
            logging.info(f"Successfully downloaded {local_path} to {bucket_name}/{s3_path}")
            return True
        except Exception as e:
            logging.error(f"Downloaded failed: {e}")
            return False
            """

            success = self.s3_conn.download_file(
                bucket_name=self.config.bucket_name,
                local_path=str(test_file),
                s3_path=object_name
            )

            if success:
                # 업로드된 객체 존재 확인
                exists = self.s3_conn.object_exists(
                    bucket_name=self.config.bucket_name,
                    object_key=object_name
                )

                if exists:
                    self._print_test_result("파일 업로드 및 존재 확인", True)
                    return object_name
                else:
                    self._print_test_result("파일 업로드", False, "업로드 후 객체를 찾을 수 없음")
                    return None
            else:
                self._print_test_result("파일 업로드", False)
                return None

        except Exception as e:
            self._print_test_result("파일 업로드", False, str(e))
            return None

    def test_presigned_get_url(self, object_name: str) -> Optional[str]:
        """Presigned GET URL 생성 테스트"""
        self._print_section_header("Presigned GET URL 테스트")

        try:
            presigned_url = self.s3_conn.create_presigned_get(
                bucket_name=self.config.bucket_name,
                object_name=object_name,
                expiration=3600  # 1시간
            )

            if presigned_url:
                self._print_test_result("Presigned GET URL 생성", True)
                print(f"  URL 길이: {len(presigned_url)} 문자")
                print(f"  URL 미리보기: {presigned_url}")

                # URL 유효성 간단 테스트
                try:
                    response = requests.head(presigned_url, timeout=10)
                    self._print_test_result("URL 접근성 확인", True)

                except requests.RequestException as e:
                    self._print_test_result("URL 접근성 확인", False, f"네트워크 오류: {e}")

                return presigned_url
            else:
                self._print_test_result("Presigned GET URL 생성", False)
                return None

        except Exception as e:
            self._print_test_result("Presigned GET URL 생성", False, str(e))
            return None

    def test_presigned_post_url(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Presigned POST URL 생성 테스트"""
        self._print_section_header("Presigned POST URL 테스트")

        try:
            object_name = self.config.upload_paths['presigned_post']

            print(f"객체 이름: {object_name}")
            print(f"버킷: {self.config.bucket_name}")

            # 기본 POST URL 생성
            basic_response = self.s3_conn.create_presigned_post(
                bucket_name=self.config.bucket_name,
                object_name=object_name,
                expiration=3600  # 1시간
            )

            if basic_response:
                self._print_test_result("기본 Presigned POST URL 생성", True)
                print(f"  POST URL: {basic_response['url']}")
                print(f"  Form Fields 수: {len(basic_response['fields'])}")

                # 조건부 POST URL 생성
                print("\n--- 조건부 POST URL 생성 ---")

                conditions = [
                    {"bucket": self.config.bucket_name},
                    {"Content-Type": "text/plain"}
                ]

                fields = {
                    "Content-Type": "text/plain",
                }

                conditional_response = self.s3_conn.create_presigned_post(
                    bucket_name=self.config.bucket_name,
                    object_name=object_name,
                    fields=fields,
                    conditions=conditions,
                    expiration=1800  # 30분
                )

                if conditional_response:
                    self._print_test_result("조건부 Presigned POST URL 생성", True)
                    return conditional_response['url'], conditional_response['fields']
                else:
                    self._print_test_result("조건부 Presigned POST URL 생성", False)
                    return basic_response['url'], basic_response['fields']
            else:
                self._print_test_result("기본 Presigned POST URL 생성", False)
                return None, None

        except Exception as e:
            self._print_test_result("Presigned POST URL 생성", False, str(e))
            return None, None

    def test_upload_via_presigned_post(self, url: str, fields: Dict[str, Any]) -> bool:
        """Presigned POST를 통한 파일 업로드 테스트"""
        self._print_section_header("Presigned POST 업로드 테스트")

        try:
            test_file = Path(self.config.test_file_name)
            object_name = self.config.upload_paths['presigned_post']

            print(f"업로드할 파일: {test_file}")
            print(f"S3 키: {object_name}")

            # 불필요한 메타데이터 필드 제거
            clean_fields = {k: v for k, v in fields.items()
                            if not k.startswith('x-amz-meta-')}

            # 파일 업로드
            with open(test_file, 'rb') as f:
                files = {'file': (object_name, f)}
                response = requests.post(url, data=clean_fields, files=files, timeout=30)

            self.logger.info(f'HTTP 응답 상태: {response.status_code}')

            if response.status_code == 204:
                self._print_test_result("Presigned POST 업로드", True, "HTTP 204 응답")

                # 업로드 확인
                exists = self.s3_conn.object_exists(
                    bucket_name=self.config.bucket_name,
                    object_key=object_name
                )

                if exists:
                    self._print_test_result("업로드 파일 존재 확인", True)
                    return True
                else:
                    self._print_test_result("업로드 파일 존재 확인", False)
                    return False
            else:
                self._print_test_result("Presigned POST 업로드", False,
                                        f"HTTP {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self._print_test_result("Presigned POST 업로드", False, str(e))
            return False

    def test_wbl_upload_workflow(self) -> Optional[str]:
        """vLLM asset 업로드 전체 워크플로우 테스트"""
        self._print_section_header("vLLM Asset 업로드 워크플로우 테스트")

        try:
            test_file = Path(self.config.test_file_name)
            object_name = self.config.upload_paths['workflow']
            s3_prefix = self.config.upload_paths['workflow_prefix']

            print(f"업로드할 파일: {test_file}")
            print(f"S3 키: {object_name}")
            print(f"S3 프리픽스: {s3_prefix}")

            presigned_url = self.s3_conn.upload_wbl_asset(
                file_path=test_file,
                s3_key=object_name,
                s3_prefix=s3_prefix
            )

            if presigned_url:
                self._print_test_result("vLLM asset 업로드 및 URL 생성", True)
                print(f"  URL 길이: {len(presigned_url)} 문자")
                print(f"  URL 미리보기: {presigned_url}")

                # 최종 객체 키 생성
                final_key = f"{s3_prefix}/{object_name}"

                # 업로드된 파일 존재 확인
                exists = self.s3_conn.object_exists(
                    bucket_name=self.config.bucket_name,
                    object_key=final_key
                )

                if exists:
                    self._print_test_result("최종 객체 존재 확인", True, f"키: {final_key}")
                else:
                    self._print_test_result("최종 객체 존재 확인", False, f"키: {final_key}")

                return presigned_url
            else:
                self._print_test_result("vLLM asset 업로드", False)
                return None

        except (S3UploadError, FileNotFoundError) as e:
            self._print_test_result("vLLM asset 업로드", False, str(e))
            return None
        except Exception as e:
            self._print_test_result("vLLM asset 업로드", False, f"예상치 못한 오류: {e}")
            return None

    def test_object_listing(self) -> None:
        """객체 목록 조회 테스트"""
        self._print_section_header("객체 목록 조회 테스트")

        try:
            # 전체 객체 목록 조회
            all_objects = self.s3_conn.list_objects(
                bucket_name=self.config.bucket_name,
                max_keys=10
            )

            self._print_test_result("전체 객체 목록 조회", True, f"{len(all_objects)}개 객체 발견")

            if all_objects:
                print("  최근 객체들:")
                for i, obj in enumerate(all_objects[:5]):
                    print(f"    {i + 1}. {obj['Key']} ({obj['Size']} bytes)")

            # 특정 프리픽스로 객체 조회
            prefix_objects = self.s3_conn.list_objects(
                bucket_name=self.config.bucket_name,
                prefix="simple-uploads/",
                max_keys=5
            )

            self._print_test_result("프리픽스별 객체 조회", True,
                                    f"'simple-uploads/' 프리픽스: {len(prefix_objects)}개")

        except Exception as e:
            self._print_test_result("객체 목록 조회", False, str(e))

    def run_all_tests(self) -> None:
        """모든 테스트 실행"""
        print("S3Connection 클래스 종합 테스트 시작")
        print(f"테스트 버킷: {self.config.bucket_name}")

        try:
            # 1. S3 연결 초기화
            if not self.initialize_s3_connection():
                print("S3 연결 실패로 테스트를 종료합니다.")
                return

            # 테스트 파일 생성
            self._create_test_file()

            # 2. 기본 파일 업로드
            uploaded_key = self.test_file_upload()

            # 3. Presigned GET URL 테스트
            if uploaded_key:
                self.test_presigned_get_url(uploaded_key)

            # 4. 기본 파일 다운로드
            self.test_file_download()

            # 5. Presigned POST URL 테스트
            post_url, post_fields = self.test_presigned_post_url()

            # 6. Presigned POST 업로드 테스트
            if post_url and post_fields:
                self.test_upload_via_presigned_post(post_url, post_fields)

            # 7. vLLM 워크플로우 테스트
            self.test_wbl_upload_workflow()

            # 8. 객체 목록 조회 테스트
            self.test_object_listing()

            # 최종 결과
            self._print_section_header("테스트 완료")
            print("모든 테스트가 완료되었습니다.")
            print("위의 결과를 확인하여 각 기능의 동작 상태를 점검하세요.")

        finally:
            # 정리 작업
            self._cleanup_test_file()


def main():
    """메인 함수"""
    try:
        config = TestConfig()
        test_runner = S3TestRunner(config)
        test_runner.run_all_tests()

    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
