# WBL Storage Utility

WBL Object Storage(S3) 및 Redis를 쉽게 사용할 수 있도록 만든 Python 유틸리티 모듈입니다.

이 리포지토리는 다음을 지원합니다.
- 멀티모달 데이터 저장 규칙


## 주요 특징

* **공통 인터페이스 (BaseStorageConnection)**
  - S3와 Redis가 동일한 인터페이스를 구현하여 **호환 가능**
  - 스토리지 백엔드를 쉽게 전환 가능
  - 다형성을 활용한 유연한 코드 작성

* **S3 기능**
  - 환경 변수 기반 **S3 클라이언트 초기화**
  - **파일 업로드/다운로드** (일반 업로드)
  - **Presigned GET URL** 생성 (다운로드용)
  - **Presigned POST URL** 생성 (업로드용)
  - **wbl asset 업로드 워크플로우** (`upload_wbl_asset`)
  - 객체 목록 조회 / 삭제 / 존재 여부 확인 / 메타데이터 조회

* **Redis 기능**
  - 환경 변수 기반 **Redis 클라이언트 초기화**
  - **파일 업로드/다운로드** (바이너리 저장)
  - **TTL(Time To Live)** 설정 및 조회
  - **메타데이터 관리** (JSON 형식)
  - 객체 목록 조회 / 삭제 / 존재 여부 확인
  - 인덱싱을 통한 효율적인 객체 관리


## 1. 폴더 구성

```text
.
├── README.md
├── pyproject.toml            # 프로젝트 설정 및 의존성
├── example_usage.py          # S3/Redis 호환성 데모
└── wbl_storage_utility/
    ├── common/
    │   ├── __init__.py
    │   └── storage_interface.py  # BaseStorageConnection 공통 인터페이스
    ├── s3_util/
    │   ├── __init__.py
    │   ├── s3_connection.py      # S3Connection 클래스
    │   └── s3_connection_test.py  # S3 종합 테스트
    └── redis_util/
        ├── __init__.py
        └── redis_connection.py   # RedisConnection 클래스
```

## 2. 선행 준비

### 2.1. Python 버전

- Python 3.8 이상 권장

### 2.2. 필수 패키지 설치

```bash
pip install .
# or
uv pip install .
```



