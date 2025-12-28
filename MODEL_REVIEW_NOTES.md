# Model Files Review Notes

오픈소스 공개 전 모델 파일 검토 결과입니다. 아래 항목들은 공개 전 수정이 필요합니다.

---

## HyperCLOVAX-SEED-Think-32B

### 1. config.json

| 라인 | 필드 | 현재 값 | 권장 조치 |
|------|------|---------|----------|
| 33 | `_name_or_path` | `/mnt/data/vuvlm/model_weights/nell-data251022-64node-lr-3e-5-resume/checkpoint-15064` | `null` 또는 `"./"` 로 변경 |
| 128 | `text_model_name_or_path` | `/mnt/data/vuvlm/model_weights/nell-data251022-64node-lr-3e-5-resume/checkpoint-15064` | `null` 또는 `"./"` 로 변경 |
| 141 | `vision_model_name_or_path` | `/mnt/ddn/vuvlm/model_weights/qwen2_5_vl_32b_visual_fa2/` | `null` 또는 `"./"` 로 변경 |

### 2. modeling_vlm.py

- 내부 링크 포함: `oss.navercorp.com`
- 해당 라인 검색 후 제거 또는 공개 URL로 대체 필요

---

## HyperCLOVAX-SEED-Omni-8B

### 1. config.json

| 라인 | 필드 | 현재 값 | 권장 조치 |
|------|------|---------|----------|
| 7 | `_name_or_path` | `/mnt/cmlssd004/HyperCLOVA-VLM/qwen2-audio-encoder-from-qwen2-audio-7b-instruct` | `null` 또는 `"./"` 로 변경 |
| 96 | `model_name_or_path` | `/mnt/cmlssd004/HyperCLOVA-VLM/cosyvoice2/tokenizer.pt` | `null` 또는 `"./"` 로 변경 |
| 108 | `model_name_or_path` | `/mnt/cmlssd004/HyperCLOVA-VLM/ta-tok/ta_tok.pth` | `null` 또는 `"./"` 로 변경 |
| 135 | `_name_or_path` | `/mnt/clovanap/checkpoints/hf/hcx-omni-8b-offee-cpt-stage2-251007-170000` | `null` 또는 `"./"` 로 변경 |
| 234 | `_name_or_path` | `/mnt/ddn/vuvlm/model_weights/qwen2_5_vl_32b_visual_fa2/` | `null` 또는 `"./"` 로 변경 |

### 2. configuration_vlm.py

| 라인 | 문제 | 권장 조치 |
|------|------|----------|
| 26 | 주석에 내부 경로 포함: `/mnt/ddn/vuvlm/outputs/251001_spd_hcx4b_stage2_c1_withmmpt_exp3/checkpoint-1500` | 해당 주석 제거 |

### 3. cosyvoice.py

| 라인 | 문제 | 권장 조치 |
|------|------|----------|
| 511 | `from_pretrained` 기본값: `/mnt/fr20tb/audiollm/models/encoder/cosyvoice2/tokenizer.pt` | 기본값을 `None`으로 변경하고 필수 파라미터로 설정 |

### 4. mambamia_videoaudio_compressor.py

| 라인 | 문제 | 권장 조치 |
|------|------|----------|
| 30 | 복사 지시사항에 내부 경로: `/mnt/cmlssd004/public/gwkim/dev/MLLM/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so` | 해당 주석 제거 또는 공개 설치 방법으로 대체 |

### 5. patch_vuvlm.py

| 라인 | 문제 | 권장 조치 |
|------|------|----------|
| 256-258 | docstring에 예시 경로: `/mnt/ocr-nfsx1/HyperCLOVA-VLM/...` | 예시 경로를 일반적인 형태로 변경 (예: `/path/to/model/`) |

### 6. modeling_vlm.py

- 내부 링크 포함: `oss.navercorp.com`
- 해당 라인 검색 후 제거 또는 공개 URL로 대체 필요

### 7. preprocessor.py

- 내부 링크 포함: `oss.navercorp.com`
- 해당 라인 검색 후 제거 또는 공개 URL로 대체 필요

---

## 발견된 내부 스토리지 경로 패턴

다음 경로 패턴들이 내부 인프라 정보를 노출합니다:

- `/mnt/data/` - 데이터 스토리지
- `/mnt/ddn/` - 고성능 스토리지
- `/mnt/cmlssd004/` - 로컬 SSD 스토리지
- `/mnt/clovanap/` - 체크포인트 저장소
- `/mnt/fr20tb/` - 오디오 모델 저장소
- `/mnt/ocr-nfsx1/` - NFS 공유 스토리지

---

## 검색 명령어

문제가 되는 부분을 찾기 위한 명령어:

```bash
# 내부 경로 검색
grep -rn "/mnt/" --include="*.py" --include="*.json" .

# 내부 링크 검색
grep -rn "oss.navercorp.com\|gist.oss.navercorp" --include="*.py" .
```
