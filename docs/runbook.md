# Runbook

## 1. Pre-flight

```bash
git status --short --branch
python -m pip install -e .
python -m pip check
pytest
```

Xác nhận:

- Đúng branch/run config.
- `ALQAC_TEAM_TOKEN` tồn tại trong secret storage, không nằm trong source.
- Public/law files đúng đường dẫn config.
- GPU T4 và Internet đã bật nếu chạy Kaggle.

## 2. CPU smoke test

```bash
python scripts/run_public.py \
  --mock \
  --limit 2 \
  --config configs/baseline.yaml \
  --resume-run outputs/smoke
```

Kiểm tra `outputs/smoke/validation.json` có `status: PASS`. Mock score không phải kết quả model.

## 3. Kaggle two-case validation

Notebook mặc định `RUN_MODE='smoke'`, tương ứng `limit=2`. Chạy từ kernel sạch và kiểm tra:

- Model revisions tải thành công.
- Không CUDA OOM.
- Case API trả kết quả và cache có record.
- Mỗi prediction parse JSON thành công.
- Validator PASS.

Sau khi thành công, đổi `RUN_MODE='full'`. Notebook tự dùng run directory khác; runner cũng từ chối resume nếu mock/limit/input/config thay đổi.

## 4. Full public run

```bash
python scripts/build_law_index.py --config configs/candidate.yaml
python scripts/run_public.py \
  --config configs/candidate.yaml \
  --resume-run outputs/public_candidate_full
```

Không thay config trong khi resume. Lưu `metrics.json`, `manifest.json` và leaderboard score bên ngoài Git.

So sánh với baseline bằng `scripts/compare_runs.py`. Chỉ sao chép/promote `candidate.yaml` thành `final.yaml` khi clean full public artifact cho thấy candidate tốt hơn theo thứ tự outcome accuracy, official score, law F1, format stability và runtime.

## 5. Private run

```bash
python scripts/run_private.py \
  --config configs/candidate.yaml \
  --input data/private/private_test.json \
  --resume-run submissions/private_candidate_full
```

Nếu session dừng, chạy lại đúng lệnh. Sau khi hoàn tất:

```bash
python scripts/validate_submission.py \
  --input submissions/private_candidate_full/submission.json \
  --test-data data/private/private_test.json
```

Chỉ upload khi validator PASS, số cases đúng private input và submission owner xác nhận lượt nộp.

## 6. Source bundle

```bash
pytest
git diff --check
python scripts/package_source.py
```

Mở zip kiểm tra có README, config, source, scripts, notebooks, tests và docs; không có `.env`, data, cache, weights, logs, outputs hoặc submissions.

## Failure handling

| Failure | Action |
|---|---|
| `403` API | Kiểm tra Kaggle/Colab secret và token name; không retry hàng loạt |
| `422` API | Kiểm tra case ID/query schema |
| `429/5xx` API | Client tự backoff; resume cùng run/cache |
| CUDA OOM khi retrieval | Giảm reranker batch size, build contexts trước rồi release models |
| CUDA OOM khi Qwen | Xác nhận NF4 4-bit, context 7.000 tokens và không còn retrieval models trên GPU |
| Model output invalid | Pipeline repair một lần; kiểm tra failed record, không tự sửa label |
| Config mismatch khi resume | Dùng run directory mới hoặc khôi phục đúng config cũ |
| Validator FAIL | Không upload; sửa đúng lỗi được report và chạy validator lại |
