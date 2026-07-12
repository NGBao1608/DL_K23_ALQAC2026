# ALQAC 2026 — Legal Case Outcome Prediction

Pipeline của team K23 cho ALQAC 2026. Hệ thống chỉ nhận `case_id` và `case_query`, truy hồi case evidence qua API chính thức, truy hồi luật trong corpus và dự đoán một trong bốn nhãn kết quả.

## Kiến trúc

```text
case_id + case_query
  ├─ deterministic queries → Case Content API → case evidence
  ├─ BM25 + Vietnamese_Embedding → RRF → Vietnamese_Reranker → law evidence
  └─ Qwen3-8B 4-bit → outcome prediction
       ↓
validated submission.json
```

Model chính:

- Outcome: `Qwen/Qwen3-8B`, NF4 4-bit, thinking disabled.
- Embedding: `AITeamVN/Vietnamese_Embedding`.
- Reranker: `AITeamVN/Vietnamese_Reranker`.
- Sparse retrieval: BM25.

Chi tiết module và data boundary nằm trong [docs/architecture.md](docs/architecture.md).

## Cấu trúc repo

```text
configs/            Cấu hình baseline và candidate
data/raw/           Public test và law corpus chính thức
data/private/       Private test, không commit
src/alqac2026/      Toàn bộ logic production
scripts/            CLI entrypoints
notebooks/          Notebook điều phối Kaggle/Colab
tests/              Unit và integration smoke tests
docs/               Kiến trúc, luật thi và runbook
cache/              API cache và law index, không commit
outputs/            Public experiment artifacts, không commit
submissions/        Private run artifacts, không commit
```

Hai cấu hình chính:

- `configs/baseline.yaml`: BM25-only law retrieval, vẫn dùng Case API và Qwen3.
- `configs/candidate.yaml`: BM25 + dense embedding + RRF + reranker + Qwen3. Chỉ promote thành `final.yaml` sau clean full public comparison.

## Cài đặt local

Yêu cầu Python 3.10+. Qwen3 NF4 4-bit cần Linux và GPU NVIDIA/CUDA; local macOS chỉ phù hợp với test, validator và BM25 smoke run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Điền `ALQAC_TEAM_TOKEN` vào `.env`. Không commit file này.

Kiểm tra môi trường:

```bash
python -m pip check
pytest
```

## Chạy smoke test không cần token/GPU

```bash
pytest
python scripts/run_public.py --mock --limit 2 --config configs/baseline.yaml
```

## Public development

Baseline BM25:

```bash
python scripts/run_public.py --config configs/baseline.yaml
```

Hybrid retrieval + reranking:

```bash
python scripts/build_law_index.py --config configs/candidate.yaml
python scripts/run_public.py --config configs/candidate.yaml
```

Đánh giá Recall@5/10 và so sánh runs:

```bash
python scripts/evaluate_retrieval.py \
  --config configs/baseline.yaml \
  --output outputs/retrieval_bm25.json

python scripts/compare_runs.py outputs/public_baseline outputs/public_candidate_full
```

Public runner tạo private-like input chỉ gồm `case_id` và `case_query`. Gold fields chỉ được evaluator đọc sau inference.

Public run thật sẽ gọi Case Content API tối đa tám query/case. Với 50 cases và rate limit 5 giây, riêng thời gian chờ API có thể vượt 30 phút; cache giúp các lần chạy sau không gọi lại query đã thành công.

Resume một run bị gián đoạn:

```bash
python scripts/run_public.py \
  --config configs/candidate.yaml \
  --resume-run outputs/public_candidate_full
```

`--resume-run` chỉ được dùng lại với đúng config ban đầu. Case API responses được cache trong SQLite; completed predictions được lấy từ checkpoint. Law context của case chưa hoàn tất có thể được tính lại nhưng không tiêu tốn lại API calls đã cache.

## Private inference

```bash
python scripts/run_private.py \
  --config configs/candidate.yaml \
  --input data/private/private_test.json \
  --resume-run submissions/private_candidate_full
```

Mỗi run tạo thư mục có:

- `submission.json` — file duy nhất dùng để upload.
- `validation.json` — phải có `status: PASS`.
- `predictions.checkpoint.json` — checkpoint nội bộ.
- `manifest.json` — config, model, Git revision, corpus hash và API stats.
- `config.resolved.json` — config chính xác của run.
- `environment.json` — Python, package versions, CUDA và GPU name.
- `predictions.json` — internal predictions theo đúng input order.
- `api_stats.json` — total/per-case API calls và cache stats của process.
- `metrics.json` — chỉ có trong public evaluation.
- `errors.json` — public cases dự đoán sai/failed để error analysis.
- `contexts.checkpoint.json` — prepared case/law contexts để resume không rerank lại.

Kiểm tra độc lập:

```bash
python scripts/validate_submission.py \
  --input submissions/<run>/submission.json \
  --test-data data/private/private_test.json
```

Leaderboard được upload thủ công bởi submission owner. Giới hạn chính thức là tối đa 3 submissions/ngày/team.

## Kaggle T4

1. Bật Accelerator → GPU T4 và Internet trong Notebook Settings.
2. Thêm secret `ALQAC_TEAM_TOKEN` trong Add-ons → Secrets.
3. Clone repo vào `/kaggle/working/DL_K23_ALQAC2026` hoặc upload source archive rồi giải nén tại đó.
4. Đặt notebook working directory tại repo root trước cell đầu tiên.
5. Với private phase, attach private JSON dưới `/kaggle/input/alqac2026-private/private_test.json` hoặc sửa biến `PRIVATE_INPUT` trong cell cấu hình.
6. Mở `notebooks/public_development.ipynb` hoặc `notebooks/private_inference.ipynb` và chạy Restart & Run All.

Notebook mặc định `RUN_MODE='smoke'` và chỉ chạy hai cases. Chỉ chuyển sang `RUN_MODE='full'` sau khi smoke run PASS; smoke và full dùng run directory riêng nên không thể resume lẫn nhau.

Pipeline chạy theo hai stage để phù hợp T4:

1. Build/query law index và rerank toàn bộ cases, sau đó release embedding/reranker khỏi GPU.
2. Load Qwen3-8B 4-bit và dự đoán outcome từ prepared contexts.

## Submission schema

```json
[
  {
    "case_id": "case_4101",
    "prediction": "A_WIN",
    "case_evidence": ["case_4101_chunk_3"],
    "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}]
  }
]
```

Không đưa `api_calls`, reasoning, evidence text hoặc token vào submission.

## Reproducibility và giới hạn

- Public data có gold annotations nhưng production inference không được đọc các trường đó.
- Case evidence recall/final score chỉ tính offline nếu có gold chunk IDs; nếu không, dùng public leaderboard.
- Cache API và checkpoint không được đưa vào source bundle gửi BTC trừ khi BTC yêu cầu.
- Source code không tự động nộp leaderboard.
- Model weights được pin bằng Hugging Face commit revision trong YAML config.
- Nếu model output sai JSON hai lần, case được đánh dấu `failed`; pipeline không tự dùng majority-label fallback và không tạo submission giả hợp lệ.

## Đóng gói source gửi BTC

```bash
python scripts/package_source.py
```

File mặc định được tạo tại `artifacts/alqac2026_source.zip`. Script chỉ đóng gói source/config/notebook/tests/docs, loại data, cache, model weights, logs, outputs và submissions, đồng thời dừng nếu phát hiện chuỗi giống ALQAC token.

Runbook đầy đủ: [docs/runbook.md](docs/runbook.md).

Theo dõi trạng thái experiment tại [docs/experiments.md](docs/experiments.md); khung technical report tại [docs/technical-report-outline.md](docs/technical-report-outline.md).
