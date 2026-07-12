# ALQAC 2026 Architecture

## Design goals

- Một production pipeline dùng chung cho CLI, Kaggle và Colab.
- Private inference chỉ nhận `case_id` và `case_query`.
- API calls, model outputs, config và source revision có thể truy vết.
- Chạy được theo stage trên GPU T4 16 GB.
- Submission luôn qua strict validator và không được upload tự động.

## Data boundary

`load_inference_cases()` chuyển cả public/private JSON thành `InferenceCase(case_id, case_query)`. Object này không có gold fields.

Public evaluation đọc gold bằng `load_public_gold()` sau khi inference hoàn tất. Các trường `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict` và `related_law_provisions` không bao giờ được truyền vào `ALQACPipeline`.

```text
Public JSON ──→ InferenceCase ──→ production pipeline ──→ PredictionResult
      └──────→ PublicGold ─────────────────────────────→ evaluator

Private JSON ─→ InferenceCase ──→ production pipeline ──→ submission
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, evidence, label và result types |
| `data.py` | Load/validate public, private và law corpus; resolve public law gold |
| `case_retrieval.py` | Query generation, API client, rate limit, retry, SQLite cache |
| `law_retrieval.py` | BM25, dense embeddings, RRF và reranking |
| `prediction.py` | Prompt, Qwen3 backend, JSON parser và repair |
| `pipeline.py` | Prepare context, predict, checkpoint serialization |
| `evaluation.py` | Public outcome accuracy, law micro F1 và optional case recall |
| `submission.py` | Build official schema và strict validation |
| `runner.py` | Stage orchestration, artifacts, resume và manifest |

Dependency chỉ đi theo hướng:

```text
scripts/notebooks → runner → pipeline
pipeline → case_retrieval, law_retrieval, prediction
evaluation/submission → schemas
```

Notebook không định nghĩa lại production functions, không đổi working directory trong code, không sửa `sys.path` và không monkey-patch API client.

## Case evidence stage

Mỗi case dùng tối đa tám query theo thứ tự:

1. Quyết định của Tòa án tuyên xử.
2. Chấp nhận yêu cầu khởi kiện của nguyên đơn.
3. Không chấp nhận yêu cầu của nguyên đơn.
4. Nhận định của Hội đồng xét xử.
5. Áp dụng điều luật.
6. Nghĩa vụ án phí dân sự sơ thẩm.
7. Cụm `tranh chấp ...` trích từ `case_query`, nếu có.
8. `case_query` gốc đã chuẩn hóa.

Client áp dụng khoảng cách tối thiểu 5 giây giữa network calls, exponential retry cho `429/5xx`, dừng ngay với `403/422`, và cache response thành công theo hash của API version + case ID + normalized query.

Evidence được deduplicate theo `chunk_id`; nếu cùng chunk xuất hiện nhiều lần, giữ hit có score cao nhất. `api_calls` chỉ đếm network calls của run, không đưa vào submission.

## Law retrieval stage

`baseline.yaml`:

```text
case query + case evidence → BM25 → top 5
```

`candidate.yaml`:

```text
BM25 top 50 ─────┐
                 ├→ RRF(k=60) top 30 → Vietnamese_Reranker → top 5
Dense top 50 ────┘
```

Dense embeddings dùng normalized vectors và dot product. Index gồm 3.352 articles, được cache cùng model revision và danh sách article keys; cache sai metadata sẽ được build lại.

Sau khi chuẩn bị contexts cho tất cả cases, embedding/reranker được release và CUDA cache được dọn trước khi load Qwen3.

## Outcome stage

Qwen3 nhận:

- `case_query`.
- Tối đa tám case chunks, ưu tiên decision/accepted/rejected/reasoning.
- Tối đa năm law articles.

Runtime mặc định: Qwen3-8B, NF4 4-bit, double quantization, FP16 compute, thinking disabled, input tối đa 7.000 tokens và output tối đa 384 tokens.

Model phải trả `{"reasoning":"...","label":"..."}`. Parser chỉ chấp nhận bốn official labels. Nếu parse lỗi, model được gọi thêm đúng một lần bằng repair prompt; lỗi lần hai tạo `PredictionResult(status="failed")`. Submission builder từ chối failed result.

## Cache, checkpoint and resume

- `cache/case_api.sqlite`: persistent successful API responses.
- `cache/law_index/`: dense embeddings và metadata.
- `<run>/contexts.checkpoint.json`: prepared case/law contexts.
- `<run>/predictions.checkpoint.json`: completed/failed prediction records.
- `--resume-run <directory>`: dùng lại checkpoint; config phải khớp tuyệt đối với `config.resolved.json`.

Khi resume, completed predictions được lọc trước retrieval/model loading. Prepared contexts được lấy từ checkpoint; API/law retrieval chỉ chạy cho contexts còn thiếu. Mock mode, limit, input, config và source fingerprint là một phần của run identity và không được thay đổi khi resume.

## Evaluation limitations

- Outcome accuracy tính trên toàn bộ public cases; failed prediction được tính sai.
- Law evidence dùng micro F1 trên các gold provisions resolve được về corpus hiện hành.
- Public file hiện không có official gold `chunk_id`; `case_evidence_recall` và `final_score` vì vậy trả `null` offline.
- Official Penalized Case Recall và final score phải lấy từ leaderboard khi BTC không cung cấp gold chunks/công thức executable.

## Artifact layout

```text
<run>/
├── config.resolved.json
├── environment.json
├── contexts.checkpoint.json
├── predictions.checkpoint.json
├── predictions.json
├── api_stats.json
├── submission.json
├── validation.json
├── manifest.json
├── metrics.json              # public run only
└── errors.json               # public run only
```

Manifest lưu model names/revisions, license metadata, corpus SHA-256, Git commit, dirty state, resolved config, case count và API call count.
