# ALQAC 2026 — Agentic RAG cho Dự đoán Kết quả Vụ án Dân sự

Repo của team **K23** cho cuộc thi **ALQAC 2026** (Automated Legal Question Answering Competition, tổ chức cùng hội nghị KSE). Bài toán năm nay: cho một mô tả ngắn về vụ tranh chấp dân sự, hệ thống tự đi tìm bằng chứng và điều luật rồi **dự đoán kết quả vụ kiện sơ thẩm**.

## Bài toán

Đầu vào là `case_query` (một đoạn tiếng Việt mô tả tranh chấp). Đầu ra gồm ba phần:

- **Dự đoán kết quả** — một trong bốn nhãn: `A_WIN` (nguyên đơn thắng toàn bộ), `PARTIAL_A_WIN` (thắng một phần), `B_WIN` (bị đơn thắng), `PARTIAL_B_WIN` (bị đơn thắng phần lớn).
- **Bằng chứng vụ án** (`case_evidence`) — các đoạn nội dung vụ án, lấy qua Evidence Retrieval API của BTC.
- **Điều luật liên quan** (`law_evidence`) — truy hồi từ corpus luật.

Điểm được tính theo công thức:

```
FinalScore = 0.70 · Outcome Accuracy + 0.20 · Penalized Case Recall + 0.10 · Micro Law F1
```

Trong đó Penalized Case Recall có phạt theo số lần gọi API (không phạt tới `2·n`, giảm về 0 ở `5·n`, với `n` là số đoạn của vụ).

## Dữ liệu

- `corpus_law_pub.json` — 18 bộ luật, 3.352 điều (BLDS, BLTTDS, Đất đai, HN&GĐ...).
- `ALQAC2026_public_test.json` — 50 vụ dân sự có đầy đủ nhãn, dùng làm tập phát triển.

Lưu ý: **tập private (tính điểm) chỉ cho `case_id` + `case_query`**, nội dung vụ án bị giấu và chỉ truy cập được qua API. Mọi pipeline không được phụ thuộc vào các trường chi tiết chỉ có ở public test.

## Cách tiếp cận

Pipeline dạng Agentic RAG, toàn bộ dùng model **mã nguồn mở dưới 10B tham số** (đúng luật thi — cấm GPT/Claude/Gemini trong hệ thống):

1. **Gom bằng chứng** — sinh nhiều truy vấn nhắm các đoạn *Quyết định / Nhận định của Tòa*, gọi Evidence API để lấy các `chunk_id` liên quan.
2. **Truy hồi luật** — BM25 trên corpus, xuất `law_evidence` dạng `{law_id, aid}`.
3. **Suy luận & dự đoán** — đưa bằng chứng + điều luật vào **Qwen2.5-7B-Instruct** (nạp 4-bit), prompt theo tam đoạn luận pháp lý (IRAC), nhấn mạnh phân biệt thắng-toàn-bộ / thắng-một-phần.

Model đặt trong `src/`, chi tiết pipeline trong notebook `ALQAC2026_final.ipynb`.

## Kết quả (trên public test, 50 vụ)

| Thành phần | Kết quả |
|---|---|
| Outcome Accuracy | ~52% |
| Penalized Case Recall | ~24% |
| Micro Law F1 | ~16% |
| **Final Score** | **~0.43** |

Đây là điểm của một hệ thống thật (không dùng đáp án có sẵn trong file public), nên kỳ vọng giữ được khi sang private test. Phân tích chi tiết và các hướng đã thử (precedent-RAG, hybrid retrieval...) nằm trong báo cáo.

## Cách chạy

Notebook thiết kế cho Google Colab (GPU T4):

1. Mở `ALQAC2026_final.ipynb` trên Colab, chọn Runtime ▸ GPU (T4).
2. Upload `alqac2026_starter.zip` (code) và 2 file dữ liệu.
3. Đặt token của team qua Colab Secrets (tên `ALQAC_TEAM_TOKEN`).
4. Chạy lần lượt các cell → sinh ra `submission.json`.

Chạy local cũng được: `pip install -r requirements.txt`, cần một GPU cho model 7B.

## Cách nộp

Nộp bằng cách **upload file `submission.json` lên trang leaderboard**. Giới hạn **20 lần / 24 giờ**. File là một JSON array, mỗi vụ một object:

```json
{
  "case_id": "case_0001",
  "prediction": "A_WIN",
  "law_evidence": [{"law_id": "91/2015/QH13", "aid": 53373}],
  "case_evidence": ["case_0001_chunk_2"],
  "api_calls": 8
}
```

Nên tự chấm offline trên public test trước, chỉ nộp cấu hình tốt hơn.

## Cấu trúc repo

```
├── ALQAC2026_final.ipynb     # pipeline hoàn chỉnh (chạy trên Colab)
├── src/                      # code: load data, retrieval, agent, gọi API, ...
├── data/raw/                 # đặt corpus + public test vào đây (git-ignored)
├── requirements.txt
└── README.md
```

## Lưu ý

- **Không commit token bí mật** lên GitHub (đã chặn trong `.gitignore`). Dùng Colab Secrets hoặc file `.env`.
- Chỉ được dùng model open-weight < 10B; không gọi API model độc quyền trong pipeline.

---
Team K23 — ALQAC 2026.
