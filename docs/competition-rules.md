# Competition rules used by the implementation

Nguồn chính: [ALQAC 2026 official website](https://sites.google.com/view/alqac2026) và thông báo BTC gửi cho đội. Khi các nguồn mâu thuẫn, xác nhận trực tiếp từ BTC được ưu tiên và tài liệu này phải được cập nhật.

## Task

Input của mỗi case:

```json
{"case_id":"case_4101","case_query":"Mô tả ngắn tranh chấp..."}
```

Private test không cung cấp nội dung bản án hoặc gold annotations. Nội dung vụ án phải được truy hồi qua Case Content API.

Prediction labels:

- `A_WIN`: chấp nhận toàn bộ main claim của nguyên đơn.
- `PARTIAL_A_WIN`: chấp nhận một phần lớn hơn 50%.
- `PARTIAL_B_WIN`: chấp nhận một phần không quá 50%.
- `B_WIN`: bác toàn bộ main claim.

Nếu case có nhiều yêu cầu, prediction tập trung vào main claim được mô tả trong `case_query`.

## Resources and API

- Public development set: 50 labeled cases do BTC cung cấp.
- Private test: chỉ `case_id` và `case_query`.
- Law corpus: 18 legal documents, 3.352 articles; submission định danh provision bằng `law_id` và `aid`.
- Case API: `POST https://alqac-api.ngrok.pro/retrieve` với header `X-API-Key` và body `{"query":"...","case_id":"..."}`.
- Mỗi call trả top-1 segment; rate limit là một request mỗi 5 giây/team.

## Evaluation

Official score gồm:

```text
0.70 × Outcome Accuracy
+ 0.20 × Penalized Case Evidence Recall
+ 0.10 × Micro Law Evidence F1
```

Case evidence recall bị phạt theo API efficiency: full efficiency đến `2n` calls và giảm về 0 tại `5n`, với `n` là số segment của case. Code không tự suy đoán phần công thức chi tiết chưa được BTC cung cấp dưới dạng executable.

## Submission

Nộp một JSON array, mỗi test case đúng một object gồm:

- `case_id`.
- `prediction`.
- `case_evidence`: danh sách official `chunk_id`, có thể rỗng.
- `law_evidence`: danh sách `{law_id, aid}` hợp lệ.

Không có duplicate/missing/unknown case ID hoặc evidence identifier. Upload leaderboard được thực hiện thủ công, tối đa ba submissions/ngày/team.

## Restrictions

- Chỉ dùng open-weight model dưới 10B parameters.
- Không dùng ChatGPT/GPT, Claude, Gemini hoặc proprietary model APIs trong pipeline.
- Không dùng externally annotated datasets được tạo riêng cho legal QA/legal entailment.
- Online legal databases có thể được truy vấn theo quy định BTC, nhưng evidence nộp phải dùng identifiers chính thức.
- BTC có thể yêu cầu source code, config và logs để kiểm tra reproducibility.

## Project enforcement

- Inference schema chỉ có `case_id` và `case_query`.
- Token chỉ đọc từ environment/Kaggle/Colab Secrets.
- Source code không tự upload submission.
- Validator từ chối field thừa như `api_calls`, failed predictions và law identifiers không có trong corpus.

