from alqac2026.schemas import OutcomeLabel
from alqac2026.vks import extract_vks_recommendation, vks_label, vks_stance


PARTIAL = (
    "Tại phiên tòa, đại diện Viện kiểm sát phát biểu. "
    "Đề nghị Hội đồng xét xử: 1. Chấp nhận một phần yêu cầu khởi kiện của nguyên đơn; "
    "buộc bị đơn bồi thường 50.000.000 đồng."
)
REJECT = (
    "Kiểm sát viên đề nghị Hội đồng xét xử không chấp nhận toàn bộ yêu cầu khởi kiện "
    "của nguyên đơn."
)
FULL = (
    "Đại diện Viện kiểm sát đề nghị Hội đồng xét xử chấp nhận yêu cầu khởi kiện của "
    "nguyên đơn, buộc bị đơn thực hiện nghĩa vụ."
)


def test_partial_stance_beats_full_keyword():
    assert vks_stance(PARTIAL) == "ACCEPT_PARTIAL"
    assert vks_label(PARTIAL) is OutcomeLabel.PARTIAL_A_WIN


def test_reject_stance():
    assert vks_stance(REJECT) == "REJECT"
    assert vks_label(REJECT) is OutcomeLabel.B_WIN


def test_full_stance():
    assert vks_stance(FULL) == "ACCEPT_FULL"
    assert vks_label(FULL) is OutcomeLabel.A_WIN


def test_no_vks_text_is_unknown():
    text = "Nguyên đơn trình bày về nguồn gốc thửa đất và đề nghị Tòa án giải quyết."
    assert vks_stance(text) == "UNKNOWN"
    assert vks_label(text) is None
    assert extract_vks_recommendation(text) is None


def test_recommendation_snippet_extracted():
    snippet = extract_vks_recommendation(PARTIAL)
    assert snippet and "một phần" in snippet
