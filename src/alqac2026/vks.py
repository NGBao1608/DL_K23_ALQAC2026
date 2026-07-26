"""Procuracy (Viện Kiểm Sát) stance signal.

The Case Content API serves the procuracy's recommendation to the trial panel
("Đề nghị Hội đồng xét xử: chấp nhận một phần yêu cầu khởi kiện...") but not the
court's own decision. In Vietnamese first-instance civil trials the panel usually
follows that recommendation closely, so its stance is a useful, near-leak-free
predictor of the outcome.

This module locates the recommendation text and normalises the stance. It is used
to (a) surface the recommendation at the top of the outcome prompt and (b) act as
a fallback prior when the model output is unparseable. It reads only retrieved
evidence text — never a gold field.
"""
from __future__ import annotations

import re

from .schemas import OutcomeLabel


# Markers of the procuracy's opinion block. The recommendation verb "đề nghị"
# combined with the panel ("hội đồng xét xử") also marks it when the VKS noun is
# in a preceding sentence.
_VKS_MARKER = re.compile(
    r"(viện\s*kiểm\s*sát|kiểm\s*sát\s*viên|đại\s*diện\s*viện\s*kiểm"
    r"|đề\s*nghị\s*hội\s*đồng\s*xét\s*xử)",
    re.IGNORECASE,
)
# Window after the marker holding the enumerated recommendation ("1. Chấp nhận
# một phần ... 2. ..."). Kept wide because items are period-separated.
_WINDOW_CHARS = 500

_PARTIAL = re.compile(r"một\s*phần", re.IGNORECASE)
_REJECT = re.compile(
    r"(không\s*chấp\s*nhận|bác\s+(toàn\s+bộ\s+)?(yêu\s*cầu|đơn\s*khởi\s*kiện|khởi\s*kiện))",
    re.IGNORECASE,
)
_FULL = re.compile(r"chấp\s*nhận", re.IGNORECASE)


STANCE_TO_LABEL = {
    "ACCEPT_FULL": OutcomeLabel.A_WIN,
    "ACCEPT_PARTIAL": OutcomeLabel.PARTIAL_A_WIN,
    "REJECT": OutcomeLabel.B_WIN,
}


def _vks_window(text: str) -> str | None:
    """Return the recommendation block (marker + following window), or None."""
    marker = _VKS_MARKER.search(text)
    if not marker:
        return None
    return text[marker.start() : marker.start() + _WINDOW_CHARS]


def extract_vks_recommendation(text: str) -> str | None:
    """Return a cleaned snippet of the procuracy's recommendation block, or None."""
    window = _vks_window(text)
    if window is None:
        return None
    return re.sub(r"\s+", " ", window).strip()[:300]


def vks_stance(text: str) -> str:
    """Normalise the procuracy stance to ACCEPT_FULL/ACCEPT_PARTIAL/REJECT/UNKNOWN.

    Partial is checked before full because "chấp nhận một phần" contains "chấp nhận".
    """
    window = _vks_window(text)
    if window is None:
        return "UNKNOWN"
    if _PARTIAL.search(window):
        return "ACCEPT_PARTIAL"
    if _REJECT.search(window):
        return "REJECT"
    if _FULL.search(window):
        return "ACCEPT_FULL"
    return "UNKNOWN"


def vks_label(text: str) -> OutcomeLabel | None:
    """Map the procuracy stance directly to an outcome label, or None if unknown."""
    return STANCE_TO_LABEL.get(vks_stance(text))
