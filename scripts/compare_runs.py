#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ALQAC run metrics")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    rows = []
    for value in args.runs:
        run = Path(value)
        metrics_path = run / "metrics.json" if run.is_dir() else run
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"run": str(run), **metrics})
    rows.sort(
        key=lambda row: (
            -(row.get("outcome_accuracy") or 0.0),
            -(row.get("law_micro_f1") or 0.0),
            row.get("format_failures") or 0,
        )
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

