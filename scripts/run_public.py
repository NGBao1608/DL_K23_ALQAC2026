#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alqac2026.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run private-like public evaluation")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--input", default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use no API/GPU")
    parser.add_argument(
        "--execution-mode",
        choices=("mock", "cache-only", "live"),
        default=None,
    )
    parser.add_argument("--cache-db", default=None)
    parser.add_argument("--cache-backup-db", default=None)
    parser.add_argument("--law-index-dir", default=None)
    parser.add_argument("--selection-profile", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--max-network-calls",
        type=int,
        default=None,
        help="Required explicit HTTP-attempt cap for live runs",
    )
    args = parser.parse_args()
    result = run_experiment(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        resume_run=args.resume_run,
        public_gold_path=args.input,
        mock=args.mock,
        limit=args.limit,
        cache_db=args.cache_db,
        max_network_calls=args.max_network_calls,
        execution_mode=args.execution_mode,
        cache_backup_db=args.cache_backup_db,
        law_index_dir=args.law_index_dir,
        selection_profile=args.selection_profile,
        adapter_path=args.adapter_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
