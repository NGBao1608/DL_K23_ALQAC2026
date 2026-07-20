#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alqac2026.data import validate_private_input
from alqac2026.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALQAC private inference")
    parser.add_argument("--config", default="configs/candidate.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="submissions")
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
    validate_private_input(args.input)
    mode = args.execution_mode or ("mock" if args.mock else "live")
    if mode == "live" and not args.selection_profile:
        parser.error("Private live runs require --selection-profile")
    result = run_experiment(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        resume_run=args.resume_run,
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
