#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from alqac2026.runner import run_experiment


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run ALQAC private inference")
    parser.add_argument("--config", default="configs/candidate.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use no API/GPU")
    parser.add_argument("--cache-db", default=None)
    parser.add_argument(
        "--max-network-calls",
        type=int,
        default=None,
        help="Required explicit HTTP-attempt cap for every non-mock run",
    )
    args = parser.parse_args()
    result = run_experiment(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        resume_run=args.resume_run,
        mock=args.mock,
        limit=args.limit,
        cache_db=args.cache_db,
        max_network_calls=args.max_network_calls,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
