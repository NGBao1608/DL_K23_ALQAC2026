#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alqac2026.rescore import rescore_prepared_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore immutable PreparedCase artifacts without Case API access"
    )
    parser.add_argument("--config", default="configs/candidate_rescore_v1.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prepared-contexts", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--public-gold", default=None)
    parser.add_argument("--selection-profile", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--corpus",
        default=None,
        help="Override the law corpus; required for Private prepared contexts",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = rescore_prepared_cases(
        config_path=args.config,
        input_path=args.input,
        prepared_contexts_path=args.prepared_contexts,
        run_dir=args.run_dir,
        public_gold_path=args.public_gold,
        selection_profile=args.selection_profile,
        adapter_path=args.adapter_path,
        corpus_path=args.corpus,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
