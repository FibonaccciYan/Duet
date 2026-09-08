#!/usr/bin/env python3
"""Print one idle GPU index from nvidia-smi, or fail."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-util", type=int, default=5)
    parser.add_argument(
        "--require-index",
        type=int,
        default=None,
        help="validate this GPU index is idle instead of selecting the first idle GPU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        print(f"failed to query nvidia-smi: {exc}", file=sys.stderr)
        return 2
    parsed = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, memory, util = map(int, parts)
        parsed.append((index, memory, util))
    if args.require_index is not None:
        for index, memory, util in parsed:
            if index == args.require_index:
                if memory <= args.max_memory_mib and util <= args.max_util:
                    print(index)
                    return 0
                print(
                    f"GPU {index} is not idle: memory={memory} MiB, util={util}%",
                    file=sys.stderr,
                )
                return 1
        print(f"GPU {args.require_index} not found", file=sys.stderr)
        return 1
    for index, memory, util in parsed:
        if memory <= args.max_memory_mib and util <= args.max_util:
            print(index)
            return 0
    print("no idle GPU found", file=sys.stderr)
    print(output, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
