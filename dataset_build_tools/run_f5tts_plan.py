#!/usr/bin/env python3
"""Run F5-TTS generation from a JSONL plan with resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--f5-cli", default="f5-tts_infer-cli")
    args = parser.parse_args()

    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.plan_jsonl, args.limit)
    done = 0
    failed = 0
    with args.log_jsonl.open("a", encoding="utf-8") as log_f:
        for idx, row in enumerate(rows):
            if idx < args.start_index:
                continue
            out_path = Path(row["synthetic_interferer"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.stat().st_size > 1024 and not args.overwrite:
                status = {"index": idx, "key": row["key"], "status": "skip_exists", "path": str(out_path)}
                log_f.write(json.dumps(status, ensure_ascii=False) + "\n")
                log_f.flush()
                done += 1
                continue

            cmd = [
                args.f5_cli,
                "--ref_audio",
                row["prompt_audio"],
                "--ref_text",
                row.get("prompt_text", ""),
                "--gen_text",
                row["target_text"],
                "--output_file",
                str(out_path),
                "--speed",
                str(args.speed),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            status = {
                "index": idx,
                "key": row["key"],
                "status": "ok" if proc.returncode == 0 and out_path.exists() else "failed",
                "returncode": proc.returncode,
                "path": str(out_path),
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-2000:],
            }
            log_f.write(json.dumps(status, ensure_ascii=False) + "\n")
            log_f.flush()
            if status["status"] == "ok":
                done += 1
            else:
                failed += 1
                print(json.dumps(status, ensure_ascii=False), file=sys.stderr, flush=True)
            if (done + failed) % 10 == 0:
                print(f"processed={done + failed} ok={done} failed={failed}", flush=True)
    print(json.dumps({"processed": done + failed, "ok": done, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
