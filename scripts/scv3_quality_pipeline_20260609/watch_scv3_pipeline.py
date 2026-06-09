#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/data/tse_cue_project")
OUT = ROOT / "diagnostics/similar_content_v3_quality_first_full_pipeline_20260609"
LOG = OUT / "logs/watchdog.log"
PIPE = OUT / "scripts/run_after_tts_full_pipeline.sh"
PID = OUT / "full_pipeline.pid"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def running(pid: str) -> bool:
    return bool(pid.strip()) and subprocess.run(f"kill -0 {pid.strip()} 2>/dev/null", shell=True).returncode == 0


def start_pipeline():
    p = subprocess.Popen(
        f"bash {PIPE}",
        shell=True,
        cwd=str(ROOT),
        stdout=open(OUT / "logs/full_pipeline.nohup.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID.write_text(str(p.pid), encoding="utf-8")
    return p.pid


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%F %T") + " " + msg + "\n")


def main():
    end = time.time() + 8 * 3600
    log("watchdog started")
    while time.time() < end:
        pid = PID.read_text(encoding="utf-8").strip() if PID.exists() else ""
        done = (OUT / "DONE").exists()
        if done:
            log("pipeline DONE")
            break
        if not running(pid):
            new_pid = start_pipeline()
            log(f"pipeline was not running, restarted pid={new_pid}")
        status = sh(
            "cd /data/tse_cue_project && "
            "BASE=diagnostics/similar_content_v3_tts_enroll_conflict_quality_first_20260609; "
            "echo tts $(wc -l < $BASE/logs/f5tts_quality_naturalized_shard0_gpu0.jsonl 2>/dev/null || echo 0) "
            "$(wc -l < $BASE/logs/f5tts_quality_naturalized_shard1_gpu1.jsonl 2>/dev/null || echo 0); "
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\\n' ';'"
        ).strip()
        log(status)
        time.sleep(600)
    summary = OUT / "scv3_quality_watchdog_summary.json"
    summary.write_text(json.dumps({"finished": (OUT / "DONE").exists(), "time": time.strftime("%F %T")}, indent=2), encoding="utf-8")
    log("watchdog ended")


if __name__ == "__main__":
    main()
