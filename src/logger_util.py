"""统一实验日志：追加写入，带时间戳与可选 error_type。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(
    filename: str,
    message: str,
    *,
    phase: str | None = None,
    task: str | None = None,
    error_type: str | None = None,
    also_journal: bool = False,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    parts = [f"[{_ts()}]"]
    if phase:
        parts.append(f"PHASE={phase}")
    if task:
        parts.append(f"TASK={task}")
    if error_type:
        parts.append(f"error_type={error_type}")
    parts.append(message)
    line = " | ".join(parts) + "\n"
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    if also_journal:
        with (LOG_DIR / "experiment_journal.txt").open("a", encoding="utf-8") as f:
            f.write(line)


def journal(message: str) -> None:
    log("experiment_journal.txt", message, also_journal=False)
