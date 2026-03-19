#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "reports" / "analysis" / "all_eval_core_metrics_20260317.csv"
LEGACY_INDEX_CSV = REPO_ROOT / "results" / "reports" / "analysis" / "all_test_core_metrics_20260311.csv"
CHRONOLOGY_MD = REPO_ROOT / "ALL_TRAIN_LOGS_PARAMS_EVAL_CHRONOLOGY_20260311.md"

MAIN_METRIC_PATTERNS = {
    "abs_rel_down": re.compile(r"^_abs_rel_diff\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "sq_rel_down": re.compile(r"^_squ_rel_diff\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "rms_linear_down": re.compile(r"^_RMS_linear\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "rms_log_down": re.compile(r"^_RMS_log\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "si_log_down": re.compile(r"^_SILog\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "delta_1_25_up": re.compile(r"^_threshold_delta_1\.25\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "delta_1_25_2_up": re.compile(r"^_threshold_delta_1\.25\^2\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
    "delta_1_25_3_up": re.compile(r"^_threshold_delta_1\.25\^3\s*:\s*([-+0-9.eE]+)\s*$", re.MULTILINE),
}


@dataclass(frozen=True)
class EvalRecord:
    sort_datetime_utc: str
    date_utc: str
    run_name: str
    split: str
    source_kind: str
    abs_rel_down: str
    sq_rel_down: str
    rms_linear_down: str
    rms_log_down: str
    si_log_down: str
    delta_1_25_up: str
    delta_1_25_2_up: str
    delta_1_25_3_up: str
    source_log: str


def chronology_rms_linear_map() -> dict[str, str]:
    if not CHRONOLOGY_MD.exists():
        return {}

    mapping: dict[str, str] = {}
    for line in CHRONOLOGY_MD.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "`runs/eval/" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 8:
            continue
        source_path_match = re.search(r"`([^`]+)`", line)
        if not source_path_match:
            continue
        source_path = source_path_match.group(1)
        rms_linear = parts[5]
        try:
            mapping[str((REPO_ROOT / source_path).resolve())] = f"{float(rms_linear):.6f}"
        except ValueError:
            continue
    return mapping


def detect_split(path: Path) -> str:
    lowered = str(path).lower()
    if "/runs/test/" in lowered:
        return "test"
    if "/runs/val/" in lowered:
        return "val"
    name = path.parent.name if path.parent.name != "run_logs" else path.stem
    lowered = name.lower()
    if lowered.startswith("test_"):
        return "test"
    if lowered.startswith("val_"):
        return "val"
    if "offline" in lowered or "mkl" in lowered:
        return "offline_eval"
    return "unknown"


def detect_split_from_text(text: str) -> str:
    lowered = text.lower()
    if "/runs/test/" in lowered:
        return "test"
    if "/runs/val/" in lowered:
        return "val"
    if "offline" in lowered or "saved npy" in lowered:
        return "offline_eval"
    return "unknown"


def detect_source_kind(path: Path) -> str:
    if path.parent.name == "run_logs":
        return "run_log_stdout"
    return "eval_metrics_log"


def record_time_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def extract_metrics(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for field, pattern in MAIN_METRIC_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"missing metric {field}")
        metrics[field] = f"{float(match.group(1)):.6f}"
    return metrics


def build_record(path: Path) -> EvalRecord:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = extract_metrics(text)
    dt_utc = record_time_utc(path)
    run_name = path.parent.name if path.parent.name != "run_logs" else path.stem
    split = detect_split(path)
    if split == "unknown":
        split = detect_split_from_text(text)
    return EvalRecord(
        sort_datetime_utc=dt_utc.isoformat(timespec="seconds"),
        date_utc=dt_utc.date().isoformat(),
        run_name=run_name,
        split=split,
        source_kind=detect_source_kind(path),
        abs_rel_down=metrics["abs_rel_down"],
        sq_rel_down=metrics["sq_rel_down"],
        rms_linear_down=metrics["rms_linear_down"],
        rms_log_down=metrics["rms_log_down"],
        si_log_down=metrics["si_log_down"],
        delta_1_25_up=metrics["delta_1_25_up"],
        delta_1_25_2_up=metrics["delta_1_25_2_up"],
        delta_1_25_3_up=metrics["delta_1_25_3_up"],
        source_log=str(path),
    )


def collect_eval_logs() -> list[Path]:
    logs = []
    for path in (REPO_ROOT / "runs" / "eval").rglob("eval_metrics*.log"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        logs.append(path)
    logs.extend((REPO_ROOT / "run_logs").glob("eval_*.log"))
    return sorted({path.resolve() for path in logs})


def load_legacy_records(existing_source_logs: set[str]) -> list[EvalRecord]:
    if not LEGACY_INDEX_CSV.exists():
        return []

    chronology_rms = chronology_rms_linear_map()
    records: list[EvalRecord] = []
    with LEGACY_INDEX_CSV.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            source_log = str(Path(row["source_log"]).resolve())
            if source_log in existing_source_logs:
                continue
            date_utc = row["date_utc"]
            sort_datetime_utc = f"{date_utc}T00:00:00+00:00"
            run_name = row["run_name"]
            result_kind = row.get("result_kind", "")
            lowered = f"{run_name} {result_kind} {source_log}".lower()
            if "offline" in lowered or "mkl" in lowered:
                split = "offline_eval"
            elif "val" in lowered:
                split = "val"
            elif "test" in lowered:
                split = "test"
            else:
                split = "unknown"
            records.append(
                EvalRecord(
                    sort_datetime_utc=sort_datetime_utc,
                    date_utc=date_utc,
                    run_name=run_name,
                    split=split,
                    source_kind="legacy_index_csv",
                    abs_rel_down=f"{float(row['abs_rel_down']):.6f}",
                    sq_rel_down=f"{float(row['sq_rel_down']):.6f}",
                    rms_linear_down=chronology_rms.get(source_log, ""),
                    rms_log_down=f"{float(row['rms_log_down']):.6f}",
                    si_log_down=f"{float(row['si_log_down']):.6f}",
                    delta_1_25_up=f"{float(row['delta_1_25_up']):.6f}",
                    delta_1_25_2_up=f"{float(row['delta_1_25_2_up']):.6f}",
                    delta_1_25_3_up=f"{float(row['delta_1_25_3_up']):.6f}",
                    source_log=source_log,
                )
            )
    return records


def write_csv(records: list[EvalRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvalRecord.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main() -> None:
    records = [build_record(path) for path in collect_eval_logs()]
    records.extend(load_legacy_records({record.source_log for record in records}))
    records.sort(key=lambda item: (item.sort_datetime_utc, item.run_name, item.source_log))
    write_csv(records, DEFAULT_OUTPUT)
    print(f"Wrote {len(records)} rows to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
