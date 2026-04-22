#!/usr/bin/env python3
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WORKDIR = Path("/root/shared-nvme/spikemamba")
RUN_SCRIPT = WORKDIR / "0323/run_0323_60plus60_halflr.sh"
DATA_ROOT = WORKDIR / "DENSE-spike"

STAGE1_CFG = WORKDIR / "0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_60ep_20260323.json"
LEGACY_STAGE2_CFG = WORKDIR / "0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to120_halflr_20260323.json"
STAGE2_KEEP_CFG = WORKDIR / "0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323.json"
STAGE3_HALF_CFG = WORKDIR / "0323/train_s2d_spiketransformer_mambassm_bidivim_nograd_2enc_ep100_to200_halflr_20260323.json"

MASTER_LOG = WORKDIR / "run_logs/0323/run_0323_60plus60_halflr.log"
STAGE1_LOG = WORKDIR / "run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323.log"
LEGACY_STAGE2_LOG = WORKDIR / "run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323.log"
STAGE2_KEEP_LOG = WORKDIR / "run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323.log"
STAGE3_HALF_LOG = WORKDIR / "run_logs/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323.log"

STAGE1_RUN_DIR = WORKDIR / "runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_60ep_20260323_bs2"
LEGACY_STAGE2_RUN_DIR = WORKDIR / "runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_bs2"
STAGE2_KEEP_RUN_DIR = WORKDIR / "runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323_bs2"
STAGE3_HALF_RUN_DIR = WORKDIR / "runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2"
ALT_STAGE3_HALF_RUN_DIR = Path("/root/runs/train/0323/train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_bs2")

RESUME_DIR = WORKDIR / "runs/resume/0323"
LEGACY_STAGE2_RESUME = RESUME_DIR / "train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to120_halflr_20260323_resume.pth.tar"
STAGE2_KEEP_RESUME = RESUME_DIR / "train_s2d_MambaSSM_bidivim_nograd_2enc_ep60_to100_keep_lr_20260323_resume.pth.tar"
STAGE3_HALF_RESUME = RESUME_DIR / "train_s2d_MambaSSM_bidivim_nograd_2enc_ep100_to200_halflr_20260323_resume.pth.tar"

STAGE2_KEEP_DIST_URL = "tcp://127.0.0.1:1274"
STAGE3_HALF_DIST_URL = "tcp://127.0.0.1:1275"


@dataclass
class EventSummary:
    event_file: Optional[Path]
    loss_epoch: Optional[int]
    loss: Optional[float]
    val_loss_epoch: Optional[int]
    val_loss: Optional[float]
    lr_epoch: Optional[int]
    learning_rate: Optional[float]


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def read_last_nonempty_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    return last


def run_command(args) -> str:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return result.stdout.strip()


def process_table() -> str:
    return run_command(["ps", "-ww", "-eo", "pid,args"])


def find_process_pid(pattern: str) -> Optional[int]:
    for line in process_table().splitlines()[1:]:
        if pattern in line and "monitor_0323_60plus60_halflr.py" not in line:
            try:
                return int(line.strip().split(None, 1)[0])
            except (ValueError, IndexError):
                continue
    return None


def find_process_pid_any(*patterns: str) -> Optional[int]:
    table = process_table().splitlines()[1:]
    for line in table:
        if "monitor_0323_60plus60_halflr.py" in line:
            continue
        if any(pattern in line for pattern in patterns):
            try:
                return int(line.strip().split(None, 1)[0])
            except (ValueError, IndexError):
                continue
    return None


def pid_exists(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_pid(pid: int, label: str, monitor_log: Path) -> None:
    if not pid_exists(pid):
        return
    append_line(monitor_log, f"time={iso_now()} | action=terminate | target={label} | pid={pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 8
    while time.time() < deadline:
        if not pid_exists(pid):
            return
        time.sleep(0.5)
    if pid_exists(pid):
        append_line(monitor_log, f"time={iso_now()} | action=kill | target={label} | pid={pid}")
        os.kill(pid, signal.SIGKILL)


def stage_processes() -> Dict[str, Optional[int]]:
    return {
        "wrapper_pid": find_process_pid(str(RUN_SCRIPT)),
        "stage1_pid": find_process_pid(str(STAGE1_CFG)),
        "legacy_stage2_pid": find_process_pid_any(str(LEGACY_STAGE2_RESUME), str(LEGACY_STAGE2_RUN_DIR)),
        "stage2_keep_pid": find_process_pid_any(str(STAGE2_KEEP_RESUME), str(STAGE2_KEEP_RUN_DIR)),
        "stage3_half_pid": find_process_pid_any(str(STAGE3_HALF_RESUME), str(STAGE3_HALF_RUN_DIR)),
    }


def latest_event_file(run_dir: Path) -> Optional[Path]:
    tensorboard_dir = run_dir / "tensorboard"
    candidates = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    return candidates[-1] if candidates else None


def load_event_summary(event_file: Path) -> EventSummary:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 1})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    def scalar(tag: str) -> Tuple[Optional[int], Optional[float]]:
        if tag not in scalar_tags:
            return None, None
        values = accumulator.Scalars(tag)
        if not values:
            return None, None
        return int(values[-1].step), float(values[-1].value)

    loss_epoch, loss = scalar("loss")
    val_loss_epoch, val_loss = scalar("val_loss")
    lr_epoch, learning_rate = scalar("learning_rate")
    return EventSummary(
        event_file=event_file,
        loss_epoch=loss_epoch,
        loss=loss,
        val_loss_epoch=val_loss_epoch,
        val_loss=val_loss,
        lr_epoch=lr_epoch,
        learning_rate=learning_rate,
    )


def latest_checkpoint_path(run_dir: Path) -> Optional[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.pth.tar"))
    return candidates[-1] if candidates else None


def latest_checkpoint_name(run_dir: Path) -> str:
    checkpoint = latest_checkpoint_path(run_dir)
    return checkpoint.name if checkpoint else ""


def checkpoint_epoch(checkpoint_name: str) -> Optional[int]:
    match = re.search(r"epoch(\d+)", checkpoint_name)
    return int(match.group(1)) if match else None


def find_epoch_checkpoint(run_dir: Path, epoch: int) -> Optional[Path]:
    pattern = f"checkpoint-epoch{epoch:03d}-loss-*.pth.tar"
    matches = sorted((run_dir / "checkpoints").glob(pattern))
    return matches[-1] if matches else None


def newest_mtime(run_dir: Path) -> float:
    newest = 0.0
    for relative in ("tensorboard", "checkpoints", "config.json"):
        path = run_dir / relative
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    latest_event = latest_event_file(run_dir)
    if latest_event and latest_event.exists():
        newest = max(newest, latest_event.stat().st_mtime)
    latest_ckpt = latest_checkpoint_path(run_dir)
    if latest_ckpt and latest_ckpt.exists():
        newest = max(newest, latest_ckpt.stat().st_mtime)
    return newest


def candidate_run_dirs(stage: str) -> Tuple[Path, ...]:
    if stage == "stage3_halflr":
        return (STAGE3_HALF_RUN_DIR, ALT_STAGE3_HALF_RUN_DIR)
    if stage == "stage2_keep_lr":
        return (STAGE2_KEEP_RUN_DIR,)
    return (STAGE1_RUN_DIR,)


def resolve_run_dir(stage: str) -> Path:
    candidates = candidate_run_dirs(stage)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0]
    return max(existing, key=newest_mtime)


def gpu_summary() -> str:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return query.splitlines()[0].strip() if query else ""


def infer_stage(processes: Dict[str, Optional[int]]) -> str:
    if processes["stage3_half_pid"] or STAGE3_HALF_RUN_DIR.exists():
        return "stage3_halflr"
    if processes["stage2_keep_pid"] or STAGE2_KEEP_RUN_DIR.exists():
        return "stage2_keep_lr"
    return "stage1"


def format_summary(stage: str, summary: EventSummary, latest_ckpt: str, processes: Dict[str, Optional[int]], master_tail: str) -> str:
    parts = [
        f"time={iso_now()}",
        f"stage={stage}",
        f"wrapper_pid={processes['wrapper_pid'] or ''}",
        f"stage1_pid={processes['stage1_pid'] or ''}",
        f"legacy_stage2_pid={processes['legacy_stage2_pid'] or ''}",
        f"stage2_keep_pid={processes['stage2_keep_pid'] or ''}",
        f"stage3_half_pid={processes['stage3_half_pid'] or ''}",
        f"gpu={gpu_summary()}",
        f"event={summary.event_file.name if summary.event_file else ''}",
        f"loss_epoch={summary.loss_epoch if summary.loss_epoch is not None else ''}",
        f"loss={summary.loss:.12f}" if summary.loss is not None else "loss=",
        f"val_loss_epoch={summary.val_loss_epoch if summary.val_loss_epoch is not None else ''}",
        f"val_loss={summary.val_loss:.12f}" if summary.val_loss is not None else "val_loss=",
        f"learning_rate_epoch={summary.lr_epoch if summary.lr_epoch is not None else ''}",
        f"learning_rate={summary.learning_rate:.12f}" if summary.learning_rate is not None else "learning_rate=",
        f"latest_ckpt={latest_ckpt}",
        f"master_tail={master_tail}",
    ]
    return " | ".join(parts)


def write_blocker_note(block_dir: Path) -> None:
    note = block_dir / "README_BLOCKED.txt"
    note.write_text(
        "This path is intentionally blocked to prevent the live 0323 wrapper\n"
        "from starting the legacy epoch60->120 half-lr continuation.\n"
        "The active monitor will instead continue epoch60->100 at lr=1e-4,\n"
        "then epoch100->200 at lr=5e-5, preserving the first 60 epochs.\n",
        encoding="utf-8",
    )


def ensure_legacy_stage2_blocked(monitor_log: Path) -> None:
    if LEGACY_STAGE2_RESUME.is_dir():
        note = LEGACY_STAGE2_RESUME / "README_BLOCKED.txt"
        if not note.exists():
            write_blocker_note(LEGACY_STAGE2_RESUME)
        return
    if LEGACY_STAGE2_RESUME.exists():
        append_line(
            monitor_log,
            f"time={iso_now()} | level=warning | detail=legacy resume path already exists as file | path={LEGACY_STAGE2_RESUME}",
        )
        return
    LEGACY_STAGE2_RESUME.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_STAGE2_RESUME.mkdir()
    write_blocker_note(LEGACY_STAGE2_RESUME)
    append_line(
        monitor_log,
        f"time={iso_now()} | action=block_legacy_stage2 | path={LEGACY_STAGE2_RESUME}",
    )
    append_line(
        MASTER_LOG,
        f"[{iso_now()}] schedule override: preserve epoch1-60, continue to epoch100 at lr=1e-4, then epoch100-200 at lr=5e-5",
    )


def prepare_resume(base_ckpt: Path, cfg_path: Path, out_path: Path) -> Path:
    checkpoint = torch.load(base_ckpt, map_location="cpu")
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint["config"] = config

    lr = config["optimizer"]["lr"]
    for group in checkpoint["optimizer"]["param_groups"]:
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    return out_path


def launch_resume_training(resume_path: Path, log_path: Path, dist_url: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        subprocess.Popen(
            [
                "/usr/bin/python",
                str(WORKDIR / "train_parallel.py"),
                "-r",
                str(resume_path),
                "-f",
                str(DATA_ROOT),
                "--gpu",
                "0",
                "--multiprocessing_distributed",
                "--dist_url",
                dist_url,
            ],
            cwd=str(WORKDIR),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def start_stage2_keep_if_ready(monitor_log: Path, processes: Dict[str, Optional[int]], current_epoch: Optional[int]) -> bool:
    if processes["stage2_keep_pid"] or STAGE2_KEEP_RUN_DIR.exists():
        return False
    if current_epoch is None or current_epoch < 60 or processes["stage1_pid"]:
        return False

    ckpt = find_epoch_checkpoint(STAGE1_RUN_DIR, 60)
    if not ckpt:
        return False

    if processes["wrapper_pid"]:
        terminate_pid(processes["wrapper_pid"], "legacy_wrapper_after_stage1", monitor_log)

    resume_path = prepare_resume(ckpt, STAGE2_KEEP_CFG, STAGE2_KEEP_RESUME)
    append_line(
        monitor_log,
        f"time={iso_now()} | action=start_stage2_keep_lr | base_ckpt={ckpt.name} | resume={resume_path.name}",
    )
    append_line(
        MASTER_LOG,
        f"[{iso_now()}] stage2 start: continue to 100 epochs, lr=1e-4, preserving epoch1-60",
    )
    launch_resume_training(resume_path, STAGE2_KEEP_LOG, STAGE2_KEEP_DIST_URL)
    return True


def start_stage3_half_if_ready(monitor_log: Path, processes: Dict[str, Optional[int]], current_epoch: Optional[int]) -> bool:
    if processes["stage3_half_pid"] or STAGE3_HALF_RUN_DIR.exists():
        return False
    if current_epoch is None or current_epoch < 100 or processes["stage2_keep_pid"]:
        return False

    ckpt = find_epoch_checkpoint(STAGE2_KEEP_RUN_DIR, 100)
    if not ckpt:
        return False

    resume_path = prepare_resume(ckpt, STAGE3_HALF_CFG, STAGE3_HALF_RESUME)
    append_line(
        monitor_log,
        f"time={iso_now()} | action=start_stage3_halflr | base_ckpt={ckpt.name} | resume={resume_path.name}",
    )
    append_line(
        MASTER_LOG,
        f"[{iso_now()}] stage3 start: continue to 200 epochs, lr=5e-5",
    )
    launch_resume_training(resume_path, STAGE3_HALF_LOG, STAGE3_HALF_DIST_URL)
    return True


def maybe_restart_active_stage(monitor_log: Path, stage: str, current_epoch: Optional[int]) -> bool:
    if stage == "stage2_keep_lr":
        pid = find_process_pid_any(str(STAGE2_KEEP_RESUME), str(STAGE2_KEEP_RUN_DIR))
        if pid or current_epoch is None or current_epoch >= 100:
            return False
        ckpt = latest_checkpoint_path(STAGE2_KEEP_RUN_DIR)
        if not ckpt:
            return False
        append_line(
            monitor_log,
            f"time={iso_now()} | action=restart_stage2_keep_lr | resume={ckpt.name}",
        )
        launch_resume_training(ckpt, STAGE2_KEEP_LOG, STAGE2_KEEP_DIST_URL)
        return True
    if stage == "stage3_halflr":
        pid = find_process_pid_any(str(STAGE3_HALF_RESUME), str(STAGE3_HALF_RUN_DIR))
        if pid or current_epoch is None or current_epoch >= 200:
            return False
        ckpt = latest_checkpoint_path(STAGE3_HALF_RUN_DIR)
        if not ckpt:
            return False
        append_line(
            monitor_log,
            f"time={iso_now()} | action=restart_stage3_halflr | resume={ckpt.name}",
        )
        launch_resume_training(ckpt, STAGE3_HALF_LOG, STAGE3_HALF_DIST_URL)
        return True
    return False


def active_run_dir(stage: str) -> Path:
    return resolve_run_dir(stage)


def active_event_file(stage: str) -> Optional[Path]:
    return latest_event_file(active_run_dir(stage))


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor the live 0323 run and switch it to a preserved 60 -> 100+100 schedule.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--log", type=Path, default=WORKDIR / "run_logs/0323/monitor_0323_60plus60_halflr.log")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    append_line(args.log, f"[{iso_now()}] monitor started")
    ensure_legacy_stage2_blocked(args.log)

    cache: Dict[str, Tuple[str, float, EventSummary]] = {}
    last_stage = ""
    last_epoch: Optional[int] = None
    stagnant_polls = 0

    while True:
        processes = stage_processes()
        master_tail = read_last_nonempty_line(MASTER_LOG)

        if processes["legacy_stage2_pid"]:
            terminate_pid(processes["legacy_stage2_pid"], "legacy_stage2_process", args.log)
            processes = stage_processes()

        stage = infer_stage(processes)
        event_file = active_event_file(stage)
        latest_ckpt = latest_checkpoint_name(active_run_dir(stage))

        if event_file and event_file.exists():
            stat = event_file.stat()
            cached = cache.get(stage)
            if cached and cached[0] == str(event_file) and cached[1] == stat.st_mtime:
                summary = cached[2]
            else:
                summary = load_event_summary(event_file)
                cache[stage] = (str(event_file), stat.st_mtime, summary)
        else:
            summary = EventSummary(None, None, None, None, None, None, None)

        current_epoch = summary.val_loss_epoch or summary.loss_epoch or checkpoint_epoch(latest_ckpt)
        append_line(args.log, format_summary(stage, summary, latest_ckpt, processes, master_tail))

        if stage == last_stage and current_epoch is not None and current_epoch == last_epoch:
            stagnant_polls += 1
        else:
            stagnant_polls = 0
        last_stage = stage
        last_epoch = current_epoch

        if stagnant_polls >= 8:
            append_line(args.log, f"time={iso_now()} | level=warning | detail=no epoch progress for {stagnant_polls} polls")

        if stage == "stage1":
            if start_stage2_keep_if_ready(args.log, processes, current_epoch):
                time.sleep(5)
        elif stage == "stage2_keep_lr":
            if start_stage3_half_if_ready(args.log, processes, current_epoch):
                time.sleep(5)
            elif maybe_restart_active_stage(args.log, stage, current_epoch):
                time.sleep(5)
        elif stage == "stage3_halflr":
            if maybe_restart_active_stage(args.log, stage, current_epoch):
                time.sleep(5)

        if stage == "stage3_halflr" and current_epoch and current_epoch >= 200 and not processes["stage3_half_pid"]:
            append_line(MASTER_LOG, f"[{iso_now()}] stage3 finished")
            append_line(args.log, f"[{iso_now()}] monitor exiting: training complete at epoch {current_epoch}")
            return 0

        if stage == "stage1" and not processes["wrapper_pid"] and not processes["stage1_pid"] and (current_epoch or 0) < 60:
            append_line(args.log, f"[{iso_now()}] monitor exiting: stage1 stopped before epoch 60")
            return 1

        if args.once:
            append_line(args.log, f"[{iso_now()}] monitor exiting after single probe")
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
