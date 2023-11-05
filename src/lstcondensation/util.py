from pathlib import Path

import coolname
from rich.console import Console


def random_trial_name() -> str:
    c = Console(width=80)
    name = coolname.generate_slug(3)
    c.rule(f"[bold][yellow]{name}[/yellow][/bold]")
    return name


def find_latest_checkpoint(log_dir: Path):
    assert log_dir.is_dir()
    if not log_dir.name == "checkpoints":
        log_dir /= "checkpoints"
    print(log_dir)
    assert log_dir.is_dir()
    checkpoints = list(log_dir.glob("*.ckpt"))
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoints found in {log_dir}")
    path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(path)
    return path
