"""Record and enforce a bounded output budget for each invocation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

GIB = 1024 ** 3
RESERVE = 64 * GIB


def tree_bytes(root: Path) -> int:
    total = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in names:
            path = Path(parent) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def preflight(output: Path, record_dir: Path, training: bool, extra_roots=(), additional_gib=None) -> dict:
    candidates = [output.resolve(), *(Path(path).resolve() for path in extra_roots)]
    tracked = []
    for path in sorted(set(candidates), key=lambda item: len(item.parts)):
        if not any(path == ancestor or ancestor in path.parents for ancestor in tracked):
            tracked.append(path)
    existing = sum(tree_bytes(path) for path in tracked)
    additional = (additional_gib if additional_gib is not None else (100 if training else 24)) * GIB
    free = shutil.disk_usage(output).free
    quota = {"status": "unavailable"}
    if shutil.which("quota"):
        try:
            result = subprocess.run(["quota", "-w"], text=True, capture_output=True, timeout=10)
            quota = {"status": "checked", "returncode": result.returncode,
                     "report": (result.stdout + result.stderr).strip()[:4000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            quota = {"status": "unavailable", "reason": type(exc).__name__}
    report = {"checked_at_unix": time.time(), "free_bytes": free,
              "existing_output_bytes": existing, "tracked_roots": [str(p) for p in tracked], "additional_peak_bytes": additional,
              "maximum_retained_bytes": existing + additional, "reserve_bytes": RESERVE,
              "quota": quota,
              "retention": "One final model and extra-state checkpoint per training run; evaluation records, configuration and bounded logs. No optimizer state or automatic deletion.",
              "allowed": free >= additional + RESERVE}
    (record_dir / "storage.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["allowed"]:
        raise RuntimeError("Insufficient free space for the run budget plus the 64 GiB reserve.")
    return report
