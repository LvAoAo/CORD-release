"""Download ALFWorld assets into a bounded local data directory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

from .launch import ROOT, execute, runtime_env
from .storage import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "data/alfworld")
    args = p.parse_args()
    directory = args.directory.expanduser().resolve()
    if not shutil.which("alfworld-download"):
        p.error("Install requirements.txt first; alfworld-download is unavailable.")
    manifest = json.loads((ROOT / "manifests/alfworld_validation.json").read_text())
    required = [directory / name for name in manifest]
    required += [directory / "logic/alfred.pddl", directory / "logic/alfred.twl2",
                 directory / "json_2.1.1/train"]
    if all(path.exists() for path in required):
        print(f"Existing ALFWorld assets verified: {directory}")
        return
    if directory.exists() and any(path.name not in ("setup", ".runtime") for path in directory.iterdir()):
        p.error("Incomplete assets already exist; choose a new --directory. Existing files are retained.")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record_dir = directory / "setup" / suffix
    record_dir.mkdir(parents=True, exist_ok=False)
    budget = preflight(directory, record_dir, False, (ROOT / "r", ROOT / "t"))
    env = runtime_env(directory)
    env["ALFWORLD_DATA"] = str(directory)
    execute(["alfworld-download", "-f"], env, directory, record_dir, budget)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Download completed with missing assets: {missing[0]}")
    print(f'export ALFWORLD_DATA="{directory}"')


if __name__ == "__main__":
    main()
