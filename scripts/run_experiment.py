"""Run a copy of a historical prototype; never overwrite archived results."""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("version", choices=("v01", "v02"))
    p.add_argument("--run-id", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--native", action="store_true", help="also compare generated C on Linux/GCC (v02 only)")
    p.add_argument("--timeout", type=int, default=600, help="seconds per subprocess")
    a = p.parse_args()
    if not __debug__:
        p.error("do not run with Python -O")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", a.run_id):
        p.error("invalid run id")
    if a.generations < 1 or a.population < 4 or a.timeout < 1:
        p.error("generations >= 1, population >= 4, timeout >= 1 required")
    if a.native and a.version != "v02":
        p.error("--native requires v02")
    if a.version == "v02" and not sys.platform.startswith("linux"):
        p.error("the original v02 benchmark requires Linux")
    if a.native and not shutil.which("gcc"):
        p.error("gcc is required for --native")
    target = ROOT / "runs" / a.run_id
    target.mkdir(parents=True, exist_ok=False)
    work = target / "work"
    work.mkdir()
    sources = {}
    for f in sorted((ROOT / "experiments" / a.version).iterdir()):
        if f.suffix in (".py", ".asm"):
            shutil.copy2(f, work / f.name)
            sources[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONOPTIMIZE="0")
    dirty = git("status", "--porcelain")
    report = {"status": "RUNNING", "arguments": vars(a), "commit": git("rev-parse", "HEAD"),
              "dirty": None if dirty is None else bool(dirty), "python": sys.version,
              "platform": platform.platform(), "sources_sha256": sources, "steps": []}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        report["cpu"] = next((s.split(":", 1)[1].strip() for s in cpuinfo.read_text().splitlines() if s.startswith("model name")), None)
    if hasattr(os, "sched_getaffinity"):
        report["affinity"] = sorted(os.sched_getaffinity(0))
    report["threads"] = {k: env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}

    def save():
        (target / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    def step(name, *args):
        command = [sys.executable, *args]
        record = {"name": name, "command": command, "log": name + ".log"}
        report["steps"].append(record)
        save()
        print(name, flush=True)
        with (target / record["log"]).open("w") as log:
            proc = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
        record["returncode"] = proc.returncode
        save()
        proc.check_returncode()

    save()
    try:
        if a.version == "v01":
            step("demo", "vgpu.py", "--seed", str(a.seed))
            step("self-test", "vgpu.py", "--self-test")
            step("stress", "vgpu.py", "--waves", "16", "--lanes", "32", "--roots", "1024", "--seed", str(a.seed), "--issue", "random")
        else:
            step("evolve", "evolve.py", "--generations", str(a.generations), "--population", str(a.population), "--seed", str(a.seed))
            step("compare", "compare.py")
            if a.native:
                step("native", "native.py")
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        save()
        print(target)


if __name__ == "__main__":
    main()
