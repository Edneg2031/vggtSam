#!/usr/bin/env python3
"""Report the environment this pipeline needs, as text you can paste into a doc.

Written for the handover: the interpreter names, checkpoints and dataset paths
are defaults inside the scripts, but whether they exist, what versions are
installed and which GPU is which are facts about the machine that only a run
here can produce.

Every probe is independent -- one missing checkpoint must not stop the rest from
being reported, because the point is to learn what is missing.

    python -m streaming_couping.scripts.report_environment
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Interpreters the pipeline uses, with the same defaults as the command files.
INTERPRETERS = (
    ("HORIZONSTREAM_PYTHON", "/home/huawei/miniconda3/envs/horizonstream/bin/python"),
    ("STREAMING_COUPING_PYTHON", None),  # falls back to the horizonstream one
)

#: What each interpreter must be able to import, and why.
PACKAGES = {
    "geometry + analysis": ("torch", "numpy", "PIL"),
    "sam3 runtime": ("iopath", "ftfy", "regex", "huggingface_hub", "timm", "einops", "pycocotools"),
    "figures": ("matplotlib",),
}

#: Paths whose existence decides whether a run can start.
PATH_ENV = {
    "HorizonStream checkpoint": ("HORIZONSTREAM_CHECKPOINT",
        "/home/bod/86Nas/95_data_bak/FoundationModels/HorizonStream.pt"),
    "ScanNet++ manifest": ("SEMANTIC_MAP_MANIFEST",
        "/data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d/manifest.json"),
    "storage root": ("VGGT_SAM_STORAGE_ROOT", "/data184/open_source/vggtSam"),
}


def _run(argv: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return 1, f"{type(error).__name__}: {error}"
    return done.returncode, (done.stdout + done.stderr).strip()


def _size(path: Path) -> str:
    try:
        if path.is_dir():
            return "dir"
        total = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if total < 1024 or unit == "TiB":
            return f"{total:.1f} {unit}"
        total /= 1024.0
    return "?"


def _path_line(label: str, env_name: str, default: str) -> tuple[str, bool]:
    value = os.environ.get(env_name) or default
    path = Path(value).expanduser()
    exists = path.exists()
    mark = "OK " if exists else "MISSING"
    return f"  [{mark}] {label:26s} {value}  ({_size(path)})", exists


def _interpreter_block(name: str, path: str | None, fallback: str | None) -> list[str]:
    resolved = os.environ.get(name) or path or fallback or ""
    lines = [f"  {name} = {resolved or '(unset)'}"]
    if not resolved:
        return lines
    probe = (
        "import sys, json\n"
        "out = {'version': sys.version.split()[0]}\n"
        "try:\n"
        "    import torch; out['torch'] = torch.__version__\n"
        "    out['cuda'] = torch.cuda.is_available()\n"
        "    out['cuda_version'] = torch.version.cuda\n"
        "    out['device_count'] = torch.cuda.device_count()\n"
        "    out['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
        "    out['vram_gb'] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)\n"
        "                      for i in range(torch.cuda.device_count())]\n"
        "except Exception as e:\n"
        "    out['torch_error'] = str(e)\n"
        "import importlib\n"
        "out['missing'] = {}\n"
        "for group, names in json.loads(sys.argv[1]).items():\n"
        "    out['missing'][group] = [n for n in names if importlib.util.find_spec(n) is None]\n"
        "print(json.dumps(out))\n"
    )
    import json

    code, output = _run([resolved, "-c", probe, json.dumps(PACKAGES)])
    if code != 0:
        lines.append(f"      cannot run: {output.splitlines()[-1] if output else 'no output'}")
        return lines
    try:
        data: dict[str, Any] = json.loads(output.splitlines()[-1])
    except (ValueError, IndexError):
        lines.append(f"      unreadable probe output: {output[:100]}")
        return lines
    lines.append(f"      python {data.get('version')}   torch {data.get('torch', data.get('torch_error', '?'))}")
    if data.get("cuda"):
        lines.append(
            f"      CUDA {data.get('cuda_version')}  {data.get('device_count')} device(s): "
            + ", ".join(
                f"{i}:{n} ({v} GiB)"
                for i, (n, v) in enumerate(zip(data.get("devices", []), data.get("vram_gb", [])))
            )
        )
    else:
        lines.append("      CUDA not available")
    for group, missing in (data.get("missing") or {}).items():
        if missing:
            lines.append(f"      MISSING ({group}): {', '.join(missing)}")
    return lines


def main() -> None:
    print("=" * 72)
    print("ENVIRONMENT REPORT")
    print("=" * 72)

    print("\n-- interpreters " + "-" * 56)
    fallback = os.environ.get("HORIZONSTREAM_PYTHON") or INTERPRETERS[0][1]
    for name, default in INTERPRETERS:
        for line in _interpreter_block(name, default, fallback):
            print(line)

    print("\n-- paths " + "-" * 63)
    for label, (env_name, default) in PATH_ENV.items():
        line, _ = _path_line(label, env_name, default)
        print(line)

    print("\n-- SAM3 (from the recovery config) " + "-" * 36)
    config = Path(__file__).resolve().parents[1] / "configs" / "recovery_dynamic_instance.yaml"
    if config.is_file():
        for line in config.read_text().splitlines():
            stripped = line.strip()
            if "path" in stripped or "checkpoint" in stripped or "device" in stripped:
                print(f"  {stripped}")
                value = stripped.split(":", 1)[-1].strip().strip('"').strip("'")
                if value.startswith("/"):
                    exists = Path(value).exists()
                    print(f"      [{'OK ' if exists else 'MISSING'}] {_size(Path(value))}")
    else:
        print(f"  MISSING config: {config}")

    print("\n-- CPU / memory " + "-" * 57)
    print(f"  cpu_count = {os.cpu_count()}")
    try:
        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        print(f"  memory    = {pages / 2**30:.1f} GiB")
    except (ValueError, OSError, AttributeError):
        print("  memory    = ?")

    print("\n-- disk " + "-" * 64)
    env_name, default = PATH_ENV["storage root"]
    for label, target in (("storage root", os.environ.get(env_name) or default),
                          ("repo", str(Path(__file__).resolve().parents[2]))):
        try:
            usage = shutil.disk_usage(target)
            print(f"  {label:14s} {target}  free {usage.free / 2**30:.1f} GiB of {usage.total / 2**30:.1f} GiB")
        except OSError as error:
            print(f"  {label:14s} {target}  unavailable: {error}")

    print("\n-- grid engine / scheduler " + "-" * 45)
    for variable in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        print(f"  {variable} = {os.environ.get(variable, '(unset)')}")

    print("\n-- host " + "-" * 64)
    code, output = _run(["uname", "-a"])
    print(f"  {output.splitlines()[0] if code == 0 else '(unavailable)'}")


if __name__ == "__main__":
    main()
