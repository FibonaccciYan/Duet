"""Save the actual uncommitted source and hardware identity for future runs."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


def save_manifest(output, options):
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "scripts/original/performance").glob("*.py"))
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    packages = {}
    for name in ("torch", "triton", "transformers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    manifest = dict(
        commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        working_tree=git("status", "--short"), source_sha256=source,
        python=sys.executable, packages=packages, options=vars(options),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0), argv=sys.argv,
    )
    path = Path(str(output) + ".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    return str(path)
