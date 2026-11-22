# Purpose: small artifact I/O helpers: read JSON/YAML configs and write JSON,
#   creating parent directories as needed.
# Pipeline role: used by benchmark/runner.py (config, run_meta.json,
#   loss_history.json) and scripts/run_cmp3d_gate2.py (config).
# No formatting or rounding happens here.
"""Small artifact I/O utilities for JSON and YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return p


# Config loader of benchmark/runner.py and scripts/run_cmp3d_gate2.py.
def read_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON or YAML config file, auto-detecting format by extension."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".json",):
        return json.loads(text)
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception:
            raise SystemExit("PyYAML not installed; use JSON config or install pyyaml.")
        return yaml.safe_load(text)
    # fallback: try json
    try:
        return json.loads(text)
    except Exception:
        raise SystemExit("Unsupported config file; use .json/.yml/.yaml")
