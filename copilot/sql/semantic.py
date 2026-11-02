"""Load the semantic layer and render it as compact LLM context."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from copilot.config import SEMANTIC_LAYER_PATH


@lru_cache(maxsize=4)
def load_semantic_layer(path: Path = SEMANTIC_LAYER_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def render_context(layer: dict | None = None) -> str:
    """Render tables, glossary, rules and examples as a prompt section."""
    layer = layer or load_semantic_layer()
    lines: list[str] = [
        f"DATABASE: {layer['dialect']} | snapshot (as-of) date: {layer['snapshot_date']} | currency: {layer['currency']}",
        "",
        "## Tables",
    ]
    for table, spec in layer["tables"].items():
        lines.append(f"### {table}: {spec['description'].strip()}")
        for col, desc in spec.get("columns", {}).items():
            lines.append(f"- {col}: {desc}")
    lines += ["", "## Relationships"] + [f"- {r}" for r in layer["relationships"]]
    lines += ["", "## Business glossary"] + [f"- {k}: {v}" for k, v in layer["glossary"].items()]
    lines += ["", "## SQL rules"] + [f"- {r}" for r in layer["sql_rules"]]
    lines += ["", "## Examples"]
    for ex in layer["examples"]:
        lines.append(f"Q: {ex['question']}\nSQL:\n{ex['sql'].strip()}\n")
    return "\n".join(lines)
