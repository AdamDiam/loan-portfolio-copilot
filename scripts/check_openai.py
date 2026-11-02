"""Verify the OpenAI API key and model before running the app.

Usage:  python scripts/check_openai.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.config import Settings  # noqa: E402


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    from openai import OpenAI

    model = Settings.from_env().openai_model
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with the JSON {"ok": true}'}],
            response_format={"type": "json_object"},
        )
        print(f"OK  model '{model}' answered: {resp.choices[0].message.content}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED with model '{model}': {exc}\n")
        try:
            names = sorted(m.id for m in client.models.list() if m.id.startswith(("gpt-", "o")))
            print("Models available to your key (set one as OPENAI_MODEL in .env):")
            for name in names:
                print("  ", name)
        except Exception as list_exc:  # noqa: BLE001
            print(f"Could not list models either: {list_exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
