from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://apirouter.zhiqiteai.cn/ApiRouterServ/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(response.model_dump_json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Print raw model response for debugging.")
    parser.add_argument(
        "--prompt",
        default="请用一句话回复：模型响应测试成功。",
        help="Text prompt to send to the model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to OPENROUTER_MODEL or a project default.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to OPENROUTER_API_KEY from environment or .env.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to OPENROUTER_BASE_URL or project default.",
    )
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "sk-or-v1-5794a8b038307965ef5bcdfea40fcfc18").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY. Set it in .env or pass --api-key.")

    model = args.model or os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    base_url = args.base_url or os.getenv("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个用于测试接口响应格式的助手。"},
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    print("=== message.content ===")
    print(response.choices[0].message.content or "")
    print()
    print("=== full response ===")
    print(json.dumps(response_to_dict(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
