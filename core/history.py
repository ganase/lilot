from pathlib import Path
from datetime import datetime
import json

# プロジェクトルート/RakutenAI_LILT/logs 配下に保存する想定
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def log_turn(user_text: str, assistant_text: str, contexts=None) -> None:
    """1ターン分の会話を JSONL で保存"""
    rec = {
        "ts": datetime.now().isoformat(),
        "user": user_text,
        "assistant": assistant_text,
    }
    if contexts is not None:
        rec["contexts"] = contexts

    path = LOG_DIR / f"{datetime.now():%Y%m%d}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
