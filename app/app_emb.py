import os
import json
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import secrets
import subprocess
import platform
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Lilot (Embedding-only RAG)
# - Embedding検索のみ（Keyword検索なし）
# - Embedding: リモートAPI(OpenAI互換) or ローカルMiniLM
# - 添付ファイル（txt/csv/pdf）を UI からアップロード → data/uploads に保存
# - PDFはページ→チャンク分割（巨大ブロック回避）
# - 参照ナレッジに Source 表示
# - 取得結果に uploads 由来を最低1件含める（uploadsが存在する場合）
# =========================================================

# ---------------------------------------------------------
# .env 読み込み & 環境変数
# ---------------------------------------------------------
load_dotenv()

LLM_API_KEY = os.getenv("LOCALLM_API_KEY") or os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LOCALLM_BASE_URL") or os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LOCALLM_CHAT_MODEL") or os.getenv("LLM_MODEL", "")

EMB_API_KEY = os.getenv("EMB_API_KEY")
EMB_BASE_URL = os.getenv("EMB_BASE_URL", "https://api.openai.com/v1")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

LOCAL_EMB_MODEL_PATH = os.getenv(
    "LOCAL_EMB_MODEL_PATH",
    "sentence-transformers/all-MiniLM-L6-v2",
)

def use_remote_embedding() -> bool:
    return bool(EMB_API_KEY)

# ---------------------------------------------------------
# パス設定
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 安定版: PDFはアップロード時に抽出して保存し、起動時は抽出済みテキストのみを参照する
UPLOAD_ORIGINAL_DIR = UPLOAD_DIR / "original"
UPLOAD_EXTRACTED_DIR = UPLOAD_DIR / "extracted"
UPLOAD_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_META_PATH = UPLOAD_DIR / "index_meta.json"

# ★★ ロゴとファビコンの絶対パス
LOGO_PATH = (BASE_DIR / "lilot.png").resolve()
FAVICON_PATH = (BASE_DIR / "lilot_mark.png").resolve()
ENV_PATH = (BASE_DIR / ".env").resolve()

# ---------------------------------------------------------
# 添付アップロード
# ---------------------------------------------------------
ALLOWED_UPLOAD_EXTS = {".txt", ".csv", ".pdf"}

def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    # Windows NG文字
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if name else "uploaded_file"

def save_uploaded_files(files) -> dict:
    """
    アップロードされたファイルを保存し、PDFは保存直後にテキスト抽出して extracted に保存します（複数同時対応）。
    返り値:
      {
        "saved": [Path...],         # 保存したファイル（txt/csv/pdf）
        "pdf_extracted": [str...],  # 抽出したPDFファイル名
        "pdf_skipped": [str...],    # 変更なしでスキップしたPDFファイル名
        "pdf_failed": [str...],     # 抽出失敗したPDFファイル名
      }
    """
    result = {"saved": [], "pdf_extracted": [], "pdf_skipped": [], "pdf_failed": []}
    if not files:
        return result

    meta = load_upload_meta()

    for f in files:
        try:
            name = os.path.basename(getattr(f, "name", ""))
            if not name:
                continue
            ext = os.path.splitext(name)[1].lower()

            data = f.getvalue() if hasattr(f, "getvalue") else f.read()
            if not isinstance(data, (bytes, bytearray)):
                continue

            # txt/csv は uploads/ 直下に保存（軽い）
            if ext in [".txt", ".csv"]:
                out = UPLOAD_DIR / name
                out.write_bytes(data)
                result["saved"].append(out)
                continue

            # pdf は original/ に保存し、必要なら抽出
            if ext == ".pdf":
                out_pdf = UPLOAD_ORIGINAL_DIR / name
                out_pdf.write_bytes(data)
                result["saved"].append(out_pdf)

                sha = _sha256_bytes(data)
                prev = meta.get(name, {})
                if prev.get("sha256") == sha and extracted_json_path_for(name).exists():
                    result["pdf_skipped"].append(name)
                    continue

                                # 抽出して保存（ここでチャンク化まで完了させ、起動時に重処理をしない）
                pages = _extract_pdf_pages(out_pdf, max_pages=80, max_chars_per_page=8000, max_total_chars=200000)

                chunks: List[Dict[str, str]] = []
                total_chars = 0
                for page_idx, page_text in enumerate(pages, 1):
                    if not (page_text or "").strip():
                        continue
                    # 1ページあたりのチャンク数も制限（暴発防止）
                    parts = chunk_text(page_text, max_chars=900, overlap=180, hard_char_limit=12000, max_chunks=40)
                    for cidx, part in enumerate(parts, 1):
                        chunks.append({"text": part, "source": f"uploads/{name}#p{page_idx}-c{cidx}"})
                        total_chars += len(part)
                    # PDF 1ファイルあたりのチャンク数上限
                    if len(chunks) >= 1200:
                        break
                chunks = chunks[:1200]

                if total_chars <= 0 or not chunks:
                    meta[name] = {
                        "sha256": sha,
                        "status": "no_text",
                        "pages": len(pages),
                        "chars": 0,
                        "chunks": 0,
                    }
                    result["pdf_failed"].append(name)
                    continue

                extracted_path = extracted_json_path_for(name)
                payload = {"chunks": chunks, "pages_count": len(pages), "chars": total_chars}
                extracted_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

                meta[name] = {
                    "sha256": sha,
                    "status": "ready",
                    "pages": len(pages),
                    "chars": total_chars,
                    "chunks": len(chunks),
                }
                result["pdf_extracted"].append(name)
                continue

        except Exception:
            continue

    save_upload_meta(meta)
    return result

def invalidate_knowledge_cache():
    """
    uploads に変更が入ったときに、ナレッジ読込キャッシュとインデックスを確実に破棄します。
    Streamlit のバージョン差で .clear() が効かないケースがあるため、両方試します。
    """
    # 個別クリア（新しめのStreamlit）
    for fn in (load_knowledge, build_corpus_index, _pdf_extract_summary):
        try:
            fn.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    # 全体クリア（互換性重視）
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


# ---------------------------------------------------------
# ログ関連
# ---------------------------------------------------------
def _get_log_path() -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    session_id = getattr(st.session_state, "session_id", None) or "default"
    return LOGS_DIR / f"{date_str}_{session_id}.jsonl"

def log_interaction(question: str, answer: str, contexts: List[Dict[str, Any]], extra=None):
    extra = extra or {}
    rec = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,  # [{"source":..., "score":..., "text":...}, ...]
    }
    rec.update(extra)
    try:
        with _get_log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def list_log_files() -> List[Path]:
    return sorted(LOGS_DIR.glob("*.jsonl"), reverse=True)

def load_history_from_log(log_path: Path):
    history = []
    if not log_path.exists():
        return history
    for line in log_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        q = rec.get("question")
        a = rec.get("answer")
        if q and a:
            history.append({"user": q, "assistant": a})
    return history

# ---------------------------------------------------------
# LLM / Embedding クライアント
# ---------------------------------------------------------
def get_llm_client():
    if not LLM_API_KEY:
        return "LLM_API_KEY がありません"
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def get_emb_client():
    if not EMB_API_KEY:
        return "EMB_API_KEY がありません"
    return OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)

# ---------------------------------------------------------
# ローカル埋め込みモデル
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_local_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers が必要です。\n"
            "pip install sentence-transformers を実行してください"
        ) from e
    return SentenceTransformer(LOCAL_EMB_MODEL_PATH)

# ---------------------------------------------------------
# system_prompt 読み込み
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_system_prompt():
    path = DATA_DIR / "system_prompt.txt"
    if path.exists():
        t = path.read_text(encoding="utf-8", errors="ignore").strip()
        if t:
            return t
    return (
        "あなたはローカルナレッジを活用する社内ヘルプデスクAIです。\n"
        "常に日本語で丁寧に回答してください。"
    )

# ---------------------------------------------------------
# テキスト正規化・チャンク分割
# ---------------------------------------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 900, overlap: int = 150,
               hard_char_limit: int = 20000, max_chunks: int = 200) -> List[str]:
    """
    文字数ベースの簡易チャンク分割（巨大ブロック対策・安全ガード付き）
    - hard_char_limit を超える入力は先頭のみで打ち切り（MemoryError防止）
    - max_chunks を超えたら打ち切り（暴発防止）
    """
    text = normalize_text(text)
    if not text:
        return []

    if hard_char_limit and len(text) > hard_char_limit:
        text = text[:hard_char_limit]

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text]

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        s = "\n\n".join(buf).strip()
        if s:
            chunks.append(s)
        buf = []
        buf_len = 0

    for p in paras:
        if max_chunks and len(chunks) >= max_chunks:
            break

        extra = 2 if buf else 0
        if buf_len + len(p) + extra <= max_chars:
            buf.append(p)
            buf_len += len(p) + extra
        else:
            flush()
            if max_chunks and len(chunks) >= max_chunks:
                break

            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    if max_chunks and len(chunks) >= max_chunks:
                        break
                    end = min(start + max_chars, len(p))
                    part = p[start:end].strip()
                    if part:
                        chunks.append(part)
                    start = max(0, end - overlap)
                    if start == end:
                        break
            else:
                buf = [p]
                buf_len = len(p)

    flush()
    out = [c for c in chunks if c.strip()]
    if max_chunks:
        out = out[:max_chunks]
    return out

# ---------------------------------------------------------
# PDF 抽出（pypdf）
# ---------------------------------------------------------
def _pdf_can_extract_text() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except Exception:
        return False

def _extract_pdf_pages(path: Path, max_pages: int = 50,
                       max_chars_per_page: int = 8000,
                       max_total_chars: int = 200000) -> List[str]:
    """
    PDFをページ単位でテキスト抽出します（キャッシュは使いません）。
    ※ 抽出は「アップロード直後」にのみ行い、結果は uploads/extracted/*.json に保存します。
    安全ガード:
      - 1ページあたり max_chars_per_page で打ち切り
      - PDF全体で max_total_chars で打ち切り（巨大PDFでのメモリ暴発防止）
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return []

    try:
        reader = PdfReader(str(path))
        pages: List[str] = []
        total = 0
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            txt = page.extract_text() or ""
            txt = normalize_text(txt)
            if max_chars_per_page and len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page]
            pages.append(txt)
            total += len(txt)
            if max_total_chars and total >= max_total_chars:
                break
        return pages
    except Exception:
        return []

    try:
        reader = PdfReader(str(path))
        pages: List[str] = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            txt = page.extract_text() or ""
            pages.append(normalize_text(txt))
        return pages
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _pdf_extract_summary(path_str: str, mtime: float):
    p = Path(path_str)
    pages = _extract_pdf_pages(p, max_pages=50)
    txt = "\n\n".join([t for t in pages if t])
    return {
        "chars": len(txt),
        "pages": len(pages),
        "sample": (txt[:160] + "…") if len(txt) > 160 else txt,
    }

# ---------------------------------------------------------
# Knowledge 読み込み（txt/csv/pdf） with source
# ---------------------------------------------------------
def _append_chunked_docs(docs: List[Dict[str, str]], text: str, source_prefix: str,
                         max_chars: int = 900, overlap: int = 150):
    parts = chunk_text(text, max_chars=max_chars, overlap=overlap)
    for i, part in enumerate(parts, 1):
        docs.append({"text": part, "source": f"{source_prefix}#c{i}"})

@st.cache_data(show_spinner=False)
def load_knowledge() -> List[Dict[str, str]]:
    """
    起動時に軽く動かすため、PDF本体は読まず、抽出済みチャンク（uploads/extracted/*.json）だけを読みます。
    対象:
      - data/knowledge.txt
      - data/uploads/*.txt, *.csv
      - data/uploads/extracted/*.json  (PDF抽出済み: chunks)
    """
    docs: List[Dict[str, str]] = []

    # main knowledge
    kp = DATA_DIR / "knowledge.txt"
    if kp.exists():
        t = kp.read_text(encoding="utf-8", errors="ignore")
        _append_chunked_docs(docs, t, "knowledge.txt")

    # uploads: txt
    for p in UPLOAD_DIR.glob("*.txt"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            _append_chunked_docs(docs, t, f"uploads/{p.name}")
        except Exception:
            continue

    # uploads: csv（1行＝1ドキュメント）
    import csv
    for p in UPLOAD_DIR.glob("*.csv"):
        try:
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for ridx, row in enumerate(reader, 1):
                    srow = ", ".join(col.strip() for col in row if col and col.strip())
                    if not srow:
                        continue
                    parts = chunk_text(srow, max_chars=900, overlap=120, hard_char_limit=6000, max_chunks=10)
                    for cidx, part in enumerate(parts, 1):
                        docs.append({"text": part, "source": f"uploads/{p.name}#row{ridx}-c{cidx}"})
        except Exception:
            continue

    # extracted: pdf chunks json（起動時はチャンク化済みを読むだけ）
    for jp in UPLOAD_EXTRACTED_DIR.glob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        chunks = None
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            chunks = data.get("chunks")
        # 後方互換: pages形式が残っていた場合は、起動時に暴発しないよう極小で取り込み
        elif isinstance(data, dict) and isinstance(data.get("pages"), list):
            pages = data.get("pages") or []
            pdf_name = jp.name[:-5]
            for page_idx, page_text in enumerate(pages[:3], 1):  # 先頭数ページだけ
                if not (page_text or "").strip():
                    continue
                parts = chunk_text(page_text, max_chars=900, overlap=180, hard_char_limit=6000, max_chunks=10)
                for cidx, part in enumerate(parts, 1):
                    docs.append({"text": part, "source": f"uploads/{pdf_name}#p{page_idx}-c{cidx}"})
            continue
        else:
            continue

        for c in chunks:
            try:
                text = (c.get("text") if isinstance(c, dict) else "") or ""
                source = (c.get("source") if isinstance(c, dict) else "") or ""
                if text.strip():
                    docs.append({"text": text, "source": source or f"uploads/{jp.name}"})
            except Exception:
                continue

    return docs


def get_knowledge_docs() -> List[Dict[str, str]]:
    return load_knowledge()

# ---------------------------------------------------------
# セッション管理
# ---------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "loaded_log_name" not in st.session_state:
        st.session_state.loaded_log_name = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(6)


def add_history(user, assistant):
    st.session_state.history.append({"user": user, "assistant": assistant})
    st.session_state.messages.append({"role": "user", "content": user})
    st.session_state.messages.append({"role": "assistant", "content": assistant})

def get_history():
    return st.session_state.history


# ---------------------------------------------------------
# クエリの派生（Embeddingのみで拾いやすくする）
# ---------------------------------------------------------
def extract_title_hint(query: str) -> str:
    """
    例:
      本論文=TITLE の著者 → TITLE を抽出
      「TITLE」の著者 → TITLE を抽出
    """
    q = (query or "").strip()

    m = re.search(r"[=＝]\s*([A-Za-z0-9][^\n]+?)\s*(の著者|の筆者|著者|author|authors)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"[「『\"](.+?)[」』\"]\s*(の著者|の筆者|著者|author|authors)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"(.+?)\s*(の著者|の筆者|著者)\b", q)
    if m:
        cand = m.group(1).strip()
        return cand[-160:].strip() if len(cand) > 160 else cand

    return ""

def build_query_variants(query: str) -> List[str]:
    q = (query or "").strip()
    title = extract_title_hint(q)
    vars = [q]
    if title and title != q:
        vars.extend([
            title,
            f"authors of {title}",
            f"{title} authors",
            f"{title} author list",
            f"{title} 著者",
        ])
    out = []
    seen = set()
    for x in vars:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# ---------------------------------------------------------
# embedding
# ---------------------------------------------------------
def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    return dot / (na * nb) if na and nb else 0.0

def embed_texts(texts: List[str]):
    if use_remote_embedding():
        c = get_emb_client()
        if isinstance(c, str):
            raise RuntimeError(c)
        resp = c.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    else:
        enc = load_local_embedder()
        arr = enc.encode(texts, show_progress_bar=False)
        return [list(map(float, v)) for v in arr]

@st.cache_resource(show_spinner=True)
def build_corpus_index():
    docs = get_knowledge_docs()  # [{"text":..., "source":...}, ...]
    if not docs:
        return [], []
    texts = [d["text"] for d in docs]
    vecs = embed_texts(texts)
    return docs, vecs

def _is_upload_source(src: str) -> bool:
    return (src or "").startswith("uploads/")

def retrieve_with_embedding(query: str, top_k: int = 3, min_uploads: int = 1, use_variants: bool = False) -> List[Dict[str, Any]]:
    """
    取得結果に uploads 由来を最低 min_uploads 件含める（uploadsが存在する場合）
    - 検索方式（Embedding）は維持
    - use_variants=True の場合、派生クエリを複数作って「最大類似度」でスコアリング（タイトル→著者行を拾いやすく）
    """
    docs, vecs = build_corpus_index()
    if not docs or not vecs:
        return []

    queries = build_query_variants(query) if use_variants else [query]
    q_vecs = embed_texts(queries)

    scored: List[Dict[str, Any]] = []
    for d, v in zip(docs, vecs):
        s = max(cosine_similarity(qv, v) for qv in q_vecs) if q_vecs else 0.0
        scored.append({
            "score": float(s),
            "text": d.get("text", ""),
            "source": d.get("source", "unknown"),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    if top_k <= 0:
        return []

    has_uploads = any(_is_upload_source(x["source"]) for x in scored)
    need_uploads = min_uploads if has_uploads else 0
    need_uploads = max(0, min(need_uploads, top_k))

    selected: List[Dict[str, Any]] = []
    used_sources = set()

    if need_uploads > 0:
        for x in scored:
            if _is_upload_source(x["source"]) and x["source"] not in used_sources:
                selected.append(x)
                used_sources.add(x["source"])
                if len(selected) >= need_uploads:
                    break

    for x in scored:
        if len(selected) >= top_k:
            break
        if x["source"] in used_sources:
            continue
        selected.append(x)
        used_sources.add(x["source"])

    if len(selected) < top_k:
        for x in scored:
            if len(selected) >= top_k:
                break
            selected.append(x)

    return selected[:top_k]

# ---------------------------------------------------------
# LLM 呼び出し
# ---------------------------------------------------------
def call_llm_with_context(query: str, contexts: List[Dict[str, Any]]) -> str:
    client = get_llm_client()
    if isinstance(client, str):
        return client

    hist = get_history()

    if contexts:
        ctx_text = "\n\n---\n\n".join(
            [f"[{c.get('source','unknown')}]\n{c.get('text','')}" for c in contexts]
        )
    else:
        ctx_text = "ローカルナレッジは見つかりませんでした。"

    sys_base = load_system_prompt()
    sys_content = f"{sys_base}\n\n【重要】以下の「抽出されたローカルナレッジ」以外を根拠に推測・補完してはいけません。答えがナレッジ内に無い場合は、必ず「ナレッジ内に見つかりませんでした。」とだけ回答してください。外部サイト検索の提案や一般論の説明も不要です。\n\n-----\n以下は抽出されたローカルナレッジです：\n{ctx_text}"

    msgs = [{"role": "system", "content": sys_content}]
    for h in hist[-5:]:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""

# ---------------------------------------------------------
# ファイルを開く（メモ帳 / フォルダ）
# ---------------------------------------------------------
def open_with_notepad(path: Path):
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["notepad.exe", str(path)])
        else:
            # mac/linux は既定アプリ
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass

def open_in_file_manager(path: Path):
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(path)])
        else:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def load_upload_meta() -> dict:
    if UPLOAD_META_PATH.exists():
        try:
            return json.loads(UPLOAD_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_upload_meta(meta: dict) -> None:
    try:
        UPLOAD_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def extracted_json_path_for(pdf_name: str) -> Path:
    return UPLOAD_EXTRACTED_DIR / f"{pdf_name}.json"


# ---------------------------------------------------------
# ログ → チャット投入（最新1件）
# ---------------------------------------------------------
def load_latest_log_into_chat():
    logs = list_log_files()
    if not logs:
        return False
    latest = logs[0]
    history = load_history_from_log(latest)
    if not history:
        return False
    st.session_state.history = history[:]
    st.session_state.messages = []
    for h in history:
        st.session_state.messages.append({"role": "user", "content": h["user"]})
        st.session_state.messages.append({"role": "assistant", "content": h["assistant"]})
    st.session_state.loaded_log_name = latest.name
    return True


def load_log_into_chat(log_path: Path):
    history = load_history_from_log(log_path)
    if not history:
        return False
    st.session_state.history = history[:]
    st.session_state.messages = []
    for h in history:
        st.session_state.messages.append({"role": "user", "content": h["user"]})
        st.session_state.messages.append({"role": "assistant", "content": h["assistant"]})
    st.session_state.loaded_log_name = log_path.name
    return True


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Lilot",
        page_icon=str(FAVICON_PATH),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("### 🔎 Lilot AIへの依頼や質問を入力して送信して下さい。")
    st.caption("お待たせし過ぎたAIチャットアプリケーション。簡単セットアップ、ローカル稼働ができます。Knowledge.txtでRAGが出来ます。")

    init_session_state()
    docs_all = get_knowledge_docs()
    doc_count = len(docs_all)
    upload_count = sum(1 for d in docs_all if str(d.get("source","")).startswith("uploads/"))
    st.caption(f"Index: {doc_count} chunks (uploads: {upload_count})")

    # -----------------------------------------------------
    # サイドバー
    # -----------------------------------------------------
    with st.sidebar:

        # ★★ ロゴをセンタリングして「半分の大きさ（50px）」で表示
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=50)

        st.markdown("### 🔎 Lilot")
        st.caption("Light-weight local AI chat application")

        if st.button("🆕 新規チャット", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.loaded_log_name = None
            st.success("新しいチャットを開始しました。")
            st.rerun()

        with st.expander("📄 ナレッジファイル", expanded=False):
            st.caption("knowledge.txt / system_prompt.txt を開きます（編集は自己責任でお願いします）。")
            cka, ckb = st.columns(2)
            with cka:
                if st.button("knowledge.txt を開く", use_container_width=True):
                    open_with_notepad(DATA_DIR / "knowledge.txt")
            with ckb:
                if st.button("system_prompt.txt を開く", use_container_width=True):
                    open_with_notepad(DATA_DIR / "system_prompt.txt")

        
        with st.expander("🧾 ログ", expanded=False):
            st.caption("logs/ に保存された会話ログ（jsonl）を確認できます。")

            if st.button("logs を開く", use_container_width=True, key="open_logs_btn"):
                open_in_file_manager(LOGS_DIR)

            logs = list_log_files()
            if logs:
                st.caption("最近のログ（→ でロード）")
                for lp in logs[:10]:
                    c1, c2 = st.columns([6, 1])
                    with c1:
                        st.write(lp.name)
                    with c2:
                        if st.button("→", key=f"log_arrow_{lp.name}"):
                            ok = load_log_into_chat(lp)
                            if ok:
                                st.success(f"{lp.name} を読み込みました。")
                                st.rerun()
                            else:
                                st.warning("ログを読み込めませんでした。")
            else:
                st.caption("ログはまだありません。")


        st.markdown("---")

        # ★★ 添付ファイル（追加）
        with st.expander("📎 添付ファイル", expanded=False):
            st.caption("txt/csv は保存後すぐ検索対象になります。PDF はアップロード直後にテキスト抽出し、以降は差分が無い限り再抽出しません。")

            files = st.file_uploader("txt / csv / pdf（複数可）", type=["txt", "csv", "pdf"], accept_multiple_files=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("保存して取り込む", use_container_width=True) and files:
                    with st.spinner("保存・取り込み中...（PDFは初回のみ時間がかかります）"):
                        res = save_uploaded_files(files)

                    if res.get("saved"):
                        invalidate_knowledge_cache()
                        msg = []
                        msg.append("保存: " + ", ".join([Path(p).name for p in res["saved"]]))
                        if res.get("pdf_extracted"):
                            msg.append("PDF抽出: " + ", ".join(res["pdf_extracted"]))
                        if res.get("pdf_skipped"):
                            msg.append("PDFスキップ(変更なし): " + ", ".join(res["pdf_skipped"]))
                        if res.get("pdf_failed"):
                            msg.append("PDF抽出失敗: " + ", ".join(res["pdf_failed"]))
                        st.success("\n".join(msg))
                        st.rerun()
                    else:
                        st.warning("保存できるファイルがありませんでした。")

            with c2:
                if st.button("uploads を開く", use_container_width=True):
                    open_in_file_manager(UPLOAD_DIR)

            # ステータス表示（重い処理はしない）
            meta = load_upload_meta()

            pdfs = sorted([p for p in UPLOAD_ORIGINAL_DIR.glob("*.pdf") if p.is_file()])
            if pdfs:
                st.caption("PDF（original/）")
                for p in pdfs[:20]:
                    info = meta.get(p.name, {})
                    status = info.get("status", "unknown")
                    pages = info.get("pages", 0)
                    chars = info.get("chars", 0)
                    chunks = info.get("chunks", 0)
                    ex = extracted_json_path_for(p.name)
                    ex_ok = "OK" if ex.exists() and status == "ready" else "NG"
                    st.write(f"- {p.name}  / extracted: {ex_ok}  / {pages} pages  / {chunks} chunks  / {chars} chars")
                if len(pdfs) > 20:
                    st.caption(f"…ほか {len(pdfs)-20} 件")
            else:
                st.caption("PDFはまだありません。")

            ex_cnt = len(list(UPLOAD_EXTRACTED_DIR.glob('*.json')))
            st.caption(f"extracted/ : {ex_cnt} 件")


        with st.expander("⚙️ .env 編集", expanded=False):
            st.caption(".env Path:")
            st.code(str(ENV_PATH))
            if ENV_PATH.exists():
                try:
                    t = ENV_PATH.read_text(encoding="utf-8", errors="ignore").strip()
                    if t:
                        st.caption("冒頭100文字")
                        st.write(t[:100])
                    else:
                        st.caption(".env は空です。")
                except Exception as e:
                    st.caption(f"読み込み失敗: {e}")
            else:
                st.caption(".env がありません。")
            if st.button(".env を編集", use_container_width=True):
                open_with_notepad(ENV_PATH)

        with st.expander("🔧 環境情報", expanded=False):
            st.write(f"[LLM] Base URL : `{LLM_BASE_URL}`")
            st.write(f"[LLM] Model    : `{LLM_MODEL}`")
            if use_remote_embedding():
                st.write("[EMB] Mode     : remote")
                st.write(f"[EMB] Base URL : `{EMB_BASE_URL}`")
                st.write(f"[EMB] Model    : `{EMB_MODEL}`")
            else:
                st.write("[EMB] Mode     : local MiniLM")
                st.write(f"[EMB] Path/ID  : `{LOCAL_EMB_MODEL_PATH}`")

    # -----------------------------------------------------
    # メッセージ表示
    # -----------------------------------------------------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # -----------------------------------------------------
    # チャット入力
    # -----------------------------------------------------
    query = st.chat_input("AIの回答には誤りが含まれることがあります。そのままではなく事実確認を行ってから利用してください。")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("ナレッジ検索中..."):
            try:
                # uploads が存在する場合は最低1件混ぜる
                # 著者・タイトル系の質問はPDF/アップロード由来を厚めに拾う（派生クエリも使う）
                qlower = (query or '').lower()
                is_author_q = ('著者' in query) or ('筆者' in query) or ('author' in qlower) or ('authors' in qlower) or ('論文' in query)
                if is_author_q:
                    ctx = retrieve_with_embedding(query, top_k=10, min_uploads=5, use_variants=True)
                else:
                    ctx = retrieve_with_embedding(query, top_k=3, min_uploads=1, use_variants=False)
            except Exception as e:
                ctx = []
                st.error(f"Embedding エラー: {e}")

        if not ctx:
            answer = "ナレッジ内に見つかりませんでした。"
        else:
            with st.spinner("LLM に問い合わせ中..."):
                answer = call_llm_with_context(query, ctx)

        with st.chat_message("assistant"):
            st.write(answer)
            if ctx:
                qlower2 = (query or "").lower()
                is_author_q2 = ("著者" in query) or ("筆者" in query) or ("author" in qlower2) or ("authors" in qlower2) or ("論文" in query)
                if is_author_q2 and not any(str(c.get("source","")).startswith("uploads/") for c in ctx):
                    st.warning("uploads/PDF が検索対象に含まれていないか、インデックスが未構築の可能性があります。左ペインでPDFをアップロードして『保存して取り込む』を押してください。")
                with st.expander("🔍 参照したローカルナレッジ"):
                    for i, c in enumerate(ctx, 1):
                        st.markdown(f"**Doc {i}**  （score={c.get('score', 0):.3f}）")
                        st.caption(f"Source: {c.get('source','unknown')}")
                        st.write(c.get("text", ""))
            else:
                st.caption("該当ナレッジなし。")
                st.caption(f"Index chunks: {doc_count} (uploads: {upload_count})")
                st.caption("※ uploads/ にPDFがあるのに uploads:0 の場合は、左ペインの『インデックス再構築』を押してください。")

        add_history(query, answer)
        log_interaction(query, answer, ctx)

if __name__ == "__main__":
    main()
