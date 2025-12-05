import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import secrets
import subprocess
import platform

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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
UPLOAD_DIR.mkdir(exist_ok=True)

# ★★ ロゴとファビコンの絶対パスを設定
LOGO_PATH = (BASE_DIR / "lilot.png").resolve()
FAVICON_PATH = (BASE_DIR / "lilot_mark.png").resolve()


def _get_log_path() -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    session_id = getattr(st.session_state, "session_id", None) or "default"
    return LOGS_DIR / f"{date_str}_{session_id}.jsonl"

# ---------------------------------------------------------
# ログ関連
# ---------------------------------------------------------
def log_interaction(question: str, answer: str, contexts: List[str], extra=None):
    extra = extra or {}
    rec = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    rec.update(extra)

    try:
        with _get_log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except:
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
        except:
            continue
        q = rec.get("question")
        a = rec.get("answer")
        if q and a:
            history.append({"user": q, "assistant": a})
    return history

# ---------------------------------------------------------
# LLMクライアント
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
        t = path.read_text(encoding="utf-8").strip()
        if t:
            return t
    return (
        "あなたはローカルナレッジを活用する社内ヘルプデスクAIです。\n"
        "常に日本語で丁寧に回答してください。"
    )

# ---------------------------------------------------------
# Knowledge 読み込み
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_knowledge():
    docs = []
    kp = DATA_DIR / "knowledge.txt"
    if kp.exists():
        t = kp.read_text(encoding="utf-8", errors="ignore")
        docs.extend(b.strip() for b in t.split("\n\n") if b.strip())

    for p in UPLOAD_DIR.glob("*.txt"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            docs.extend(b.strip() for b in t.split("\n\n") if b.strip())
        except:
            continue

    import csv
    for p in UPLOAD_DIR.glob("*.csv"):
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    s = ", ".join(col.strip() for col in row if col.strip())
                    if s:
                        docs.append(s)
        except:
            continue

    return docs

def get_knowledge_docs():
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


def embed_texts(texts):
    if use_remote_embedding():
        c = get_emb_client()
        resp = c.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    else:
        enc = load_local_embedder()
        arr = enc.encode(texts, show_progress_bar=False)
        return [list(map(float, v)) for v in arr]


@st.cache_resource(show_spinner=True)
def build_corpus_index():
    docs = get_knowledge_docs()
    if not docs:
        return [], []
    vecs = embed_texts(docs)
    return docs, vecs


def retrieve_with_embedding(query, top_k=3):
    docs, vecs = build_corpus_index()
    if not docs or not vecs:
        return []
    q = embed_texts([query])[0]
    scored = []
    for d, v in zip(docs, vecs):
        s = cosine_similarity(q, v)
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

# ---------------------------------------------------------
# LLM 呼び出し
# ---------------------------------------------------------
def call_llm_with_context(query, contexts):
    client = get_llm_client()
    if isinstance(client, str):
        return client

    hist = get_history()

    ctx_text = "\n\n---\n\n".join(contexts) if contexts else "ローカルナレッジは見つかりませんでした。"

    sys_base = load_system_prompt()
    sys_content = f"{sys_base}\n\n-----\n以下は抽出されたローカルナレッジです：\n{ctx_text}"

    msgs = [{"role": "system", "content": sys_content}]
    for h in hist[-5:]:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""

# ---------------------------------------------------------
# メモ帳で開く
# ---------------------------------------------------------
def open_with_notepad(path: Path):
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["notepad.exe", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except:
        pass

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Lilot",
        page_icon=str(FAVICON_PATH),   # ★★ ファビコン絶対パス
        layout="wide",
    )

    st.markdown("### 🔎 Lilot AIへの依頼や質問を入力して送信して下さい。")
    st.caption("お待たせし過ぎたAIチャットアプリケーション。簡単セットアップ、ローカル稼働ができます。Knowledge.txtでRAGが出来ます。")

    init_session_state()
    docs = get_knowledge_docs()
    doc_count = len(docs)

    # -----------------------------------------------------
    # サイドバー
    # -----------------------------------------------------
    with st.sidebar:

        # ★★ ロゴをセンタリングして 100px で表示
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(str(LOGO_PATH), width=100)

        st.markdown("### 🔎 Lilot")
        st.caption("Light-weight local AI chat application")

        if st.button("🆕 新規チャット", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.loaded_log_name = None
            st.success("新しいチャットを開始しました。")
            st.rerun()

        st.markdown("---")

        with st.expander("📝 ログ / 履歴", expanded=False):
            logs = list_log_files()
            if not logs:
                st.caption("ログがありません。")
            else:
                st.caption("直近20件")
                for p in logs[:20]:
                    label = p.stem
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(label)
                    with c2:
                        if st.button("→", key=f"load_{label}"):
                            hist = load_history_from_log(p)
                            st.session_state.history = hist
                            st.session_state.messages = []
                            for h in hist:
                                st.session_state.messages.append({"role": "user", "content": h["user"]})
                                st.session_state.messages.append({"role": "assistant", "content": h["assistant"]})
                            st.session_state.loaded_log_name = label
                            st.success(f"{label} を読み込みました。")
                            st.rerun()

            if st.session_state.loaded_log_name:
                st.info(f"読み込み中ログ: {st.session_state.loaded_log_name}")

        with st.expander("📚 ローカルナレッジ", expanded=False):
            st.write(f"文書数: **{doc_count} 件**")

            kp = DATA_DIR / "knowledge.txt"
            st.caption("knowledge.txt Path:")
            st.code(str(kp))

            if kp.exists() and doc_count:
                st.caption("knowledge.txt の冒頭100文字")
                st.write(docs[0][:100])

            if st.button("knowledge.txt をメモ帳で開く", use_container_width=True):
                open_with_notepad(kp)

        with st.expander("🧠 system_prompt 設定", expanded=False):
            sp = DATA_DIR / "system_prompt.txt"
            st.caption("system_prompt.txt Path:")
            st.code(str(sp))

            if sp.exists():
                try:
                    t = sp.read_text(encoding="utf-8").strip()
                    if t:
                        st.caption("冒頭100文字")
                        st.write(t[:100])
                    else:
                        st.caption("system_prompt.txt は空です。")
                except Exception as e:
                    st.caption(f"読み込み失敗: {e}")
            else:
                st.caption("system_prompt.txt がありません。")

            if st.button("system_prompt.txt をメモ帳で開く", use_container_width=True):
                open_with_notepad(sp)

        with st.expander("🔧 環境情報", expanded=False):
            st.write(f"[LLM] Base URL : `{LLM_BASE_URL}`")
            st.write(f"[LLM] Model    : `{LLM_MODEL}`")
            if use_remote_embedding():
                st.write(f"[EMB] Mode     : remote")
                st.write(f"[EMB] Base URL : `{EMB_BASE_URL}`")
                st.write(f"[EMB] Model    : `{EMB_MODEL}`")
            else:
                st.write(f"[EMB] Mode     : local MiniLM")
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
                ctx = retrieve_with_embedding(query, top_k=3)
            except Exception as e:
                ctx = []
                st.error(f"Embedding エラー: {e}")

        with st.spinner("LLM に問い合わせ中..."):
            answer = call_llm_with_context(query, ctx)

        with st.chat_message("assistant"):
            st.write(answer)
            if ctx:
                with st.expander("🔍 参照したローカルナレッジ"):
                    for i, c in enumerate(ctx, 1):
                        st.markdown(f"**Doc {i}**")
                        st.write(c)
            else:
                st.caption("該当ナレッジなし。")

        add_history(query, answer)
        log_interaction(query, answer, ctx)


if __name__ == "__main__":
    main()
