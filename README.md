# Lilot  
ローカルナレッジ検索 × ローカル / クラウド LLM  
（Keyword Search / Embedding Search 対応）

---

## 📌 Overview

**Lilot（リロット）** は、ローカル PC に保存したナレッジ (`knowledge.txt` など) を  
**キーワード検索** または **ベクトル検索（MiniLM Embedding）** によって取得し、  
必要に応じて **OpenAI / Azure / RakutenAI / OpenAI互換API** の LLM に渡して回答を生成するアプリです。

- データはすべてローカル保存 → **情報漏えいリスク最小化**
- MiniLM ローカルモデル同梱 → **Embedding 検索がオフライン動作**
- setup.bat による完全自動セットアップ → **初心者でも簡単**
- 個人用ナレッジ管理から企業 FAQ システムまで幅広く対応

---

# 🏗️ System Architecture

```mermaid
flowchart TD
    User["User (Local PC)"]
    UI["Lilot UI (Streamlit Web UI)"]
    SP["system_prompt.txt"]
    KB["knowledge.txt"]
    Uploads["uploads/（追加ファイル）"]
    Emb["Local MiniLM Embedding Model<br>all-MiniLM-L6-v2"]
    API["Optional: OpenAI / Azure / RakutenAI / OpenAI-Compatible API"]
    Logs["Local Chat Logs<br>(logs/*.jsonl)"]

    User --> UI

    subgraph "Local Knowledge Search"
        UI --> KB
        KB --> UI
        UI --> Uploads
        Uploads --> UI
        UI --> Emb
        Emb --> UI
    end

    subgraph "LLM (Optional API Mode)"
        UI --> API
        API --> UI
    end

    UI --> SP
    UI --> Logs
