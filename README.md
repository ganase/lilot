# Lilot
ローカルナレッジ検索 × OpenAI互換API（Embedding Search / ローカルRAG）

---

## 📌 Overview
**Lilot（リロ）** は、ローカルPC上のナレッジ（`data/knowledge.txt` と `data/uploads/`）を **Embedding検索（ベクトル検索）** で参照し、
必要に応じて **OpenAI互換API（OpenAI / Azure OpenAI / LocalAI / LM Studio など）** に問い合わせて回答を生成する、Streamlit製のローカルRAGチャットアプリです。

- 検索方式：**Embedding検索のみ**（Keyword検索は廃止）
- Embedding：**ローカル MiniLM** / **リモート Embedding API（OpenAI互換）** 切替
- LLM：**OpenAI互換API**（ローカルLLM直結は未実装）
- ナレッジ / ログ / アップロードファイルは **すべてローカル保存**

---

## ✨ Features
### 🔍 Embedding検索（ローカル/リモート切替）
- ローカル：`sentence-transformers/all-MiniLM-L6-v2`
- リモート：`.env` に `EMB_API_KEY` を設定すると OpenAI互換 Embedding API を使用

### 📎 添付ファイル（アップロード）
Streamlit UI からファイルをアップロードできます。

- 対応：`txt` / `csv` / `pdf`（複数PDFアップロード可）
- 保存先：`data/uploads/`
- アップロード後、**検索インデックスに反映**（UIから「保存して取り込む」を実行）

> PDFは `pypdf` によるテキスト抽出です。画像PDF（スキャンのみ等）は文字が取れない場合があります。

#### PDFの「再抽出しない」仕組み（起動を軽くするため）
- PDFは **アップロード時にテキスト抽出 → JSON保存** します
- 次回以降の起動時は **PDF本体を再抽出せず**、抽出済みJSONを読み込みます
- PDFが更新された場合（ファイル内容が変わった場合）は、再アップロードで再抽出されます

---

## 📁 Folder Structure
```
lilot/
├── app/
│   └── app_emb.py                  # Embedding検索版（本体）
├── data/
│   ├── knowledge.txt               # メインナレッジ
│   ├── system_prompt.txt           # LLM振る舞い設定
│   └── uploads/                    # 添付ファイル保存先
│       ├── original/               # 元ファイル（txt/csv/pdf）
│       ├── extracted/              # PDF抽出テキスト（json）
│       └── index_meta.json         # 抽出/更新管理（内部用）
├── models/
│   └── all-MiniLM-L6-v2/           # ローカルEmbeddingモデル（任意配置）
├── logs/
│   └── YYYYMMDD_xxx.jsonl          # チャットログ
├── run_app_emb.bat                 # 起動（Embedding）
├── install_requirements_conda.bat  # 依存インストール（Miniforge base）
├── setup.bat                       # セットアップ（環境構築）
├── requirements.txt                # Python依存
└── README.md
```

---

## 🧑‍💻 Setup Guide（Windows）
### 1) Miniforge をインストール
- 既にある場合は不要
- 既定パス：`%USERPROFILE%\miniforge3`

### 2) セットアップ
- `setup.bat` を実行（推奨）
  - 依存ライブラリのインストール
  - 必要に応じてショートカット作成（構成による）

### 3) 起動
- `run_app_emb.bat` を実行

---

## ⚙️ .env 設定（任意）
プロジェクト直下の `.env` に以下を設定できます。

### LLM（回答生成）
- `LLM_API_KEY`
- `LLM_BASE_URL`（例：`https://api.openai.com/v1` / Azure / ローカルAPI）
- `LLM_MODEL`（例：`gpt-4o-mini` 等）

### Embedding（リモートに切替）
- `EMB_API_KEY`（これが入るとリモートEmbeddingを使用）
- `EMB_BASE_URL`（既定：`https://api.openai.com/v1`）
- `EMB_MODEL`（既定：`text-embedding-3-small`）

---

## 🧩 ナレッジの更新方法
- `data/knowledge.txt` を編集 → 次回検索から反映
- UI の「📎 添付ファイル」からアップロード → `data/uploads/original/` に保存 → 「保存して取り込む」実行で検索対象に反映

---

## 🧾 ログ
- 会話ログは `logs/*.jsonl` に保存されます
- UI 左ペインの「🧾 ログ」からログを読み込めます（→ボタン）

---

## 🛠️ Troubleshooting
- **起動しない / Miniforgeが見つからない**  
  `run_app_emb.bat` が `%USERPROFILE%\miniforge3\python.exe` を探します。  
  Miniforgeのインストール先が違う場合は、`run_app_emb.bat` の `CONDA_ROOT` を変更してください。

- **PDFが検索にヒットしない**  
  画像PDF（スキャン）だとテキスト抽出できません。テキスト付きPDFでお試しください。  
  また、アップロード後に「保存して取り込む」を実行したか確認してください。

---

## 📝 License
MIT License
