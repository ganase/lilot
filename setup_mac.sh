{\rtf1\ansi\ansicpg932\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env bash\
set -euo pipefail\
\
cd "$(dirname "$0")"\
\
ENV_NAME="lilot"\
PY_VER="3.11"\
\
# conda detection (Miniforge/Miniconda/Anaconda)\
if command -v conda >/dev/null 2>&1; then\
  CONDA_BIN="$(command -v conda)"\
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then\
  CONDA_BIN="$HOME/miniforge3/bin/conda"\
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then\
  CONDA_BIN="$HOME/miniconda3/bin/conda"\
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then\
  CONDA_BIN="$HOME/anaconda3/bin/conda"\
else\
  echo "[ERROR] conda \uc0\u12364 \u35211 \u12388 \u12363 \u12426 \u12414 \u12379 \u12435 \u12290 Miniforge/Miniconda/Anaconda \u12434 \u20808 \u12395 \u23566 \u20837 \u12375 \u12390 \u12367 \u12384 \u12373 \u12356 \u12290 "\
  exit 1\
fi\
\
echo "============================================"\
echo " Lilot Setup (macOS)"\
echo " Project: $(pwd)"\
echo "============================================"\
echo "[INFO] Using conda: $CONDA_BIN"\
\
# make conda function available in non-interactive shells\
# shellcheck disable=SC1090\
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"\
\
if conda env list | awk '\{print $1\}' | grep -qx "$ENV_NAME"; then\
  echo "[INFO] Conda env \\"$ENV_NAME\\" already exists."\
else\
  echo "[INFO] Creating conda env \\"$ENV_NAME\\" (python=$PY_VER) ..."\
  conda create -n "$ENV_NAME" python="$PY_VER" -y\
fi\
\
conda activate "$ENV_NAME"\
\
echo "[INFO] Upgrading pip..."\
python -m pip install --upgrade pip\
\
echo "[INFO] Installing requirements..."\
pip install -r requirements.txt\
\
echo\
echo "\uc0\u9989  Setup complete."\
echo "\uc0\u27425 \u12399  streamlit run app/app_emb.py \u12391 \u36215 \u21205 \u12391 \u12365 \u12414 \u12377 \u12290 "\
}