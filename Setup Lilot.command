{\rtf1\ansi\ansicpg932\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env bash\
set -euo pipefail\
\
DIR="$(cd "$(dirname "$0")" && pwd)"\
cd "$DIR"\
\
# \uc0\u21021 \u22238 \u12384 \u12369 \u23455 \u34892 \u27177 \u38480 \u12364 \u28961 \u12356 \u12465 \u12540 \u12473 \u12364 \u12354 \u12427 \u12398 \u12391 \u20184 \u19982 \
chmod +x "./setup_mac.sh" || true\
\
echo "============================================"\
echo " Lilot Setup Launcher (macOS)"\
echo " Project: $DIR"\
echo "============================================"\
\
./setup_mac.sh\
\
echo\
echo "Press Enter to close..."\
read -r\
}