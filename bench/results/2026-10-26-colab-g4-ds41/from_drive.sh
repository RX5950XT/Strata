#!/bin/bash
# On a fresh Colab G4: pull DeepSeek-V4.1-Flash Uncensored IQ2_XXS (MTP, vision, 200K) from Google Drive and serve it on :8095.
# The 194 GB disk cannot hold all 240 GiB: shards 1 and 4, the drafter and the vision tower go to /dev/shm (RAM), as the pack expects.
set -e
command -v rclone >/dev/null || curl -s https://rclone.org/install.sh | bash >/dev/null
R="rclone --transfers 8 --multi-thread-streams 8 --stats 60s --stats-one-line"
S=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS
mkdir -p /content/m /content/shared /dev/shm/m /content/logs
$R copy gdrive:Strata-DS41/m /dev/shm/m --include "$S-00001-*" --include "$S-00004-*" --include "dspark-*" --include "mmproj-*" &
$R copy gdrive:Strata-DS41/m /content/m --include "$S-00002-*" --include "$S-00003-*" --include "maya-ds4/**" &
$R copy gdrive:Strata-DS41/shared /content/shared &
$R copy gdrive:Strata-DS41/eng /content/eng &
$R copy gdrive:Strata-DS41/scripts /content &
wait
chmod +x /content/eng/strata-dsv41
for f in /dev/shm/m/*.gguf; do ln -sf "$f" /content/m/; done
[ -d /content/Strata ] || git clone -q https://github.com/Niko1221/Strata.git /content/Strata
git -C /content/Strata checkout -q 4b9e05a2
cd /content/Strata
setsid nohup python3 serve/server.py --engine strata --config /content/strata-ds.json --port 8095 > /content/logs/server.out 2>&1 < /dev/null &
echo "serving on http://127.0.0.1:8095/v1 once /content/logs/server.out says ready (the load takes ~5 minutes)"
