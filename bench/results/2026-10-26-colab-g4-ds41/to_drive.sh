#!/bin/bash
# copies everything the DeepSeek run needs to Google Drive (gdrive:Strata-DS41); symlinks are followed
set -x
R="rclone --config /root/.config/rclone/rclone.conf --transfers 8 --drive-chunk-size 256M -L --stats 60s --stats-one-line"
$R copy /content/m gdrive:Strata-DS41/m --exclude ".cache/**"
$R copy /content/shared gdrive:Strata-DS41/shared --exclude ".cache/**"
$R copy /content/eng gdrive:Strata-DS41/eng --exclude e.tgz
for f in strata-ds.json mkcfg.py bench_ds4.py vision_ds4.py ctx200k.py summ.py; do $R copyto /content/$f gdrive:Strata-DS41/scripts/$f; done
$R copy /content/logs gdrive:Strata-DS41/logs --include "*.jsonl"
echo DRIVE_DONE
