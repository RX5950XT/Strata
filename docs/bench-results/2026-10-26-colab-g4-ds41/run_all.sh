#!/bin/bash
# the load-time settings, one engine start each; results append to /content/logs/bench.jsonl
cd /content
M="--dspark on --dspark-conf 0.65"
python bench_ds4.py base $M
BENCH_LONG_LENS=16384 python bench_ds4.py e5 $M --experts 5
BENCH_LONG_LENS=16384 python bench_ds4.py mmoff $M --mmap-experts off
BENCH_LONG_LENS=16384 python bench_ds4.py t24 $M --cpu-threads 24
BENCH_LONG_LENS=16384 python bench_ds4.py blk3 $M --dspark-block 3
echo ALLDONE > /content/logs/run_all.done
