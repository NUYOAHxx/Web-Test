#!/bin/bash
# v13 / v14 交替计时（ABAB）：两个模型 × 两轮，每轮 400 次一次性求解（bench/v13/stress.py）
# 用法：PY=<python> bash timing_ab.sh <v13 包目录> <v14 包目录> <输出目录>
#       输出目录里生成 {v13,v14}_{two_arm,wide}_r{1,2}.pkl，再用 timing_summary.py <输出目录> 汇总
set -e
PY=${PY:-python}
V13=$1; V14=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
B=$HERE/../v13/stress.py
mkdir -p "$OUT"
for round in 1 2; do
  for model in two_arm wide; do
    for ver in v13 v14; do
      if [ "$ver" = v13 ]; then root=$V13; else root=$V14; fi
      "$PY" "$B" "$root" "$OUT/${ver}_${model}_r${round}.pkl" $model 200 "" > "$OUT/${ver}_${model}_r${round}.log" 2>&1
      echo "$ver $model r$round done"
    done
  done
done
