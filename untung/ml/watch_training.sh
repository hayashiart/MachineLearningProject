#!/bin/zsh
# Watcher training MobileNetV3-Large — dipanggil loop tiap 5 menit.
# Exit codes: 0 = masih jalan | 42 = training+evaluasi selesai | 43 = training mati tanpa selesai
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin

PIDFILE=/tmp/leafscan_train.pid
BASE=/Users/tanujaya/Desktop/Hagongda/deep-learning-image/experiment/MachineLearningProject
ML=$BASE/untung/ml
RUN=$BASE/untung/runs/mobilenetv3_mps_full
PY=$BASE/.venv/bin/python
TRAIN_LOG=/tmp/mobilenet_train.log
STATUS=/tmp/leafscan_watch_status.txt
EVAL_LOG=/tmp/leafscan_eval.log

notify() {
  osascript -e "display notification \"$1\" with title \"LeafScan training\"" 2>/dev/null
}

PID=$(cat $PIDFILE 2>/dev/null)

if [[ -n $PID ]] && ps -p $PID > /dev/null 2>&1; then
  last=$(grep -E '^\[' $TRAIN_LOG | tail -1)
  echo "$(date '+%d %H:%M:%S') RUNNING ${last:-<epoch belum selesai>}" >> $STATUS
  exit 0
fi

if [[ -f $RUN/history.json ]]; then
  if [[ ! -f $RUN/.eval_done ]]; then
    echo "$(date '+%d %H:%M:%S') TRAINING SELESAI - mulai evaluasi" >> $STATUS
    notify "Training selesai! Evaluasi test set sedang jalan..."
    cd $ML && $PY evaluate.py --checkpoint $RUN/best.pt >> $EVAL_LOG 2>&1
    touch $RUN/.eval_done
    acc=$(grep '"test_accuracy"' $RUN/results.json 2>/dev/null | head -1 | tr -d ' ,')
    echo "$(date '+%d %H:%M:%S') EVALUASI SELESAI $acc" >> $STATUS
    notify "Evaluasi selesai! $acc"
  fi
  exit 42
fi

echo "$(date '+%d %H:%M:%S') MATI TANPA SELESAI - cek $TRAIN_LOG" >> $STATUS
notify "Training mati sebelum selesai! Cek log."
exit 43
