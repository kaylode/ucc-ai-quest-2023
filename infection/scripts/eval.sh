PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/eval.py \
    -a /home/mpham/workspace/ucc-ai-quest-2023/data/warmup/ann/valid \
    -p $PREDICT_FOLDER