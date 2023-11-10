PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/submission.py \
    -i /home/mpham/workspace/ucc-ai-quest-2023/data/warmup/img/valid \
    -p $PREDICT_FOLDER \
    -o /home/mpham/workspace/ucc-ai-quest-2023/submissions/submission