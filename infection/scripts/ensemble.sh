PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/ensemble.py \
    -o /home/mpham/workspace/ucc-ai-quest-2023/submissions/prediction/warmup/ensemble/ \
    -l /home/mpham/workspace/ucc-ai-quest-2023/submissions/prediction/warmup/unetpp \
    /home/mpham/workspace/ucc-ai-quest-2023/submissions/prediction/warmup/dlv3pp