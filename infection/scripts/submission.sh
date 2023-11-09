MODEL_PATH=$1

PYTHONPATH=. python infection/submission/prepare_results_for_submission.py \
    -c $MODEL_PATH \
    -d /home/mpham/workspace/ucc-ai-quest-2023/data \
    -o /home/mpham/workspace/ucc-ai-quest-2023/submissions/