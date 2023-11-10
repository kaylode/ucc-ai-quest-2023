PHASE=$1
MODEL_PATH=$2
MODEL_NAME=$(basename $MODEL_PATH)

PYTHONPATH=. python infection/tools/predict.py \
    -c $MODEL_PATH \
    --return_probs \
    -d /home/mpham/workspace/ucc-ai-quest-2023/data \
    -o /home/mpham/workspace/ucc-ai-quest-2023/submissions/prediction/$PHASE/$MODEL_NAME