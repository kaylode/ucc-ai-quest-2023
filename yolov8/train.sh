PHASE=$1
MODEL_NAME=$2

python yolov8/train.py \
    -n $MODEL_NAME \
    -cfg yolov8/$PHASE.yml \
    -sz 640 \
    --phase $PHASE