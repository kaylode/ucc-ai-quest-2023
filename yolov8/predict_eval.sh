PHASE=$1
MODEL_NAME=$2

python yolov8/predict.py \
    --model-path runs/$PHASE/yolov8/$MODEL_NAME/weights/best.pt \
    --image-folder data/$PHASE/img/valid \
    -sz 640 \
    --phase $PHASE \
    -o submissions/validation/$PHASE/$MODEL_NAME
