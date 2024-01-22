PHASE=$1
MODEL_NAME=$2

PYTHONPATH=. python infection/tools/predict.py \
    --return_probs \
    -c runs/$PHASE/$MODEL_NAME/checkpoints/best.ckpt \
    -cfg runs/$PHASE/$MODEL_NAME/pipeline.yaml \
    -d data/$PHASE/img/valid \
    -o submissions/validation/$PHASE/$MODEL_NAME