PHASE=$1
MODEL_NAME=$2

PYTHONPATH=. python infection/tools/train.py \
    --config-dir /home/mpham/workspace/ucc-ai-quest-2023/infection/configs \
    --config-name $PHASE \
    model.model_name=$MODEL_NAME \
    trainer.save_dir=/home/mpham/workspace/ucc-ai-quest-2023/runs/$PHASE/$MODEL_NAME