PHASE=$1
MODEL_NAME=$2
RUN_NAME=${MODEL_NAME}_diceohemce_mosaic

PYTHONPATH=. python infection/tools/train.py \
    --config-dir infection/configs \
    --config-name $PHASE \
    model.model_name=$MODEL_NAME \
    trainer.save_dir=runs/$PHASE/$RUN_NAME \
    data.train_img_dir=data/$PHASE/img/train \
    data.train_ann_dir=data/$PHASE/ann/train \
    data.val_img_dir=data/$PHASE/img/valid \
    data.val_ann_dir=data/$PHASE/ann/valid

PYTHONPATH=. python infection/tools/predict.py \
    --return_probs \
    -c runs/$PHASE/$RUN_NAME/checkpoints/best.ckpt \
    -cfg runs/$PHASE/$RUN_NAME/pipeline.yaml \
    -d data/$PHASE/img/valid \
    -o submissions/validation/$PHASE/$RUN_NAME

PYTHONPATH=. python infection/tools/eval.py \
    -a data/$PHASE/ann/valid \
    -p submissions/validation/$PHASE/$RUN_NAME