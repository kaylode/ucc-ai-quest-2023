PHASE=$1
MODEL_NAME=$2

PYTHONPATH=. python infection/tools/train.py \
    --config-dir infection/configs \
    --config-name $PHASE \
    model.model_name=$MODEL_NAME \
    trainer.save_dir=runs/$PHASE/${MODEL_NAME} \
    data.train_img_dir=data/$PHASE/img/train \
    data.train_ann_dir=data/$PHASE/ann/train \
    data.val_img_dir=data/$PHASE/img/valid \
    data.val_ann_dir=data/$PHASE/ann/valid