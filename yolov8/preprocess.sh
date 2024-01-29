PHASE=$1

python yolov8/preprocess.py \
    --mask_dir data/$PHASE/ann/train \
    --save_dir data/yolov8/$PHASE/train/labels

# python yolov8/preprocess.py \
#     --mask_dir data/$PHASE/ann/valid \
#     --save_dir data/yolov8/$PHASE/valid/labels

# ln -s $PWD/data/$PHASE/img/train $PWD/data/yolov8/$PHASE/train/images
# ln -s $PWD/data/$PHASE/img/valid $PWD/data/yolov8/$PHASE/valid/images