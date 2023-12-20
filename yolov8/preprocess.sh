python yolov8/preprocess.py \
    --mask_dir data/public/ann/train \
    --save_dir data/yolov8/public/train/labels

python yolov8/preprocess.py \
    --mask_dir data/public/ann/valid \
    --save_dir data/yolov8/public/valid/labels