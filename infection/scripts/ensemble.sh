PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/ensemble.py \
    -o submissions/prediction/public/ensemble/ \
    -l submissions/validation/public/deeplabv3plus.timm-efficientnet-b4 \
    submissions/validation/public/unetplusplus.timm-efficientnet-b4