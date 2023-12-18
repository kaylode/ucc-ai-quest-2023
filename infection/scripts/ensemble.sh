PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/ensemble.py \
    -o submissions/prediction/warmup/ensemble/ \
    -l submissions/prediction/warmup/unetpp \
    submissions/prediction/warmup/dlv3pp