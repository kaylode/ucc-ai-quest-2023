PHASE=$1
PREDICT_FOLDER=$2

PYTHONPATH=. python infection/tools/eval.py \
    -a data/$PHASE/ann/valid \
    -p $PREDICT_FOLDER