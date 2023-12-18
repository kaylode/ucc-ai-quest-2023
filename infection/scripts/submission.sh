PHASE=$1
PREDICT_FOLDER=$2

PYTHONPATH=. python infection/tools/submission.py \
    -i data/$PHASE/img/test \
    -p $PREDICT_FOLDER \
    -o submissions/submission