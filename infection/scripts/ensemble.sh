PREDICT_FOLDER=$1

PYTHONPATH=. python infection/tools/ensemble.py \
    -o submissions/prediction/public/ensemble/ \
    -l submissions/prediction/public/deeplabv3plus.timm-efficientnet-b4_diceohemce_mosaic \
    submissions/prediction/public/unetplusplus.timm-efficientnet-b4_dicece \
    submissions/prediction/public/unetplusplus.timm-efficientnet-b4_dicece_mosaic \
    submissions/prediction/public/unetplusplus.timm-efficientnet-b4_diceohemce_mosaic \
    submissions/prediction/public/unetplusplus.timm-efficientnet-b4_lovaszce_mosaic \
    submissions/prediction/public/unetplusplus.timm-efficientnet-b4_tverskyce_mosaic
    # submissions/validation/public/unetplusplus.timm-efficientnet-b4_diceohemce_mosaic_1024 \