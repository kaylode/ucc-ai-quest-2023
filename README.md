# UCC AI Quest 2023 - Infection Team

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OiTAt0y2GJjcXsYt1zs98PbiTlgV3LEn?usp=sharing)


Official links:
- Competition information: https://challenges.ai/ucc-ai-quest-2023
- This repo is built based on the provided baseline code: https://github.com/ReML-AI/ucc-ai-quest-baseline

## Suggestion for improvements

- Augmentations:
  - [x] [Mosaic Augmentation](https://medium.com/mlearning-ai/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf)
- Loss functions:
  - [x] Dice loss
  - [x] Focal Tversky loss
  - [x] Lovasz-Softmax loss
  - [x] [Online Hard Example Mining (OHEM)](https://paperswithcode.com/method/ohem)
- Models:
  - [ ] Transformers-based models: [OneFormer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OneFormer/Fine_tune_OneFormer_for_semantic_segmentation.ipynb), [SegFormer](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer), [MaskFormer and Mask2Former](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Mask2Former), [DinoV2](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb),...
  - [ ] [YOLOv8](https://github.com/ultralytics/ultralytics)
- Postprocessing:
  - [x] Ensemble 
    


## Data

- Warmup: https://drive.google.com/file/d/1OqHUM5z5AMmXQxE-R0zzPeePRDGZY1uB
- Public set: https://drive.google.com/file/d/1rkecNZKd-dQFbXalUCRMyQkrMmL_xwPN

Unzip warmup.zip/public.zip in data folder so it has the following structure

```
data/$PHASE/img/train
data/$PHASE/img/valid
data/$PHASE/img/test
data/$PHASE/ann/train
data/$PHASE/ann/valid
```
where $PHASE could be warmup/public

## Install dependencies
```
conda create -n ucc python=3.10
conda activate ucc
git clone https://github.com/kaylode/ucc-ai-quest-2023
cd ucc-ai-quest-2023
pip install -e .
```

## Execution scripts

Firtly, configuration can be adjusted inside `infection/configs` folder depending to the $PHASE

To train models, run the below script
```
sh infection/scripts/train.sh $PHASE $MODEL_NAME
```
where $MODEL_NAME follows this format: $ARCHITECTURE.$BACKBONE (for example, deeplabv3plus.timm-efficientnet-b4) or other models could be implemented and specified accordingly. Run experiments will be stored in `runs/$PHASE/$MODEL_NAME`.

Training pipeline can be seen in [infection/tools/train.py](./infection/tools/train.py)


For evaluation and submission, prediction script should be run first
```
sh infection/scripts/predict.sh $PHASE $MODEL_NAME
```
then the prediction will be saved into $PREDICTION_FOLDER=`submissions/prediction/$PHASE/$MODEL_NAME`

then the submission script can be run
```
sh infection/scripts/submission.sh $PHASE $PREDICTION_FOLDER
```
this will result in a json file for submission in `submissions/submission` folder. **Remember to zip it before submission**

> [!WARNING]
> Before submission, please rename the json file to `results.json` and zip it as `results.zip`


> [!TIP]
> Ensemble method can be run using `scripts/ensemble.sh`. It will gather results from all the npy files generated from `scripts/predict.sh` and combine into a new folder. Then `scripts/submission.sh` can be run to generate submission file from this folder.
