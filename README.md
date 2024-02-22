# Infection Team's solution for the [UCC AI Quest 2023](https://challenges.ai/ucc-ai-quest-2023)

We are team Infection. This repository demonstrates our methods in the UCC AI Quest 2023 in Cork, Ireland. This repo is built based on the provided [baseline code](https://github.com/ReML-AI/ucc-ai-quest-baseline).

A brief introduction of the competition:
> Cork is blessed with breathtaking landscapes and serene greenery. This year, UCC AI Quest will focus on stunning aerial images of a high-resolution camera to recognise vegetation patches in Irish natural places. The challenge aims to foster the development of reliable artificial intelligence models with the goal of informing sustainable development. It includes the release of a new dataset of realistic drone images for benchmarking semantic segmentation from various above ground levels. There are a number of awards for the best team (€5,000), the most creative solution (€1,000) and the top women of influence (€1,000). Read more about the competition on their [website](https://challenges.ai/ucc-ai-quest-2023).

## Highlighted techniques 

- Our main contributions are:
  - Training a [DinoV2-ViTB14](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb) with a customized two-layer FCN for semantic segmentation task.
  - Combining both region-based and class-based loss functions as objective function, namely Dice loss and [OHEMCE loss](https://paperswithcode.com/method/ohem)
  - Applying [Mosaic Augmentation](https://medium.com/mlearning-ai/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf) to generate variety of complex data scenarios to enhance models' training
     <details close>
     <summary><strong>Introducing simple yet effective technique to finetune small-scaled dataset on such large state-of-the-art model</strong></summary>
     <p align="justify">
        <ol>
          <li>Freeze the encoder backbone layers, only train/fine-tune the segmentation head/decoder layers.</li>
          <li>Unfreeze all the layers of the network, fine-tune the whole model with 10 times smaller learning rate</li>
        </ol>
     </p>
    </details>
    <details close>
     <summary><strong>Employing ensemble method that further boost the precision of predicted masks</strong></summary>
     <p align="justify">
       We choose top 5 models that have highest metric score on our validation set:
        <ul>
          <li> two deeplabv3+ (efficientnet B4&B5) </li>
          <li> one unet++ (efficientnet B4).</li>
          <li> two dinov2-base </li>
        </ul>
      We gathered all the probability masks predicted by the models and average them to get the final segmentation mask for the private set
     </p>
    </details>


## Data

- Warmup, Public set and private sets are provided from the competition organizers
- Unzip warmup.zip/public.zip/private.zip in data folder so it has the following structure

```
data/$PHASE/img/train
data/$PHASE/img/valid
data/$PHASE/img/test
data/$PHASE/ann/train
data/$PHASE/ann/valid
```
where $PHASE could be warmup/public/private

## Install dependencies
```
conda create -n ucc python=3.10
conda activate ucc
git clone https://github.com/kaylode/ucc-ai-quest-2023
cd ucc-ai-quest-2023
pip install -e .
```

## Execution scripts

- We provide simple Colab Notebook to execute our code. [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OiTAt0y2GJjcXsYt1zs98PbiTlgV3LEn?usp=sharing)

- Firtly, configuration can be adjusted inside `infection/configs` folder depending to the $PHASE
- To train models, run the below script
```
sh infection/scripts/train.sh $PHASE $MODEL_NAME
```
where $MODEL_NAME follows this format: $ARCHITECTURE.$BACKBONE (for example, deeplabv3plus.timm-efficientnet-b4) or other models could be implemented and specified accordingly. 
- Run experiments will be stored in `runs/$PHASE/$MODEL_NAME`.
- Our best performing model is DinoV2-base

- Training pipeline can be seen in [infection/tools/train.py](./infection/tools/train.py)


- For evaluation and submission, prediction script should be run first
```
sh infection/scripts/predict.sh $PHASE $MODEL_NAME
```
- then the prediction will be saved into $PREDICTION_FOLDER=`submissions/prediction/$PHASE/$MODEL_NAME`

- then the submission script can be run
```
sh infection/scripts/submission.sh $PHASE $PREDICTION_FOLDER
```
- this will result in a json file for submission in `submissions/submission` folder. **Remember to zip it before submission**

> [!WARNING]
> Before submission, please rename the json file to `results.json` and zip it as `results.zip`

> [!TIP]
> Ensemble method can be run using `scripts/ensemble.sh`. It will gather results from all the npy files generated from `scripts/predict.sh` and combine into a new folder. Then `scripts/submission.sh` can be run to generate submission file from this folder.


## Discussion

What we have tried but did not work:
  - Several state-of-the-art models: Original ViT, OneFormer, SegFormer, MaskFormer, Mask2Former,YOLOv8
  - Different segmentation losses: Focal Tversky loss, Lovasz-Softmax loss
  - “Smoothen” the boundary of the segmentation masks in predictions since data annotations do not have good quality

What we haven't tried:
  - Implement hyperparameter tuning more thoroughly
  - Replace two-layer FCNs with more complex segmentation head (for example, Mask2Former head)
