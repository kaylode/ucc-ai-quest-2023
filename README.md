# UCC AI Quest 2023

https://challenges.ai/ucc-ai-quest-2023
https://github.com/ReML-AI/ucc-ai-quest-baseline


## Install dependencies
```
conda create -n ucc python=3.10
conda activate ucc
git clone https://github.com/kaylode/ucc-ai-quest-2023
cd ucc-ai-quest-2023
pip install -e .
```

## Unzip warmup.zip in data folder so it has the following structure

```
data/warmup/img/train
data/warmup/img/valid
data/warmup/ann/train
data/warmup/ann/valid
```

## Training
```
python train.py
```

## Prepare results for submission

After training, the checkpoints are stored in folder `checkpoints`. We need to prepare a file named "results.json" for submission on CodaLab. Use the script `prepare_results_for_submission.py` and replace the checkpoint path 

```
model = SegModel.load_from_checkpoint("checkpoints/epoch=6-val_loss=0.47-val_high_vegetation_IoU=63.55-val_mIoU=66.48.ckpt")
```
and the phase (warmup/public/private) and the split accordingly
```
ds = SegDataset(phase="warmup", split="valid", transform=ToTensorV2())
```

then run`
```
python prepare_results_for_submission.py
```

there should be a file "results.json" generated in the current directory. Zip the file

```
zip results.zip results.json
```

Done! You should be able to submit the file `results.zip` now, good luck!