import os
import os.path as osp
import numpy as np
import cv2
import json
import torch
from infection.augmentations import get_augmentations
from infection.datasets import SegDataset
from infection.models import SegModel
from infection.submission.utils import mask_to_rle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", '-c', type=str, required=True)
parser.add_argument("--root_dir", '-d', type=str, default="data")
parser.add_argument("--out_dir", '-o', type=str, default="results/")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    model = SegModel.load_from_checkpoint(args.ckpt_path)
    ds = SegDataset(
        root_dir=args.root_dir,
        phase="warmup", 
        split="valid", 
        transform=get_augmentations("valid")
    )
    model.eval()
    results = {}
    for i, batch in enumerate(ds):
        filename = ds.img_list[i]
        img, _, ori_size = batch

        out = model(img.float().unsqueeze(dim=0).to(model.device))
        if isinstance(out, dict):
            out = out["out"]
        
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1)
        pred = pred.detach().cpu().numpy().squeeze()
        stacked_pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2) # h,w,c
        resized_pred = cv2.resize(stacked_pred.astype(np.uint8), tuple([ori_size[1], ori_size[0]]))[...,-1]

        rle = mask_to_rle(resized_pred)
        results[filename] = {
            "counts": rle,
            "height": resized_pred.shape[0],
            "width": resized_pred.shape[1],
        }
    
    with open(osp.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
