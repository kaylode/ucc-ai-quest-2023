import os
import os.path as osp
import numpy as np
import torch
from infection.augmentations import get_augmentations
from infection.datasets import SegDataset
from infection.models import SegModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", '-c', type=str, required=True)
parser.add_argument("--root_dir", '-d', type=str, default="data")
parser.add_argument("--out_dir", '-o', type=str, default="results/")
parser.add_argument("--return_probs", action='store_true', default=False)



def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    model = SegModel.load_from_checkpoint(args.ckpt_path)
    ds = SegDataset(
        root_dir=args.root_dir,
        phase="warmup", 
        split="valid", 
        transform=get_augmentations("valid", image_size=512)
    )
    model.eval()
    for i, batch in enumerate(ds):
        filename = osp.splitext(ds.img_list[i])[0]
        img, _, ori_size = batch

        out = model(img.float().unsqueeze(dim=0).to(model.device))
        if isinstance(out, dict):
            out = out["out"]
        
        probs = torch.softmax(out, dim=1)
        if not args.return_probs:
            pred = torch.argmax(probs, dim=1)
        else:
            pred = probs[:,1,...]
        
        pred = pred.detach().cpu().numpy().squeeze()
        np.save(osp.join(args.out_dir, f"{filename}.npy"), pred)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
