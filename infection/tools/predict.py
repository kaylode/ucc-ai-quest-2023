import os
import os.path as osp
import numpy as np
import torch
from infection.augmentations import get_augmentations
from infection.datasets import SegDataset
from infection.models import SegModel
import argparse
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", '-c', type=str, required=True)
parser.add_argument("--config-file", '-cfg', type=str, required=True)
parser.add_argument("--img_dir", '-d', type=str, default="data")
parser.add_argument("--out_dir", '-o', type=str, default="results/")
parser.add_argument("--return_probs", action='store_true', default=False)



def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    model = SegModel.load_from_checkpoint(args.ckpt_path)
    ds = SegDataset(
        img_dir=args.img_dir,
        transform=get_augmentations("valid", image_size=config['data'].get('image_size', 512))
    )
    model.eval()
    for i, batch in tqdm(enumerate(ds)):
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
