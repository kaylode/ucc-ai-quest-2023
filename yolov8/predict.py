from ultralytics import YOLO
import argparse
import os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",'-ckpt', type=str)
parser.add_argument("--image-folder", '-i', type=str)
parser.add_argument("--out-dir", '-o', type=str)
parser.add_argument("--image-size", '-sz', type=int)
parser.add_argument("--phase", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    model = YOLO(args.model_path)

    os.makedirs(args.out_dir, exist_ok=True)

    image_names = sorted(os.listdir(args.image_folder))
    for image_name in tqdm(image_names):
        filename = image_name.split('.')[0]
        image_path = osp.join(args.image_folder, image_name)
        
        if not osp.exists(image_path):
            print(f"File {image_path} does not exist")
            continue
        
        result = model(image_path, imgsz=args.image_size, device=0)
        result=result[0]

        # Process results list
        pred = result.masks
        ori_size = result.orig_shape
        if pred:
            pred = torch.nn.functional.interpolate(
                pred.data[0].unsqueeze(0).unsqueeze(0), 
                size=(ori_size[0], ori_size[1]), 
                mode="bilinear", align_corners=False
            ).cpu().squeeze().numpy()

        else:
            pred = np.zeros((ori_size[0], ori_size[1]))

        np.save(osp.join(args.out_dir, f"{filename}.npy"), pred)
