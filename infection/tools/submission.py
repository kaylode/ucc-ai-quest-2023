import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import json
from infection.submission.utils import mask_to_rle
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", '-i', type=str)
parser.add_argument("--predict_dir", '-p', type=str)
parser.add_argument("--out_dir", '-o', type=str, default="results/")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    image_names = sorted(os.listdir(args.image_dir))
    
    results = {}
    for filename in image_names:
        prefilename = osp.splitext(filename)[0]
        predict_path = osp.join(args.predict_dir, prefilename+'.npy')

        if not osp.exists(predict_path):
            print(f"{predict_path} does not exists")
            continue
        
        image = Image.open(osp.join(args.image_dir, filename))
        pred = np.load(predict_path)
        pred = (pred > 0.5).astype(np.uint8)
        stacked_pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2) # h,w,c
        
        resized_pred = cv2.resize(
            stacked_pred, 
            tuple([image.width, image.height])
        )[...,-1]

        rle = mask_to_rle(resized_pred)
        results[filename] = {
            "counts": rle,
            "height": image.height,
            "width": image.width,
        }

    print(f"Number of results: {len(results)}")
    
    with open(osp.join(args.out_dir, f"{datetime.now()}.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
