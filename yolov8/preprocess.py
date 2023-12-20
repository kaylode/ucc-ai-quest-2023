import os
import os.path as osp
import numpy as np
from rasterio import features
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mask_dir', type=str, default='data/public/ann/train')
parser.add_argument('--save_dir', type=str, default='data/yolov8/public/train/labels')

def create_label(mask_path, save_path):
    arr = np.asarray(Image.open(mask_path))

    # There may be a better way to do it, but this is what I have found so far
    cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[0]} {int(cord[1])/arr.shape[1]}' for cord in cords])

    with open(save_path, 'w') as f:
        f.write(label_line)
    return label_line

def convert_data(mask_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mask_names = os.listdir(mask_dir)
    
    for mask_name in mask_names:
        mask_path = osp.join(mask_dir, mask_name)
        mask_txt_name = mask_name.replace('.png', '.txt')
        out_path = osp.join(save_dir, mask_txt_name)
        create_label(mask_path, out_path)

if __name__ == '__main__':
    args = parser.parse_args()
    convert_data(
        args.mask_dir,
        args.save_dir 
    )