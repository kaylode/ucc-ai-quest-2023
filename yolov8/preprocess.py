import os
import os.path as osp
import numpy as np
from rasterio import features
from PIL import Image
import argparse
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--mask_dir', type=str, default='data/public/ann/train')
parser.add_argument('--save_dir', type=str, default='data/yolov8/public/train/labels')

def create_label2(file_path, out_path):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    width, height = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for obj in contours:
        coords = []
            
        for point in obj:
            coords.append(int(point[0][0])/height)
            coords.append(int(point[0][1])/width)

        polygons.append(coords)

    with open(out_path, 'w') as f:
        for polygon in polygons:
            label_line = '0 ' + ' '.join([str(cord) for cord in polygon])
            f.write(label_line)
            f.write('\n')

    return polygons

def create_label(mask_path, save_path):
    arr = np.asarray(Image.open(mask_path))

    # There may be a better way to do it, but this is what I have found so far
    all_cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates']
    
    import pdb; pdb.set_trace()
    
    with open(save_path, 'w') as f:
        for cords in all_cords:
            label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[1]} {int(cord[1])/arr.shape[0]}' for cord in cords])
            f.write(label_line)
            f.write('\n')
    return label_line

def convert_data(mask_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mask_names = os.listdir(mask_dir)
    
    for mask_name in tqdm(mask_names):
        mask_path = osp.join(mask_dir, mask_name)
        mask_txt_name = mask_name.replace('.png', '.txt')
        out_path = osp.join(save_dir, mask_txt_name)
        create_label2(mask_path, out_path)

if __name__ == '__main__':
    args = parser.parse_args()
    convert_data(
        args.mask_dir,
        args.save_dir 
    )