import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from infection.metrics import SMAPIoUMetric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ann_dir", '-a', type=str, default="data")
parser.add_argument("--predict_dir", '-p', type=str, default="data")

def main(args):
    evaluator = SMAPIoUMetric()
    
    filenames = sorted(os.listdir(args.ann_dir))
    for filename in filenames:
        ann_path = osp.join(args.ann_dir, filename)
        mask = np.array(Image.open(ann_path))

        filename = osp.splitext(filename)[0]
        predict_path = osp.join(args.predict_dir, filename+'.npy')
        pred = np.array(np.load(predict_path))

        pred = (pred > 0.5).astype(np.uint8)
        stacked_pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2) # h,w,c
        resized_pred = cv2.resize(
            stacked_pred, 
            tuple([mask.shape[1], mask.shape[0]])
        )[...,-1]

        evaluator.process(input={"pred": resized_pred, "gt": mask})

    print("Number of samples:", len(filenames))

    metrics = evaluator.evaluate(0)
    print(metrics)

    # with open(osp.join(args.predict_dir, 'metrics.json'), 'w') as f:
    #     json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
