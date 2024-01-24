import os
import os.path as osp
import json
import cv2
import numpy as np
from PIL import Image
from infection.metrics import SMAPIoUMetric
from infection.utilities.visualization import Visualizer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ann_dir", '-a', type=str, default="data")
parser.add_argument("--predict_dir", '-p', type=str, default="data")

def main(args):
    evaluator = SMAPIoUMetric()
    visualizer = Visualizer()
    
    filenames = sorted(os.listdir(args.ann_dir))
    for filename in filenames:
        ann_path = osp.join(args.ann_dir, filename)
        mask = np.array(Image.open(ann_path))

        filename = osp.splitext(filename)[0]
        predict_path = osp.join(args.predict_dir, filename+'.npy')
        pred = np.array(np.load(predict_path))

        pred = (pred > 0.5).astype(np.uint8)

        # img_show = visualizer.denormalize(inputs)
        # decode_mask = visualizer.decode_segmap(mask.numpy())
        # img_show = TFF.to_tensor(img_show)
        # decode_mask = TFF.to_tensor(decode_mask / 255.0)
        # img_show = torch.cat([img_show, decode_mask], dim=-1)
        # batch.append(img_show)
        # grid_img = visualizer.make_grid(batch)
        try:
            evaluator.process(input={"pred": pred, "gt": mask})
        except:
            print(pred.shape, mask.shape)
            import pdb; pdb.set_trace()

    print("Number of samples:", len(filenames))

    metrics = evaluator.evaluate(0)
    print(metrics)

    # with open(osp.join(args.predict_dir, 'metrics.json'), 'w') as f:
    #     json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
