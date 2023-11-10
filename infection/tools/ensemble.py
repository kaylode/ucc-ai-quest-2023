from typing import *
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", '-o', type=str, default="data")
parser.add_argument('--ann_list', '-l', nargs='+', default=[])

class SemanticEnsembler:
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """
    def __init__(self, out_dir:str) -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def ensemble_inference(self, ann_list: List[str], reduction='sum', **kwargs):
        
        filenames = sorted(os.listdir(ann_list[0]))
        for filename in filenames:
            filepaths = [osp.join(ann, filename) for ann in ann_list]
            stacked_logits = np.stack([np.load(filepath) for filepath in filepaths], axis=0) # [N, H, W]
            if reduction == 'sum':
                if stacked_logits.dtype == 'uint8':
                    raise ValueError('stacked_logits.dtype must be float')
                stacked_logits = np.sum(stacked_logits, axis=0) #[H, W]
            elif reduction == 'max':
                stacked_logits = np.max(stacked_logits, axis=0) #[B, H, W]
            elif reduction == 'mean':
                stacked_logits = np.mean(stacked_logits, axis=0)
            elif reduction == 'mostcommon':
                if stacked_logits.dtype != 'uint8':
                    raise ValueError('stacked_logits.dtype must be uint8')
                stacked_logits = np.argmax(np.bincount(stacked_logits), axis=0)
            else:
                raise ValueError(f"reduction {reduction} is not supported")

            np.save(osp.join(self.out_dir, f"{filename}"), stacked_logits)

def main(args):
    ensembler = SemanticEnsembler(args.out_dir)
    ensembler.ensemble_inference(args.ann_list, reduction='mean')

    # with open(osp.join(args.predict_dir, 'metrics.json'), 'w') as f:
    #     json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
