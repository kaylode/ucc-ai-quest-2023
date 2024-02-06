from .datasets import SegDataset, SegDataModule
from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
)
import torch
from torch.utils.data import DataLoader
from infection.augmentations import get_augmentations, Mosaic

class TransformersDataset(SegDataset):
    def __init__(self, model_name:str, **kwargs):
        super().__init__(**kwargs)
        if model_name == 'maskformer':
            self.preprocessor = MaskFormerImageProcessor(
                ignore_index=0, reduce_labels=True,
                do_resize=False, do_rescale=False, 
                do_normalize=False
            )

        elif model_name == 'mask2former':
            self.preprocessor = AutoImageProcessor.from_pretrained(
                "facebook/mask2former-swin-base-coco-panoptic",
            )
        else:
            raise NotImplementedError

    def collate_fn(self, batch):
        
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]
        ori_sizes = inputs[2]

        batch = self.preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        # batch["ori_sizes"] = torch.LongTensor(ori_sizes)
        # import pdb; pdb.set_trace()

        return batch['pixel_values'],  batch['mask_labels'], batch['class_labels']


class TransformersDataModule(SegDataModule):
    def __init__(
            self, 
            model_name,
            **kwargs
        ):
        super().__init__(
            **kwargs
        )

        self.model_name = model_name

    def setup(self, stage: str):
        self.train = TransformersDataset(
            model_name=self.model_name,
            img_dir=self.train_img_dir,
            ann_dir=self.train_ann_dir,
            use_mosaic=self.use_mosaic,
            image_size=self.image_size,
            transform=get_augmentations("train", image_size=self.image_size, stats=self.stats),
        )
        self.valid = TransformersDataset(
            model_name=self.model_name,
            img_dir=self.val_img_dir,
            ann_dir=self.val_ann_dir,
            use_mosaic=0,
            image_size=self.image_size,
            transform=get_augmentations("valid", image_size=self.image_size, stats=self.stats)
        )
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=self.train.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.valid.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.valid.collate_fn)