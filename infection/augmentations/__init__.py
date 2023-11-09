from typing import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .mosaic import Mosaic

def get_augmentations(split:str):

    train_augments = [
        A.Resize(512, 512),
        A.RandomCrop(384, 384),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]

    val_augments = [
        A.Resize(384, 384),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]

    test_augments = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]

    if split == 'train':
        return A.Compose(train_augments)
    elif split == 'valid':
        return A.Compose(val_augments)
    elif split == 'test':
        return A.Compose(test_augments)
    raise ValueError(f"Invalid split name: {split}")