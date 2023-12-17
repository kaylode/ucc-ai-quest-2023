from typing import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .mosaic import Mosaic

def get_augmentations(split:str, image_size:int=512):

    train_augments = [
        A.Resize(int(image_size*1.25), int(image_size*1.25)),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(),
        A.VerticalFlip(),
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
        A.Resize(image_size, image_size),
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