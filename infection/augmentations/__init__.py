import albumentations as A
from albumentations.pytorch import ToTensorV2
from .mosaic import Mosaic

def get_augmentations(split:str):
    if split == 'train':
        return A.Compose([
            A.RandomCrop(380, 380),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            ToTensorV2(),
        ])