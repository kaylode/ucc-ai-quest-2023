import os
import random
import numpy as np
import lightning.pytorch as pl
from infection.augmentations import get_augmentations, Mosaic
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SegDataset(Dataset):
    def __init__(self, root_dir="data", phase="warmup", split="train", image_size:int=512, use_mosaic:float=0, transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / phase / "img" / split
        self.ann_dir = self.root_dir / phase / "ann" / split
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)
        self.use_mosaic = use_mosaic

        if self.use_mosaic > 0:
            self.mosaic = Mosaic(image_size, image_size)

    def __len__(self):
        return len(self.img_list)


    def load_mosaic(self, index:int):
        indexes = [index] + [random.randint(0, len(self.img_list) - 1) for _ in range(3)]
        images_list = []
        masks_list = []

        for index in indexes:
            img = np.array(Image.open(self.img_dir / self.img_list[index]))
            ann_path = self.ann_dir / f"{Path(self.img_list[index]).stem}.png"
            ann = np.array(Image.open(ann_path))
            images_list.append(img)
            masks_list.append(ann)

        result_image, result_mask = self.mosaic(
            images_list, 
            masks_list
        )
            
        return result_image, result_mask
    
    def __getitem__(self, idx):

        if self.use_mosaic > 0:
            if random.random() < self.use_mosaic:
                img, ann = self.load_mosaic(idx)
                ori_size = img.shape[:2]
                if self.transform:
                    augmented = self.transform(image=img, mask=ann)
                    img = augmented["image"]
                    ann = augmented["mask"]
                return img, ann, ori_size
        
        img = np.array(Image.open(self.img_dir / self.img_list[idx]))
        ori_size = img.shape[:2]
        ann_path = self.ann_dir / f"{Path(self.img_list[idx]).stem}.png"
        ann = np.array(Image.open(ann_path))

        if self.transform:
            augmented = self.transform(image=img, mask=ann)
            img = augmented["image"]
            ann = augmented["mask"]

        return img, ann, ori_size


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir="data", phase="warmup", batch_size: int = 8, image_size:int=512, use_mosaic:float=0):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_mosaic = use_mosaic

    def setup(self, stage: str):
        self.train = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="train",
            use_mosaic=self.use_mosaic,
            image_size=self.image_size,
            transform=get_augmentations("train", image_size=self.image_size),
        )
        self.valid = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="valid",
            image_size=self.image_size,
            transform=get_augmentations("valid", image_size=self.image_size)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False, num_workers=4)


if __name__ == "__main__":
    ds = SegDataset()
    img, ann, _ = ds[10]
    print("Done!")
