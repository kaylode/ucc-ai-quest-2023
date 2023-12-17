import os
import random
import numpy as np
import lightning.pytorch as pl
from infection.augmentations import get_augmentations
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    def __init__(self, root_dir="data", phase="warmup", split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / phase / "img" / split
        self.ann_dir = self.root_dir / phase / "ann" / split
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)


    def load_mosaic(self, index:int):
        indexes = [index] + [random.randint(0, len(self.fns) - 1) for _ in range(3)]
        images_list = []
        masks_list = []

        for index in indexes:
            img_path, label_path = self.fns[index]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            mask = self._load_mask(label_path)
            images_list.append(img)
            masks_list.append(mask)

        result_image, result_mask = self.mosaic(
            images_list, 
            masks_list)
            
        return result_image, result_mask
    
    def __getitem__(self, idx):
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
    def __init__(self, root_dir="data", phase="warmup", batch_size: int = 8, image_size:int=512):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage: str):
        self.train = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="train",
            transform=get_augmentations("train", image_size=self.image_size),
        )
        self.valid = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="valid",
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
