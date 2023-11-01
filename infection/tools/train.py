import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import SegDataModule
from models import SegModel
from lightning import seed_everything

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='fcn_r50_baseline')
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--log_every_n_steps", type=int, default=10)
parser.add_argument("--root_dir", type=str, default='/home/mpham/workspace/ucc-ai-quest-2023/data')

seed_everything(2023)

def main(args):
    datamodule = SegDataModule(
        root_dir=args.root_dir,
        batch_size=args.batch_size
    )
    model = SegModel(
        model_name=args.model_name,
        loss_configs={
            'label_smoothing': args.label_smoothing,
        },
        optimizer_configs={
            'lr': 1e-3,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_high_vegetation_IoU",
        mode="max",
        filename="{epoch}-{val_loss:.2f}-{val_high_vegetation_IoU:.2f}-{val_mIoU:.2f}",
        save_top_k=3,
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback], log_every_n_steps=args.log_every_n_steps)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
