import os
import os.path as osp
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from infection.datasets import SegDataModule
from infection.models import SegModel
from infection.callbacks import WandbCallback, VisualizerCallback
from lightning import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

seed_everything(2023)

@hydra.main(version_base=None)
def main(args: DictConfig):

    datamodule = SegDataModule(
        root_dir=args.data.root_dir,
        batch_size=args.data.batch_size,
        image_size=args.data.image_size
    )

    model = SegModel(
        model_name=args.model.model_name,
        loss_configs=args.loss,
        optimizer_configs=args.optimizer
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(args.trainer.save_dir, 'checkpoints'),
        monitor="val_high_vegetation_IoU",
        mode="max",
        filename="{epoch}-{val_loss:.2f}-{val_high_vegetation_IoU:.2f}-{val_mIoU:.2f}",
        save_top_k=3,
    )

    # os.makedirs(args.trainer.save_dir, exist_ok=True)
    # wandb_callback = WandbCallback(
    #     username = args.logger.wandb.username,
    #     project_name=args.logger.wandb.project_name,
    #     group_name=args.logger.wandb.group_name,
    #     save_dir=args.trainer.save_dir,
    #     config_dict=args
    # )

    visualizer_callback = VisualizerCallback(
        save_dir=args.trainer.save_dir,
    )

    # Save configs
    with open(osp.join(args.trainer.save_dir, "pipeline.yaml"), "w") as f:
        OmegaConf.save(config=args, f=f)

    trainer = pl.Trainer(
        max_epochs=args.trainer.max_epochs, 
        callbacks=[visualizer_callback, checkpoint_callback], 
        log_every_n_steps=args.trainer.log_every_n_steps,
        default_root_dir=args.trainer.save_dir,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
