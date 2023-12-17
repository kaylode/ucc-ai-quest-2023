from typing import Dict, Any

import matplotlib.patches as mpatches
import matplotlib as mpl
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import torch
from torchvision.transforms import functional as TFF

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from infection.utilities.visualization import Visualizer, color_list


def write_image(value, savepath):
    if isinstance(value, go.Figure):
        value.write_image(savepath + ".png")
    if isinstance(value, mpl.figure.Figure):
        value.savefig(savepath)

class VisualizerCallback(Callback):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, save_dir:str, **kwargs) -> None:
        super().__init__()

        self.save_dir = osp.join(save_dir, 'figures')
        os.makedirs(self.save_dir, exist_ok=True)
        self.visualizer = Visualizer()
        self.classnames = ['background', 'vegetation']

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Sanitycheck before starting. Run only when debug=True
        """
        iters = trainer.global_step
        trainloader = trainer.datamodule.train_dataloader()
        valloader = trainer.datamodule.val_dataloader()
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))

        self.visualize_gt(train_batch, val_batch, iters, self.classnames)

    def visualize_gt(self, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check
        """

        images = train_batch[0]
        masks = train_batch[1]

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)

        # segmentation color legends
        patches = [
            mpatches.Patch(color=np.array(color_list[i][::-1]), label=classnames[i])
            for i in range(len(classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        write_image(fig, osp.join(self.save_dir, f"train_gt_{iters}"))

        # Validation
        images = val_batch[0]
        masks = val_batch[1]

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        write_image(fig, osp.join(self.save_dir, f"val_gt_{iters}"))

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.params = {}
        self.params['last_batch'] = batch

    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """
        iters = trainer.global_step
        last_batch = self.params['last_batch']
        model = pl_module.net

        # Vizualize model predictions
        model.eval()

        images = last_batch[0]
        masks = last_batch[1]
        preds = []
        for img in images:
            out = model(img.float().unsqueeze(dim=0))
            if isinstance(out, dict):
                out = out["out"]
            pred = torch.argmax(out, dim=1)
            pred = pred.detach().cpu().numpy().squeeze()
            preds.append(pred)

        batch = []
        for idx, (inputs, mask, pred) in enumerate(zip(images, masks, preds)):
            img_show = self.visualizer.denormalize(inputs.cpu())
            decode_mask = self.visualizer.decode_segmap(mask.cpu().numpy())
            decode_pred = self.visualizer.decode_segmap(pred)
            img_cam = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            decode_pred = TFF.to_tensor(decode_pred / 255.0)
            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.title("Raw image - Prediction - Ground Truth")
        plt.imshow(grid_img)

        # segmentation color legends
        patches = [
            mpatches.Patch(color=np.array(color_list[i][::-1]), label=self.classnames[i])
            for i in range(len(self.classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(self.classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        write_image(fig, osp.join(self.save_dir, f"pred_{iters}"))

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()
