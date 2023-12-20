import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from infection.metrics import SMAPIoUMetric
from infection.losses import get_loss


class SegModel(pl.LightningModule):
    def __init__(
            self, 
            model_name:str="fcn_r50_baseline", 
            loss_configs:dict=None,
            optimizer_configs:dict=None,
            scheduler_configs:dict=None,
        ):
        super(SegModel, self).__init__()

        self.loss_configs = loss_configs
        self.optimizer_configs = optimizer_configs
        self.scheduler_configs = scheduler_configs

        if model_name == "fcn_r50_baseline":
            self.net = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        else:
            # example: model_name = "unetplusplus.tf_efficientnetv2_b0"
            #  model names from Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN,

            # encoder name from https://smp.readthedocs.io/en/latest/encoders_timm.html
            
            arch_name, encoder_name = model_name.split(".")

            self.net = smp.create_model(
                arch=arch_name,
                encoder_name=encoder_name,
                in_channels=3,
                encoder_weights = "imagenet",
                classes=2,
            )
        
        if self.loss_configs is not None:
            self.criterion = get_loss(
                self.loss_configs
            )
        self.evaluator = SMAPIoUMetric()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask, _ = batch
        img = img.float()
        mask = mask.long()

        out = self.forward(img)
        
        if isinstance(out, dict):
            out = out["out"]

        loss, loss_dict = self.criterion(out, mask)

        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        img, mask, _ = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        
        if isinstance(out, dict):
            out = out["out"]
        
        loss, loss_dict = self.criterion(out, mask)

        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = preds.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        self.evaluator.process(input={"pred": preds, "gt": mask})
        
        self.log_dict(
            loss_dict, prog_bar=True, sync_dist=True
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.evaluator.evaluate(0)
        self.log(
            f"val_high_vegetation_IoU",
            metrics["high_vegetation__IoU"],
            sync_dist=True,
        )
        self.log(f"val_mIoU", metrics["mIoU"], sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.net.parameters(), 
            lr=self.optimizer_configs.get('lr', 1e-3)
        )
