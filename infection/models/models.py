import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from infection.metrics import SMAPIoUMetric
from infection.losses import get_loss
from transformers import (
    SegformerForSemanticSegmentation,
    MaskFormerForInstanceSegmentation,
    Mask2FormerForUniversalSegmentation
)
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput
from transformers import MaskFormerImageProcessor


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
        self.model_name = model_name

        id2label = {
            0: "background",
            1: "vegetation",
        }
        label2id = {v: k for k, v in id2label.items()}

        if model_name == "fcn_r50_baseline":
            self.net = torchvision.models.segmentation.fcn_resnet50(num_classes=2)
        elif model_name == 'segformer':
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=2, 
                id2label=id2label, 
                label2id=label2id,
            )
        elif model_name == 'maskformer':
            # Replace the head of the pre-trained model
            self.net = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-base-ade",
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )

             # Create a preprocessor
            self.processor = MaskFormerImageProcessor(
                ignore_index=0, reduce_labels=False, 
                do_resize=False, do_rescale=False, 
                do_normalize=False
            )

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

    def forward_logits(self, x):
        out = self.net(x)
        if isinstance(out, SemanticSegmenterOutput):
            logits = out.logits
            with torch.no_grad():
                out = nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        elif isinstance(out, MaskFormerForInstanceSegmentationOutput):
            out = self.processor.post_process_semantic_segmentation(
                out
            )
            out = torch.stack(out, dim=0).long()
        elif isinstance(out, dict):
            out = out["out"]
        elif isinstance(out, torch.Tensor):
            return out
        else:
            raise NotImplementedError
        return out


    def forward_with_loss(self, img, mask):
        if self.model_name == 'segformer':
            outputs = self.net(
                pixel_values=img, 
                labels=mask
            )
            loss, logits = outputs.loss, outputs.logits
            with torch.no_grad():
                out = nn.functional.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss_dict = {"T": loss.item()}
        elif self.model_name == 'maskformer':
            # one hot encoding mask
            onehot_mask = torch.nn.functional.one_hot(mask, num_classes=2).permute(0, 3, 1, 2).float()
            class_labels=torch.Tensor([0, 1]).unsqueeze(0).repeat(img.shape[0], 1).long().cuda()
            outputs = self.net(
                img,
                class_labels=class_labels,
                mask_labels=onehot_mask
            )
            loss, out = outputs.loss, outputs
            loss_dict = {"T": loss.item()}
        else:
            out = self.forward(img)
            loss, loss_dict = self.criterion(out, mask)
        return out, loss, loss_dict

    def training_step(self, batch, batch_nb):
        img, mask, _ = batch
        img = img.float()
        mask = mask.long()
        _, loss, loss_dict = self.forward_with_loss(img, mask)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        img, mask, _ = batch
        img = img.float()
        mask = mask.long()
        out, loss, loss_dict = self.forward_with_loss(img, mask)

        if isinstance(out, MaskFormerForInstanceSegmentationOutput):
            target_sizes = [(mask.shape[1], mask.shape[2]) for i in range(mask.shape[0])]
            out = self.processor.post_process_semantic_segmentation(
                out, target_sizes=target_sizes
            )
            preds = torch.stack(out, dim=0)
        else:
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
