import torch
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from infection.metrics import SMAPIoUMetric
from infection.losses import get_loss
from transformers import (
    SegformerForSemanticSegmentation,
    MaskFormerForInstanceSegmentation,
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    MaskFormerImageProcessor
)
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
from .dinov2 import Dinov2ForSemanticSegmentation

class SegModel(pl.LightningModule):
    def __init__(
            self, 
            model_name:str="fcn_r50_baseline", 
            loss_configs:dict=None,
            optimizer_configs:dict=None,
            scheduler_configs:dict=None,
            postprocessor=None,
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
                ignore_mismatched_sizes=True,
            )

             # Create a preprocessor
            if postprocessor is None:
                self.processor = MaskFormerImageProcessor(
                    ignore_index=0, reduce_labels=False, 
                    do_resize=False, do_rescale=False, 
                    do_normalize=False
                )
            else:
                self.processor = postprocessor

        elif model_name == 'mask2former':
            self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-base-coco-panoptic",
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            # Create a preprocessor
            if postprocessor is None:
                self.processor = AutoImageProcessor.from_pretrained(
                    "facebook/mask2former-swin-base-coco-panoptic"
                )
            else:
                self.processor = postprocessor

        elif model_name.startswith('dinov2'):
            self.net = Dinov2ForSemanticSegmentation.from_pretrained(
                f"facebook/{model_name}", 
                id2label=id2label, 
                num_labels=len(id2label),
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

    def freeze_backbone(self):
        if self.model_name.startswith('dinov2'):
            self.net.freeze()
        else:
            for pname, param in self.net.named_parameters():
                if not pname.startswith("segmentation_head"):
                    param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.net(x)

    def forward_logits(self, x):
        out = self.net(x)
        if isinstance(out, SemanticSegmenterOutput):
            logits = out.logits
            with torch.no_grad():
                out = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
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


    def forward_with_loss(self, batch):
        if self.model_name == 'segformer':
            outputs = self.net(
                pixel_values=batch[0], 
                labels=batch[1]
            )
            loss, logits = outputs.loss, outputs.logits
            with torch.no_grad():
                out = F.interpolate(logits, size=batch[1].shape[-2:], mode="bilinear", align_corners=False)
            loss_dict = {"T": loss.item()}
        elif self.model_name == 'maskformer' or self.model_name == 'mask2former':
            # one hot encoding mask
            outputs = self.net(
                pixel_values=batch[0],
                mask_labels=torch.stack(batch[1], dim=0).float(),
                class_labels=torch.stack(batch[2], dim=0).long(),
            )
            loss, out = outputs.loss, outputs
            loss_dict = {"T": loss.item()}
        elif self.model_name.startswith('dinov2'):
            out = self.net(
                pixel_values=batch[0], 
            )
            loss, loss_dict = self.criterion(out, batch[1].long())
        else:
            out = self.net(batch[0])
            loss, loss_dict = self.criterion(out, batch[1].long())
        return out, loss, loss_dict

    def training_step(self, batch, batch_nb):
        _, loss, loss_dict = self.forward_with_loss(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        out, loss, loss_dict = self.forward_with_loss(batch)

        if isinstance(out, (
                MaskFormerForInstanceSegmentationOutput,
                Mask2FormerForUniversalSegmentationOutput
            )
        ):
            import pdb; pdb.set_trace()
            target_sizes = [(batch[0].shape[-2], batch[0].shape[-1]) for i in range(batch[0].shape[0])]
            out = self.processor.post_process_semantic_segmentation(
                out, target_sizes=target_sizes
            )
            preds = torch.stack(out, dim=0)
            mask = torch.cat(batch[1], dim=0)
            import pdb; pdb.set_trace()
        else:
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            mask = batch[1]

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
