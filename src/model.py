import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=3, freeze_backbone=False):
    """DenseNet121 with ImageNet weights, classifier replaced for num_classes.

    If freeze_backbone=True, only the new classifier layer trains;
    all other DenseNet weights stay at their pretrained ImageNet values.
    """
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    return model


def count_trainable_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total