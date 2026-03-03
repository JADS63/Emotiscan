# ResNet18 adapté 1 canal (grayscale), 7 classes émotions
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def set_seed(seed=42):
    """Fixe la graine pour reproductibilité (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="checkpoint.pth"):
    """Sauvegarde state_dict + métadonnées (epoch, optimizer, etc.)."""
    torch.save(state, filename)


class EmotiScanResNet18(nn.Module):
    """ResNet18 : entrée 1 canal, pas de maxpool initial, classif 7 classes. Option freeze_layers et couche cachée."""

    def __init__(
        self, 
        num_classes=7, 
        in_channels=1, 
        dropout=0.3, 
        pretrained=True,
        freeze_layers=0,
        hidden_size=0
    ):
        super(EmotiScanResNet18, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)
        # Première couche : 1 canal (grayscale) au lieu de 3
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        if pretrained and in_channels == 1:
            with torch.no_grad():
                orig_w = original_conv.weight.data.mean(dim=1, keepdim=True)
                self.model.conv1.weight.data = nn.functional.interpolate(
                    orig_w, size=(3, 3), mode='bilinear', align_corners=False
                )
        # Pas de maxpool pour petites images (96x96)
        self.model.maxpool = nn.Identity()
        self._freeze_layers = freeze_layers
        self._apply_freeze(freeze_layers)
        
        num_features = self.model.fc.in_features
        self.dropout = nn.Dropout(p=dropout)
        self._dropout_rate = dropout
        self._hidden_size = hidden_size
        
        if hidden_size > 0:
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            self.model.fc = nn.Linear(num_features, num_classes)

    def _apply_freeze(self, freeze_layers):
        """Gèle les premières couches (conv1, bn1, layer1..layer4) selon freeze_layers (0-4)."""
        layers_to_freeze = []
        if freeze_layers >= 1:
            layers_to_freeze.extend([self.model.conv1, self.model.bn1, self.model.layer1])
        if freeze_layers >= 2:
            layers_to_freeze.append(self.model.layer2)
        if freeze_layers >= 3:
            layers_to_freeze.append(self.model.layer3)
        if freeze_layers >= 4:
            layers_to_freeze.append(self.model.layer4)
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self._freeze_layers = 0
    
    def freeze_backbone(self):
        self._apply_freeze(4)
        self._freeze_layers = 4

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        return x
    
    def set_dropout(self, p: float):
        self._dropout_rate = p
        self.dropout.p = p
        if self._hidden_size > 0:
            self.model.fc[2].p = p / 2
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(name="resnet18", num_classes=7, in_channels=1, dropout=0.3, pretrained=True, freeze_layers=0, hidden_size=0):
    """Crée une instance EmotiScanResNet18 (argument name ignoré)."""
    return EmotiScanResNet18(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=dropout,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        hidden_size=hidden_size
    )