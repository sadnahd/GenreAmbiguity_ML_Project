import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

from transformers import ViTModel

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace classifier for 7 classes
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 7)
        )

        # Unfreeze the last two residual stages
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Unfreeze the classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class UncertAwareResNet(nn.Module):
    def __init__(self):
        super(UncertAwareResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        #Two outputs now: logits and log sigma

        self.logits_head = nn.Linear(512, 7)
        self.log_sigma_head = nn.Linear(512, 7)

        # Unfreeze the last two residual stages
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Unfreeze the classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        for param in self.logits_head.parameters():
            param.requires_grad = True
        
        for param in self.log_sigma_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)

        logits = self.logits_head(features)
        log_sigma = self.log_sigma_head(features).clamp(-5, 3)
        sigma = torch.exp(log_sigma)

        return logits, sigma

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class UncertaintyAwareViT(nn.Module):
    def __init__(self, model_name, num_labels, id2label=None, label2id=None):
        super().__init__()

        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # f_i in eq. 10
        self.logits_head = nn.Linear(512, num_labels)

        # log sigma_i, one uncertainty value per class-logit
        self.log_sigma_head = nn.Linear(512, num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0]

        h = self.head(cls)

        logits = self.logits_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5, 3)

        sigma = torch.exp(log_sigma)

        return logits, sigma