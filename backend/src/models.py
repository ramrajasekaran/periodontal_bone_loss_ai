import torch
import torch.nn as nn
import torchvision.models as models


class TabularMLP(nn.Module):
    """
    Tabular-only baseline for ablation study (Experiment B).
    Accepts the same (images, tabular_features) signature but ignores images.
    """
    def __init__(self, num_classes=3, tabular_dim=6):
        super(TabularMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, images, tabular_features=None):
        if tabular_features is None:
            raise ValueError("TabularMLP requires tabular_features")
        return self.classifier(tabular_features)


class MultiModalSimpleCNN(nn.Module):
    """
    Vision-first CNN with optional auxiliary tabular refinement.
    """
    def __init__(self, num_classes=3, tabular_dim=6):
        super(MultiModalSimpleCNN, self).__init__()
        self.tabular_dim = tabular_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.visual_classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )
        self.tabular_classifier = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, images, tabular_features=None):
        x = self.features(images)
        x = x.view(x.size(0), -1) # Flatten
        visual_logits = self.visual_classifier(x)
        if self.tabular_dim > 0 and tabular_features is not None and torch.any(tabular_features != 0):
            tabular_logits = self.tabular_classifier(tabular_features)
            return visual_logits + 0.3 * tabular_logits
        return visual_logits

class MultiModalResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, tabular_dim=6):
        super(MultiModalResNet50, self).__init__()
        self.tabular_dim = tabular_dim
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layer3 and layer4 for deep feature representation
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = int(self.backbone.fc.in_features)
        setattr(self.backbone, 'fc', nn.Identity())
        
        self.visual_classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.tabular_classifier = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, images, tabular_features=None):
        x_img = self.backbone(images)
        visual_logits = self.visual_classifier(x_img)
        if self.tabular_dim > 0 and tabular_features is not None and torch.any(tabular_features != 0):
            tabular_logits = self.tabular_classifier(tabular_features)
            return visual_logits + 0.3 * tabular_logits
        return visual_logits

class MultiModalEfficientNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, tabular_dim=6):
        super(MultiModalEfficientNet, self).__init__()
        self.tabular_dim = tabular_dim
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze from block 4 onwards
        for param in self.backbone.features[4:].parameters():
            param.requires_grad = True
            
        classifier_linear = self.backbone.classifier[1]
        num_ftrs = int(getattr(classifier_linear, 'in_features', 1280))
        setattr(self.backbone, 'classifier', nn.Identity())
        
        self.visual_classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.tabular_classifier = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, images, tabular_features=None):
        x_img = self.backbone(images)
        visual_logits = self.visual_classifier(x_img)
        if self.tabular_dim > 0 and tabular_features is not None and torch.any(tabular_features != 0):
            tabular_logits = self.tabular_classifier(tabular_features)
            return visual_logits + 0.3 * tabular_logits
        return visual_logits

def get_model(model_name, num_classes=3):
    if model_name.lower() == 'cnn':
        return MultiModalSimpleCNN(num_classes=num_classes, tabular_dim=6)
    elif model_name.lower() == 'resnet50':
        return MultiModalResNet50(num_classes=num_classes, tabular_dim=6)
    elif model_name.lower() == 'efficientnet':
        return MultiModalEfficientNet(num_classes=num_classes, tabular_dim=6)
    elif model_name.lower() == 'tabular_mlp':
        return TabularMLP(num_classes=num_classes, tabular_dim=6)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
