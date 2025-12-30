import torch.nn as nn
import torchvision.models as models

class FlowerCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Önceden eğitilmiş ResNet18
        self.model = models.resnet18(pretrained=True)

        # Sadece son katmanı eğiteceğiz
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)
