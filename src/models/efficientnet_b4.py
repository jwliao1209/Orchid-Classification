import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, convnext_base
from .base import BaseModule
from .convnext import convnext_base


class EfficientNet_b4(BaseModule):
    def __init__(self, num_classes=219):
        super(EfficientNet_b4, self).__init__()
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)
        model_child = list(model.children())
        self.feature_extr = nn.Sequential(*model_child[:-2])
        self.avgpool = model_child[-2]
        self.classifier = model_child[-1]

    def forward(self, inputs, use_mc_loss=False, *args, **kwargs):
        feature = self.feature_extr(inputs)
        x = self.avgpool(feature)
        x = x.view(x.size(0), -1)
        outputs = self.classifier(x)

        if use_mc_loss:
            return outputs, feature

        else:
            return outputs


class ConvNeXt_B(BaseModule):
    def __init__(self, num_classes=219):
        super(ConvNeXt_B, self).__init__()
        self.model = convnext_base(pretrained=True, in_22k=True)
        #self.model = convnext_base(pretrained=True)
        #model = timm.create_model('convnext_base', pretrained=True)
        print(self.model)
        self.model.head.fc = torch.nn.Linear(
            self.model.head.fc.in_features, num_classes)


    def forward(self, inputs, *args, **kwargs):
        outputs = self.model(inputs)

        return outputs
        

if __name__ == '__main__':
    model = ConvNeXt_B(219)
    #print(model)
