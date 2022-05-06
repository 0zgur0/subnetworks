import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from masksembles.torch import Masksembles1D, Masksembles2D

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn',
]

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, features,  n, scale, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n = n
        self.scale = scale

        self.classifier = nn.Sequential(nn.Linear(512, 512),
                                        Masksembles1D(512, self.n, self.scale).float(),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output = self.classifier.forward(x.float())

        return output

    def get_layer(self, name):
        return getattr(self, name)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, n, scale):
    layers = []
    in_channels = 3
    for ix, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            bn_layer = nn.BatchNorm2d(v)
            maskEnsemble_layer = Masksembles2D(v, n, scale)

            layers += [conv2d, bn_layer, nn.ReLU(inplace=True), maskEnsemble_layer]

            in_channels = v
    return nn.Sequential(*layers)




def vgg_maskEnsemble(cfg, n=4, scale=2.0, num_classes=10, pretrained=False,  **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:

        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg=cfg, n=n, scale=scale, **kwargs), n=n, scale=scale, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model



def test():
    net = vgg_maskEnsemble(cfg['VGG11'], n=int(4), scale=float(2.0))
    print(net)
    x = torch.randn(16,3,32,32)
    y = net(x)
    print(y.size())

# test()
