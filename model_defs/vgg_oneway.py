import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from depth_to_space import DepthToSpace

__all__ = [
    'VGG', 'VGG_Oneway', 'vgg16_bn', 'vgg16_bn_oneway',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def conv_nxn_with_init(in_channels, out_channels, kernel_size, stride, padding, bias):
    """nxn convolution with initialization"""
    layer_ = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     bias=bias);
    nn.init.xavier_normal(layer_.weight, gain=1.0);
    if bias :
            nn.init.constant(layer_.bias, 0.0);
    return layer_;



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_Oneway(nn.Module):

    def __init__(self, model_org, num_classes):
        super(VGG_Oneway, self).__init__()

        self.tmp = list(model_org.features.children());

        self.tmp.append(conv_nxn_with_init(in_channels=512,
                                           out_channels=32,
                                           kernel_size=1, stride=1,
                                           padding=0, bias=False));
        self.tmp.append(conv_nxn_with_init(in_channels=32,
                                           out_channels=2048,
                                           kernel_size=1, stride=1,
                                           padding=0, bias=False));
        self.tmp.append(DepthToSpace(num_split=num_classes));
        self.tmp.append(conv_nxn_with_init(in_channels=2,
                                           out_channels=2,
                                           kernel_size=15, stride=1,
                                           padding=7, bias=False));

        # add spatial dropout
        dropout_ind = [];
        count = 0;
        for i in range(len(self.tmp)) :
            if self.tmp[i].__class__.__name__ == "MaxPool2d" :
                dropout_ind.append(i+1+count);
                count += 1;

        for i in dropout_ind :
            self.tmp.insert(i, nn.Dropout2d(p=0.25));

        # model
        self.model = nn.Sequential(*self.tmp);

    def forward(self, x):
        x = self.model(x);
        return x;



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg16_bn_oneway(num_classes=2, pretrained=True) :
    """VGG16-layer model (configuration "GAS") upto 4-3 layer and GAP-
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = vgg16_bn(pretrained=pretrained)
    model = VGG_Oneway(model, num_classes);
    return model
