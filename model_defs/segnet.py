import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from depth_to_space import DepthToSpace

__all__ = [
    'VGG', 'SegNet',
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


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class SimpleBlock(nn.Module):

    def __init__(self, inplanes, planes, init_params=True):
        super(SimpleBlock, self).__init__()
        self.conv = conv3x3(inplanes, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if init_params :
            nn.init.xavier_normal(self.conv.weight);
            nn.init.constant(self.bn.weight, 1.0);
            nn.init.constant(self.bn.bias, 0.0);

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)));


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


class SegNet(nn.Module):

    def __init__(self, model_org, num_classes):
        super(SegNet, self).__init__()

        self.num_classes = num_classes;
        self.tmp = list(model_org.features.children());

        max_pool_ids = [];
        for i in range(len(self.tmp)) :
            if self.tmp[i].__class__.__name__ == "MaxPool2d" :
                self.tmp[i] = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True);
                max_pool_ids.append(i);


        id_ = -1;
        self.conv_layer1, id_ = self._make_layer_conv(id_);
        self.conv_layer2, id_ = self._make_layer_conv(id_);
        self.conv_layer3, id_ = self._make_layer_conv(id_);
        self.conv_layer4, id_ = self._make_layer_conv(id_);
        self.conv_layer5, id_ = self._make_layer_conv(id_);

        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True);
        self.maxunpool = nn.MaxUnpool2d(2, 2);

        self.deconv_layer5 = self._make_layer_deconv(SimpleBlock, 512, 512, 3 );
        self.deconv_layer4 = self._make_layer_deconv(SimpleBlock, 512, 256, 3 );
        self.deconv_layer3 = self._make_layer_deconv(SimpleBlock, 256, 128, 3 );
        self.deconv_layer2 = self._make_layer_deconv(SimpleBlock, 128, 64, 2 );
        self.deconv_layer1 = self._make_layer_final(SimpleBlock, 64, 2 );


    def _make_layer_conv(self, id_) :
        layers = [];
        for i in range(id_+1, len(self.tmp)) :
            if self.tmp[i].__class__.__name__ == "MaxPool2d" :
                break;
            layers.append(self.tmp[i]);

        return nn.Sequential(*layers), i;


    def _make_layer_deconv(self, block, inplanes, outplanes, num_layers):
        layers = []
        for i in range(1, num_layers) :
            layers.append(block(inplanes, inplanes));
        layers.append(block(inplanes, outplanes));

        return nn.Sequential(*layers)


    def _make_layer_final(self, block, planes, num_layers):
        layers = []
        for i in range(1, num_layers) :
            layers.append(block(planes, planes));

        layers.append(conv3x3(planes, self.num_classes) );

        return nn.Sequential(*layers);


    def forward(self, x):
        x = self.conv_layer1(x);
        x, indices_1 = self.maxpool(x);
        #print(x.size());
        x = self.conv_layer2(x);
        x, indices_2 = self.maxpool(x);
        #print(x.size());
        x = self.conv_layer3(x);
        x, indices_3 = self.maxpool(x);
        #print(x.size());
        x = self.conv_layer4(x);
        x, indices_4 = self.maxpool(x);
        #print(x.size());
        x = self.conv_layer5(x);
        x, indices_5 = self.maxpool(x);
        #print(x.size());

        x = self.maxunpool(x, indices_5);
        x = self.deconv_layer5(x);
        #print(x.size());
        x = self.maxunpool(x, indices_4);
        x = self.deconv_layer4(x);
        #print(x.size());
        x = self.maxunpool(x, indices_3);
        x = self.deconv_layer3(x);
        #print(x.size());
        x = self.maxunpool(x, indices_2);
        x = self.deconv_layer2(x);
        #print(x.size());
        x = self.maxunpool(x, indices_1);
        x = self.deconv_layer1(x);
        #print(x.size());

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


def segnet(num_classes=2, pretrained=True) :
    """VGG16-layer model (configuration "GAS") upto 4-3 layer and GAP-
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = vgg16_bn(pretrained=pretrained);
    model = SegNet(model, num_classes);
    return model
