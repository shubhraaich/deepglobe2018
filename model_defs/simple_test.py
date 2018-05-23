import torch
from resnet_oneway import resnet50_oneway
from depth_to_space import DepthToSpace


model = resnet50_oneway(num_classes=2);
a = torch.autograd.Variable(torch.rand(8,3,224,224));
b = model(a);

print(b.size())
