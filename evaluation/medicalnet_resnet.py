"""Vendored MedicalNet 3D ResNet-50 encoder -- the backbone `fid_3d_medicalnet` runs on.

Source: github.com/Tencent/MedicalNet, `models/resnet.py`, as re-expressed by
github.com/Warvito/MedicalNet-models (`medicalnet_models/models/resnet.py`) -- the same module the
CCELLA protocol loads through `torch.hub`. Only the encoder is here: the released
`resnet_50_23dataset.pth` holds no segmentation head, so its state dict maps onto this module with
nothing missing and nothing left over, which `medical_fid.load_medicalnet` asserts.

Like `fid_inception.py`, this file is a reference implementation and **must not be edited** -- the
point of copying it is that our 3D FID is the same arithmetic on the same weights as every FID
quoted against MedicalNet. The architecture, in case it matters to a reader: stride 2 at `conv1`
and at the max pool, stride 2 into `layer2`, then `layer3`/`layer4` at stride 1 with dilation 2 and
4, so the feature map is the input divided by 8 and `layer4` is 2048 channels wide.
"""

import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride,
                     padding=dilation, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class ResNet(nn.Module):
    """MedicalNet's encoder. `forward` returns the `layer4` feature map, `(N, 2048, D/8, H/8, W/8)`
    -- the representation the task-specific `conv_seg` head sits on."""

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample, dilation)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
