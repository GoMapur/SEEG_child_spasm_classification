 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
import torchvision.models as models

"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # print(self.conv1)
        # print(self.conv2_x)
        # print(self.conv3_x)
        # print(self.conv4_x)
        # print(self.conv5_x)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        output = self.conv1(x)
        # print(output.shape)
        output = self.conv2_x(output)
        # print(output.shape)
        output = self.conv3_x(output)
        # print(output.shape)
        output = self.conv4_x(output)
        # print(output.shape)
        output = self.conv5_x(output)
        # print(output.shape)
        output = self.avg_pool(output)
        # print(output.shape)
        output = output.reshape(output.size(0), -1)
        # print(output.shape)
        output = self.fc(output)
        # print(output.shape)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


class Neural_CNN(torch.nn.Module):
    def __init__(self, num_classes=2, num_extra_features=0, 
                 dropout_p=0.0,  # <--- 0.5 dropout probability by default
                 freeze_layers=False):
        super(Neural_CNN, self).__init__()

        # Use ResNet-34 by default
        self.cnn = resnet34()

        # Optionally freeze all layers except the final FC:
        if freeze_layers:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Replace the ResNet's final FC layer with a smaller linear layer
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 32),
            nn.Dropout(dropout_p),  # <--- Dropout inside the CNN's final block
            nn.ReLU(inplace=True)
        )
        # Ensure these new layers are trainable
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

        # Another BN + FC stack
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.LeakyReLU()

        # Possibly more layers
        self.fc = nn.Linear(32 + num_extra_features, 32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout_p)  # <--- dropout to use after FCs

        if num_classes < 3:
            self.fc_out = nn.Linear(16, 1)
            self.final_ac = nn.Sigmoid()
        else:
            self.fc_out = nn.Linear(16, num_classes)
            self.final_ac = nn.Softmax(dim=-1)

    def forward(self, x, additional_feature=None, return_features=False):
        """
        forward pass: x => CNN => FC => output
        """
        # x shape: (batch_size, 1, H, W) for EEG images or spectrograms
        batch = self.cnn(x[:, None, :, :])  # inserts dropout from self.cnn.fc

        # BN + ReLU
        batch = self.bn0(batch)
        batch = self.relu0(batch)

        CNN_features = batch.clone()

        # Optionally incorporate extra features
        if additional_feature is not None:
            batch = torch.cat((batch, additional_feature), dim=1)

        # FC => dropout => activation
        batch = self.dropout(self.bn(self.relu(self.fc(batch))))
        classifier_features_1 = batch.clone()

        batch = self.dropout(self.bn1(self.relu1(self.fc1(batch))))
        classifier_features_2 = batch.clone()

        # final linear => Sigmoid or Softmax
        batch = self.final_ac(self.fc_out(batch))

        if return_features:
            return batch, CNN_features, classifier_features_1, classifier_features_2
        else:
            return batch
