import torch
import torch.nn as nn
import torch.nn.init as init
import os
import sys
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(AttentiveStatisticsPooling, self).__init__()

        #self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0)==1:
            attentions = F.softmax(torch.tanh(weights),dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1), weighted
        else:
            noise = 1e-5*torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

            representations = torch.cat((avg_repr,std_repr),1)

            return representations, weighted
            
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out

class ResBlock1(nn.Module):

    def __init__(self, mode=1):
        super(ResBlock1, self).__init__()
        self.block = FSELayer1(64, 16, mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.block(res)
        x += res
        x = self.relu(x)
        return x

class ResBlock2(nn.Module):

    def __init__(self, mode=1):
        super(ResBlock2, self).__init__()
        self.block = FSELayer2(128, 8, mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.block(res)
        x += res
        x = self.relu(x)
        return x

class ResBlock3(nn.Module):

    def __init__(self, mode=1):
        super(ResBlock3, self).__init__()
        self.block = FSELayer3(256, 4, mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.block(res)
        x += res
        x = self.relu(x)
        return x

class ResBlock4(nn.Module):

    def __init__(self, mode=1):
        super(ResBlock4, self).__init__()
        self.block = FSELayer4(512, 2, mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.block(res)
        x += res
        x = self.relu(x)
        return x
     
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
# RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  # '28': [[3, 4, 6, 3], PreActBlock],
                  # '34': [[3, 4, 6, 3], PreActBlock],
                  # '50': [[3, 4, 6, 3], PreActBottleneck],
                  # '101': [[3, 4, 23, 3], PreActBottleneck]
                  # }
RESNET_CONFIGS = {'18': [[2, 2, 2, 2], SEBasicBlock]}

class FSELayer1(nn.Module):
    def __init__(self, channel, reduction=16, mode=1):
        super(FSELayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(84, 84 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(84 // reduction, 84, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        self.mode = mode

    def forward(self, x):
        b, c, _, _ = x.size()
        z = torch.mean(x, dim=(1,3), keepdim=False)
        z = self.fc(z)
        z = z.view(b, 1, -1, 1)
        if self.mode == 1:
            x = x*z
        else:
            z = x*z
        m = torch.mean(x, dim=2, keepdim=False)
        m = self.conv(m).view(b, c, 1, -1)
        if self.mode == 1:
            x = x * m
        elif self.mode == 2:
            x = x * m * z
        else:
            x = (x * m) + z
        
        return x


class FSELayer2(nn.Module):
    def __init__(self, channel, reduction=16, mode=1):
        super(FSELayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(42, 42 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(42 // reduction, 42, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.mode = mode

    def forward(self, x):
        b, c, _, _ = x.size()
        z = torch.mean(x, dim=(1,3), keepdim=False)
        z = self.fc(z)
        z = z.view(b, 1, -1, 1)
        if self.mode == 1:
            x = x*z
        else:
            z = x*z
        m = torch.mean(x, dim=2, keepdim=False)
        m = self.conv(m).view(b, c, 1, -1)
        if self.mode == 1:
            x = x * m
        elif self.mode == 2:
            x = x * m * z
        else:
            x = (x * m) + z
        
        return x

class FSELayer3(nn.Module):
    def __init__(self, channel, reduction=8, mode=1):
        super(FSELayer3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(21, 21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(21 // reduction, 21, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        self.mode = mode

    def forward(self, x):
        b, c, _, _ = x.size()
        z = torch.mean(x, dim=(1,3), keepdim=False)
        z = self.fc(z)
        z = z.view(b, 1, -1, 1)
        if self.mode == 1:
            x = x*z
        else:
            z = x*z
        m = torch.mean(x, dim=2, keepdim=False)
        m = self.conv(m).view(b, c, 1, -1)
        if self.mode == 1:
            x = x * m
        elif self.mode == 2:
            x = x * m * z
        else:
            x = (x * m) + z
        
        return x

class FSELayer4(nn.Module):
    def __init__(self, channel, reduction=4, mode=1):
        super(FSELayer4, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(11, 11 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(11 // reduction, 11, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        self.mode = mode

    def forward(self, x):
        b, c, _, _ = x.size()
        z = torch.mean(x, dim=(1,3), keepdim=False)
        z = self.fc(z)
        z = z.view(b, 1, -1, 1)
        if self.mode == 1:
            x = x*z
        else:
            z = x*z
        m = torch.mean(x, dim=2, keepdim=False)
        m = self.conv(m).view(b, c, 1, -1)
        if self.mode == 1:
            x = x * m
        elif self.mode == 2:
            x = x * m * z
        else:
            x = (x * m) + z
        
        return x


class FTANet(nn.Module):

    def __init__(
                    self, 
                    resnet_type='18',
                    mode=3
                ):
        ''' 
            
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(FTANet, self).__init__()      
        # resnet
        self.in_planes = 16
        enc_dim = 256
        layers, block = RESNET_CONFIGS[resnet_type]
        self._norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.ft1 = ResBlock1(mode=mode)
        self.ft2 = ResBlock2(mode=mode)
        self.ft3 = ResBlock3(mode=mode)
        self.ft4 = ResBlock4(mode=mode)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(6, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(6, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.fc = nn.Sequential(nn.Linear(256 * 2, enc_dim),
                                nn.Linear(enc_dim, 1))
        self.bnf = nn.BatchNorm2d(2)

        self.initialize_params()
        self.attention = AttentiveStatisticsPooling(256)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    

    def forward(self, inputs):
        # print('input: ', inputs.size())
        cspecs = inputs[:,:,1:]

        x = self.conv1(cspecs)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.ft1(x)

        x = self.layer2(x)
        x = self.ft2(x)

        x = self.layer3(x)
        x = self.ft3(x)

        x = self.layer4(x)
        x = self.ft4(x)

        x = self.activation(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = self.activation(x).squeeze(2)

        stats, weghited = self.attention(x.permute(0, 2, 1).contiguous())
        mu = self.fc(stats)
        
        return mu, weghited


if __name__ == '__main__':
    pass