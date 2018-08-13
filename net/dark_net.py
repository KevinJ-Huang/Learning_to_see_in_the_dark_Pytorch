import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np




class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(32, 12, kernel_size=1, stride=1, padding=0, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, target=None):
        conv1 = F.leaky_relu(self.conv1_2(F.leaky_relu(self.conv1_1(x),0.2)),0.2)
        conv2 = F.leaky_relu(self.conv2_2(F.leaky_relu(self.conv2_1(self.maxpool(conv1)),0.2)),0.2)
        conv3 = F.leaky_relu(self.conv3_2(F.leaky_relu(self.conv3_1(self.maxpool(conv2)), 0.2)), 0.2)
        conv4 = F.leaky_relu(self.conv4_2(F.leaky_relu(self.conv4_1(self.maxpool(conv3)), 0.2)), 0.2)
        conv5 = F.leaky_relu(self.conv5_2(F.leaky_relu(self.conv5_1(self.maxpool(conv4)), 0.2)), 0.2)
        conv6 = F.leaky_relu(self.conv6_2(F.leaky_relu(self.conv6_1(self.upsample_and_concat(conv5, conv4, 256, 512)), 0.2)), 0.2)
        conv7 = F.leaky_relu(self.conv7_2(F.leaky_relu(self.conv7_1(self.upsample_and_concat(conv6, conv3, 128, 256)), 0.2)), 0.2)
        conv8 = F.leaky_relu(self.conv8_2(F.leaky_relu(self.conv8_1(self.upsample_and_concat(conv7, conv2, 64, 128)), 0.2)), 0.2)
        conv9 = F.leaky_relu(self.conv9_2(F.leaky_relu(self.conv9_1(self.upsample_and_concat(conv8, conv1, 32, 64)), 0.2)), 0.2)
        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10,2)
        # out = self.space_to_depth(conv10,0.5)
        if target is  not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)


    def upsample_and_concat(self, x1, x2, output_channels, in_channels):
        deconv = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=False)
        deconv = deconv.cuda()
        # deconv = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2, padding=1, bias=False)
        deconv_out = deconv(x1, output_size = x2.shape)
        output = torch.cat((deconv_out, x2), 1)
        # output.reshape([None,output_channels * 2 ,None, None])
        return output


    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result


