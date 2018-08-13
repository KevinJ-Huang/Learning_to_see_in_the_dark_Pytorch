import torch
import torch.nn as nn
import torch.nn.functional as functional
import math


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.2)
    )


class upsample(nn.Module ):
    def __init__(self, output_channels, in_channels):
        super(upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(output_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False)
    def forward(self, x1, x2):
        deconv_out = self.deconv(x1, output_size = x2.shape)
        return deconv_out


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.conv1   = add_conv_stage(4, 32)
        self.conv2   = add_conv_stage(32, 64)
        self.conv3   = add_conv_stage(64, 128)
        self.conv4   = add_conv_stage(128, 256)
        self.conv5   = add_conv_stage(256, 512)

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128,  64)
        self.conv1m = add_conv_stage( 64,  32)

        self.conv0  = nn.Sequential(
            nn.Conv2d(32, 12, 1, 1, 0)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128,  64)
        self.upsample21 = upsample(64 ,  32)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, target = None):
        conv1_out = self.conv1(x)
        #return self.upsample21(conv1_out)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out,conv4_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)

        conv4m_out_ = torch.cat((self.upsample43(conv4m_out,conv3_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out,conv2_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out,conv1_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        conv0_out = self.conv0(conv1m_out)
        out = torch.nn.functional.pixel_shuffle(conv0_out, 2)
        if target is not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)


    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result