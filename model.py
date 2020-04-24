import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fft_utils

class Net(nn.Module):
    def __init__(self, ksize=5, num=5):
        super(Net, self).__init__()
        self.shuffle_down_4 = ComplexShuffleDown(4)
        self.shuffle_up_4 = ComplexShuffleUp(4)
        self.shuffle_up_2 = ComplexShuffleUp(2)
        self.backbone = Backbone(16, 64)

        self.branch1 = nn.Sequential(ComplexConv2d(4, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())
        self.branch2 = nn.Sequential(ComplexConv2d(16, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())
        self.branch3 = nn.Sequential(ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())
        self.pixel_conv = PerPixelConv()
        self.pyramid_decompose = PyramidDecompose()
        self.pyramid_reconstruct = PyramidReconstruct()
        self.num = num

    def forward(self, temp, x_k, mask):
        for i in range(self.num):
            gaussian_pyramid, laplacian_pyramid = self.pyramid_decompose(temp)
            lap_1, lap_2 = laplacian_pyramid[0], laplacian_pyramid[1]
            gaussian_3 = gaussian_pyramid[-1]

            # x.size() = torch.Size([Batch, 1, 256, 256])
            temp = self.shuffle_down_4(temp)          # torch.Size([Batch, 16, 64, 64])
            temp = self.backbone(temp)                # torch.Size([Batch, 64, 64, 64])

            # 4x
            branch_1 = self.shuffle_up_4(temp)     # torch.Size([Batch, 4, 256, 256])
            branch_1 = self.branch1(branch_1)   # torch.Size([Batch, 25, 256, 256])
            # 2x
            branch_2 = self.shuffle_up_2(temp)     # torch.Size([Batch, 16, 128, 128])
            branch_2 = self.branch2(branch_2)   # torch.Size([Batch, 25, 128, 128])
            # 1x
            branch_3 = self.branch3(temp)          # torch.Size([Batch, 25, 64, 64])

            output1 = torch.stack((self.pixel_conv(branch_1[...,0], lap_1[...,0]),
                                   self.pixel_conv(branch_1[...,1], lap_1[...,1])), dim=-1)
            output2 = torch.stack((self.pixel_conv(branch_2[...,0], lap_2[...,0]),
                                  self.pixel_conv(branch_2[...,1], lap_2[...,1])), dim=-1)
            output3 = torch.stack((self.pixel_conv(branch_3[...,0], gaussian_3[...,0]),
                                   self.pixel_conv(branch_3[...,1], gaussian_3[...,1])), dim=-1)

            output = self.pyramid_reconstruct(output3, output2)
            output = self.pyramid_reconstruct(output, output1)
            temp = fft_utils.ifft2((1.0 - mask) * fft_utils.fft2(output) + x_k)
        return temp


class PerPixelConv(nn.Module):
    def __init__(self):
        super(PerPixelConv, self).__init__()

    def forward(self, kernel, image):
        b, ksize2, h, w = kernel.size()
        ksize = np.int(np.sqrt(ksize2))
        padding = (ksize - 1) // 2
        image = F.pad(image, (padding, padding, padding, padding))
        image = image.unfold(2, ksize, 1).unfold(3, ksize, 1)
        image = image.permute(0, 2, 3, 1, 5, 4).contiguous()
        image = image.reshape(b, h, w, 1, -1)
        kernel = kernel.permute(0, 2, 3, 1).unsqueeze(-1)
        output = torch.matmul(image, kernel)
        output = output.reshape(b, h, w, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.residualblock = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.residualblock(x)


class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Backbone, self).__init__()
        backbone = [ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()]
        for i in range(4):
            backbone.append(ResidualBlock(out_channels,out_channels))
        backbone.append(ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        backbone.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        return self.backbone(x)


class PyramidDecompose(nn.Module):
    def __init__(self):
        super(PyramidDecompose, self).__init__()
        kernel = np.float32([1, 4, 6, 4, 1])
        kernel = np.outer(kernel, kernel)
        kernel = kernel[:, :, None, None] / kernel.sum()
        kernel = torch.from_numpy(np.transpose(kernel, (2, 3, 0, 1)))                # (1, 1, 256, 256, 2)
        self.downfilter = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.downfilter.weight = nn.Parameter(kernel, requires_grad=False)
        self.upfilter = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.upfilter.weight = nn.Parameter(kernel * 4, requires_grad=False)

    @staticmethod
    def downsample(x):
            return x[:, :, ::2, ::2]

    @staticmethod
    def upsample(x):
        b, c, hin, win = x.size()
        hout, wout = hin * 2, win * 2
        x_upsample = torch.zeros((b, c, hout, wout), device='cuda')
        x_upsample[:, :, ::2, ::2] = x
        return x_upsample

    def forward(self, x):
        gaussian_pyramid = [x]
        laplacian_pyramid = []
        for i in range(2):
            downsample_result = torch.stack((self.downsample(self.downfilter(gaussian_pyramid[-1][..., 0])),
                                             self.downsample(self.downfilter(gaussian_pyramid[-1][..., 1]))), dim=-1)   # g = [256]         g = [256, 128]
            gaussian_pyramid.append(downsample_result)                                  # g = [256, 128]    g = [256, 128, 64]
            upsample_result = torch.stack((self.upfilter(self.upsample(downsample_result[..., 0])),
                                           self.upfilter(self.upsample(downsample_result[..., 1]))), dim=-1)
            residual = gaussian_pyramid[-2] - upsample_result
            laplacian_pyramid.append(residual)                                          # l = [256]         l = [256, 128]
        return gaussian_pyramid, laplacian_pyramid


class PyramidReconstruct(nn.Module):
    def __init__(self):
        super(PyramidReconstruct, self).__init__()
        kernel = np.float32([1, 4, 6, 4, 1])
        kernel = np.outer(kernel, kernel)
        kernel = kernel[:, :, None, None] / kernel.sum()
        kernel = torch.from_numpy(np.transpose(kernel, (2, 3, 0, 1)))
        self.upfilter = nn.Conv2d(1, 1, 5, 1, 2, bias=False)
        self.upfilter.weight = nn.Parameter(kernel * 4, requires_grad=False)

    @staticmethod
    def upsample(x):
        b, c, hin, win = x.size()
        hout, wout = hin * 2, win * 2
        x_upsample = torch.zeros((b, c, hout, wout), device='cuda')
        x_upsample[:, :, ::2, ::2] = x
        return x_upsample

    def forward(self, x_gaussian, x_laplacian):
        upsample_result = torch.stack((self.upfilter(self.upsample(x_gaussian[..., 0])), self.upfilter(self.upsample(x_gaussian[..., 1]))), dim=-1)
        recon = upsample_result + x_laplacian
        return recon


class ComplexShuffleDown(nn.Module):
    def __init__(self, scale):
        super(ComplexShuffleDown, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, cin, hin, win, complex_c = x.size()
        cout = cin * self.scale ** 2
        hout = hin // self.scale
        wout = win // self.scale
        output = x.view(b, cin, hout, self.scale, wout, self.scale, complex_c)
        output = output.permute(0, 1, 5, 3, 2, 4, 6).contiguous()
        output = output.view(b, cout, hout, wout, complex_c)
        return output


class ComplexShuffleUp(nn.Module):
    def __init__(self, scale):
        super(ComplexShuffleUp, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, cin, hin, win, complex_c = x.size()
        cout = cin // (self.scale ** 2)
        hout = hin * self.scale
        wout = win * self.scale
        output = x.view(b, cout, self.scale, self.scale, hin, win, complex_c)
        output = output.permute(0, 1, 4, 3, 5, 2, 6).contiguous()
        output = output.view(b, cout, hout, wout, complex_c)
        return output


class DataConsistency(nn.Module):
    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, x, x_k, mask):
        output = torch.rand(size=x.size()).cuda()
        for i in range(len(x)):
            output[i] = fft_utils.ifft2(x_k[i] + fft_utils.fft2(x[i]) * (1.0 - mask[i]))
        return output


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()

        self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # x.shape = (batch, channel, axis1, axis2, 2)
        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])
        imag = self.conv_imag(x[..., 0]) + self.conv_real(x[..., 1])
        output = torch.stack((real, imag), dim=4)
        return output
