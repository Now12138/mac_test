import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from torch import nn
from torch.nn import init

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))#对输入张量x在指定的维度-d上执行一维快速傅里叶变换 (FFT)
        r = torch.stack((t.real, t.imag), -1)
        return r#形状为输入张量的形状加一维，最后一维大小为 2（表示实部和虚部）

    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real#返回一个实数张量，形状为原始输入 x 中去掉最后一维的形状

"""
这段代码实现了一个名为 `dct_channel_block` 的 PyTorch 模块，该模块包含了离散余弦变换（DCT）和通道注意力机制的功能。
`dct_channel_block`模块实现了对输入数据进行离散余弦变换和通道注意力机制的功能，可以被用于深度学习模型中对时间序列数据的处理和特征提取。
"""

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)#目的是实现DCT对称性

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


#结合了DCT的SE-block
class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.dct_norm = nn.LayerNorm([96], eps=1e-6)  # for lstm on length-wise

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight  # result


import einops
import pywt
from functools import partial
import torch.nn.functional as F


def create_wavelet_filter_1d(wave, in_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)#分解用的高通滤波器
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)#------低通滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type)#合成用的高通滤波器
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type)#------低通滤波器

    # Decomposition filters
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    # Reconstruction filters
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)
    rec_filters = rec_filters[:, None].repeat(in_size, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform_1d(x, filters):
    b, c, l = x.shape
    pad = filters.shape[2] // 2 - 1
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, l // 2)
    return x

def inverse_wavelet_transform_1d(x, filters):
    b, c, _, l_half = x.shape
    pad = filters.shape[2] // 2 - 1
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x
#论文里是用2D进行图像处理，这里将其转化为1D，将图像领域的扩展到时间序列一维领域
class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = create_wavelet_filter_1d(wt_type, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_1d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_1d, filters=self.iwt_filter)

        self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', stride=1, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size, padding='same', stride=1, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 2, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv1d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 > 0:
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1, :])

        #逆小波变换阶段
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h.unsqueeze(2)], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DepthwiseSeparableConvWithWTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv1d, self).__init__()

        self.depthwise = WTConv1d(in_channels, in_channels, kernel_size=kernel_size)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.dct = dct_channel_block(channel=self.input_size)
        self.wtconv = DepthwiseSeparableConvWithWTConv1d(in_channels=10, out_channels=10) #对应窗口
    def forward(self, input_seq,modle_num):
        if (modle_num == 4 or modle_num == 2):
            #模型进过WTC卷积降噪处理
            batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
            input_seq = dct(input_seq)
            #在经过lstm处理后在放入注意力里面
            h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
            c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
            output, (h, c) = self.lstm(input_seq, (h_0, c_0))
            #output = dct(output)
            return output, h
        elif (modle_num == 1 or modle_num == 3):
            batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
            h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
            c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
            output, (h, c) = self.lstm(input_seq, (h_0, c_0))
            return output, h

class LSTMMain(nn.Module):
    def __init__(self, input_size, output_len, lstm_hidden, lstm_layers, batch_size, device="cpu"):
        super(LSTMMain, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstmunit = LSTM(input_size, lstm_hidden, lstm_layers, batch_size, device)
        self.linear = nn.Linear(lstm_hidden, output_len)
        self.wtconv = DepthwiseSeparableConvWithWTConv1d(in_channels=10, out_channels=10)  # 对应窗口
    def forward(self, input_seq, modle_num):
        if (modle_num == 4 or modle_num == 3):
            #input_seq = self.wtconv(input_seq)
            ula, h_out = self.lstmunit(input_seq,modle_num)
            out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
            out = self.linear(out)
            out = out.view(ula.shape[0], ula.shape[1], -1)
            out = self.wtconv(out)
            out = out[:, -1, :]
            return out
        elif (modle_num == 2 or modle_num == 1):
            # input_seq = self.wtconv(input_seq)
            ula, h_out = self.lstmunit(input_seq,modle_num)
            out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
            out = self.linear(out)
            out = out.view(ula.shape[0], ula.shape[1], -1)
            out = out[:, -1, :]
            return out
