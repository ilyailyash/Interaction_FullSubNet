from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:

        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.k_conv = ConvBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.q_conv = ConvBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.v_conv = ConvBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.enc_conv = ConvBlock(in_channels//2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Attention Block

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """

        b, c, f, t = x.size()

        k = self.k_conv(x).reshape(b, c//2*f, t).permute(0, 2, 1)/(c//2*f)**0.5  # [B, T, C//2*F]
        q = self.q_conv(x).reshape(b, c//2*f, t).permute(0, 2, 1)
        v = self.v_conv(x).reshape(b, c//2*f, t).permute(0, 2, 1)

        mask = torch.sigmoid(torch.matmul(q, k.permute(0, 2, 1)))  # [B, T, T]
        sa = torch.matmul(mask, v).permute(0, 2, 1).reshape(b, c//2, f, t)  # [B, C//2, F, T]

        sa = self.enc_conv(sa) + x

        return sa


def unfold(signal, frame_size, stride):
    """
    Args:
        frame_size: int
        stride: int
        signal: [B, T]

    Return:
        [B, F, (T-S)//S]
    """
    signal = functional.pad(signal, [0, stride, 0, 0])
    signal = signal.unsqueeze(1).unsqueeze(-1)
    output = functional.unfold(signal, [frame_size, 1], stride=stride)

    return output


def fold(signal, output_size, frame_size, stride):
    """
    Args:
        output_size: tuple
        frame_size: int
        stride: int
        signal: [B, F, T]

    Return:
        [B, 1, T]
    """
    output_size = (output_size[0]+stride,  output_size[1])
    output = functional.fold(signal, output_size, [frame_size, 1], stride=stride)

    return output.squeeze(-1)[:, :, :-stride]


class Merge(nn.Module):
    def __init__(self, frame_size, unfold_stride):
        super().__init__()

        self.frame_size = frame_size
        self.unfold = partial(unfold, frame_size=frame_size, stride=unfold_stride)
        self.fold = partial(fold, frame_size=frame_size, stride=unfold_stride)
        self.temporal_attention = AttentionBlock(3)
        self.conv1 = ConvBlock(3, 3, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.conv2 = ConvBlock(3, 3, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.conv3 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(3, 7),
            stride=1,
            padding=(1, 3)
        )

    def forward(self, x, s, n):
        """
        Args:
            x: [B, T]
            s: [B, T]
            n: [B, T]

        Return:
            enhanced: [B, T]
        """
        x_unfold = self.unfold(x)                    # [B, F, T]
        s_unfold = self.unfold(s)
        n_unfold = self.unfold(n)

        x_stacked = torch.stack((x_unfold, s_unfold, n_unfold), dim=1)  # [B, 3, F, T]
        h = self.conv1(x_stacked)                                       # [B, 3, F, T]
        h = self.temporal_attention(h)                                  # [B, 3, F, T]
        h = self.conv2(h)                                               # [B, 3, F, T]
        h = self.conv3(h)                                               # [B, 1, F, T]
        mask = torch.sigmoid(h).squeeze(1)
        enhanced = mask*s_unfold + (1-mask)*(x_unfold-n_unfold)
        enhanced = enhanced
        enhanced = self.fold(enhanced, (x.shape[-1], 1))

        input_ones = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        divisor = self.fold(self.unfold(input_ones), (x.shape[-1], 1))
        enhanced = enhanced/divisor

        return enhanced.squeeze(1)

def framing(sigpad, nframe, wind, window_length):
    hsize = window_length//2
    frames = torch.zeros((sigpad.size(0), nframe, window_length)).type_as(sigpad)
    for i, frame_sampleindex in enumerate(range(0, nframe * hsize, hsize)):
        frames[:, i, :] = sigpad[:, 0, frame_sampleindex:frame_sampleindex +
                                 window_length]*wind.type_as(sigpad)
    return frames

def unframing(x_enh, nframe, zpleft, window_length):
    hsize = window_length//2

    sout = torch.zeros((x_enh.size(0), 1, nframe * hsize)).type_as(x_enh)
    x_old = torch.zeros(hsize).type_as(x_enh)
    for i, frame_sampleindex in enumerate(range(0, nframe * hsize, hsize)):
        sout[:, :, frame_sampleindex:frame_sampleindex + hsize] = x_old + x_enh[:, i:i + 1, :hsize]
        x_old = x_enh[:, i:i + 1, hsize:window_length]
    sout = sout[:, :, zpleft:]
    return sout

if __name__ == "__main__":
    x, s, n = torch.rand((3, 3, 48000), device='cuda:0')
    merge_model = Merge(512, 256)
    merge_model.to(torch.device('cuda:0'))
    enhanced = merge_model(x, s, n)


    print(enhanced)