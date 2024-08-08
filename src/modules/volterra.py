import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.weight: nn.Parameter
        self.bias: nn.Parameter
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class VolterraGate(nn.Module):
    def __init__(self, q_rank):
        super().__init__()
        self.q_rank = q_rank

    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=1)
        out = x1 * x2  
        x_shape = out.shape
        out = out.view(x_shape[0], self.q_rank, x_shape[1] // self.q_rank, x_shape[2], x_shape[3]).sum(dim=1)
        return  out

class VolterraBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, padding = 0, stride = 1, dilation = 1, groups = 1, q_rank = 4, drop_out_rate=0.):
        super().__init__()

        # First-order convolution
        self.f_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, dilation=dilation,
                               bias=True)
        # Second-order convolution
        DW_Expand = 2
        dw_channel = in_channels * DW_Expand
        self.s_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=groups, bias=True)
        self.s_conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel * q_rank, kernel_size=kernel_size, padding=padding, stride=stride, groups=dw_channel, dilation=dilation,
                               bias=True)
        ## SimpleGate
        self.sg = VolterraGate(q_rank= q_rank)
        ## Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.s_conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=groups, bias=True)

        self.norm1 = LayerNorm2d(in_channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        f_out = self.f_conv(x)
        x = self.norm1(x)

        x = self.s_conv1(x)
        x = self.s_conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.s_conv3(x)

        x = self.dropout1(x)

        y = f_out + x * self.beta

        return y