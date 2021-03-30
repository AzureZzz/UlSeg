import torch
from torch.nn import Module, Conv2d, Parameter, Softmax

__all__ = ['PAMModule', 'CAMModule']


class PAMModule(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_channels):
        super(PAMModule, self).__init__()
        self.in_channels = in_channels

        self.query_conv = Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        N, C, H, W = x.size()
        proj_query = self.query_conv(x).view(N, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)    # tensor matrix multiply
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(N, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma * out + x
        return out


class CAMModule(Module):
    """ Channel attention module"""

    def __init__(self, in_channels):
        super(CAMModule, self).__init__()
        self.in_channels = in_channels

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)

        out = self.gamma * out + x
        return out
