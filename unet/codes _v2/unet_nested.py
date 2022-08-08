"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.nest01 = NestedBlock(chans, chans*2)
        self.nest02 = NestedBlock(chans, chans*2)
        self.nest03 = NestedBlock(chans, chans*2)
        self.nest11 = NestedBlock(chans*2, chans*4)
        self.nest12 = NestedBlock(chans*2, chans*4)
        self.nest21 = NestedBlock(chans*4, chans*8)
                                
            
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        
        

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        
        
        # apply down-sampling layers
        for layer in (self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        
        ##Itr31
        transpose_conv = self.up_transpose_conv[0]
        conv = self.up_conv[0]
        downsample_layer = stack.pop()  
        output = transpose_conv(output)

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if output.shape[-1] != downsample_layer.shape[-1]:
            padding[1] = 1  # padding right
        if output.shape[-2] != downsample_layer.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            output = F.pad(output, padding, "reflect")

        output = torch.cat([output, downsample_layer], dim=1)
        output = conv(output)
       
        ds_last = downsample_layer.detach().clone() 
        
        ##Itr22
        transpose_conv = self.up_transpose_conv[1]
        conv = self.up_conv[1]
        downsample_layer = stack.pop()  
        
       
        nest_op_21 = downsample_layer + self.nest21(downsample_layer,ds_last)
        
        output = transpose_conv(output)

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if output.shape[-1] != downsample_layer.shape[-1]:
            padding[1] = 1  # padding right
        if output.shape[-2] != downsample_layer.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            output = F.pad(output, padding, "reflect")

        output = torch.cat([output, nest_op_21], dim=1)
        output = conv(output)
        
        ds_last = downsample_layer.detach().clone() 
    
        
        ##Itr13
        transpose_conv = self.up_transpose_conv[2]
        conv = self.up_conv[2]
        downsample_layer = stack.pop()  
        
        nest_op_11 = downsample_layer + self.nest11(downsample_layer,ds_last)
        nest_op_12 = downsample_layer + self.nest12(nest_op_11,nest_op_21)
        
        output = transpose_conv(output)

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if output.shape[-1] != downsample_layer.shape[-1]:
            padding[1] = 1  # padding right
        if output.shape[-2] != downsample_layer.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            output = F.pad(output, padding, "reflect")

        output = torch.cat([output, nest_op_12], dim=1)
        output = conv(output)
       
    
    
        ds_last = downsample_layer.detach().clone() 
     
        ##Itr04
        transpose_conv = self.up_transpose_conv[3]
        conv = self.up_conv[3]
        downsample_layer = stack.pop()  
        
        nest_op_01 = downsample_layer + self.nest01(downsample_layer,ds_last)
        nest_op_02 = downsample_layer + self.nest02(nest_op_01,nest_op_11)
        nest_op_03 = downsample_layer + self.nest03(nest_op_02,nest_op_12)
        
        output = transpose_conv(output)

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if output.shape[-1] != downsample_layer.shape[-1]:
            padding[1] = 1  # padding right
        if output.shape[-2] != downsample_layer.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            output = F.pad(output, padding, "reflect")

        output = torch.cat([output, nest_op_03], dim=1)
        output = conv(output)
       
       

    
        # apply up-sampling layers
#         for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
#             downsample_layer = stack.pop()  
#             d_layer = stack_dense.pop()
#             output = transpose_conv(output)

#             # reflect pad on the right/botton if needed to handle odd input dimensions
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # padding bottom
#             if torch.sum(torch.tensor(padding)) != 0:
#                 output = F.pad(output, padding, "reflect")
            
#             output = torch.cat([output, d_layer], dim=1)
#             output = conv(output)

        return output

class NestedBlock(nn.Module):
    def __init__(self, in_chan1: int, in_chan2: int):
        super().__init__()

        self.in_chan1 = in_chan1
        self.in_chan2 = in_chan2

        self.layers = nn.Sequential(
            nn.Conv2d(in_chan1*2, in_chan1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan1),
            nn.ReLU(),
            )
        
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_chan2, in_chan1, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_chan1),
            nn.ReLU(),
            )
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:

        return self.layers(torch.cat([self.downsample(image2),image1],dim=1))
    
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
