#########################################################################
# Model similar to arc_change_net: more readable but less configurable #
########################################################################

#########################################
# https://arxiv.org/pdf/1505.04597.pdf #
########################################

import torch
import torch.nn as nn


""" Class for UNet architecture. 
    Specifically, the implemented UNet network architecture utilizes padded 
    convolution, therefore the padding of the double-layer convolutions is set to 1. """
class UNet(nn.Module):
    """ Initialize configurations. """
    def __init__(self, args):
        super(UNet, self).__init__()
        self.args = args
        # contracting-path
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = self.conv_layer(3, 64)
        self.down_2 = self.conv_layer(64, 128)
        self.down_3 = self.conv_layer(128, 256)
        self.down_4 = self.conv_layer(256, 512)
        self.down_5 = self.conv_layer(512, 1024)
        # expanding-path
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = self.conv_layer(1024, 512)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = self.conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = self.conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = self.conv_layer(128, 64)

        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.output_activation = nn.Sigmoid()

    """ Method used to create a convolutional layer. """
    def conv_layer(self, input_channels, output_channels):
        if self.args.use_batch_norm == True:
            conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        elif self.args.use_double_batch_norm == True:
            conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        elif self.args.use_inst_norm == True:
            conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU()
            )
        elif self.args.use_double_inst_norm == True:
            conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU()
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU()
            )

        return conv

    """ Method used to define the computations needed to 
        create the output of the neural network given its input. """ # (es.)
    def forward(self, img):                        # img --> bs x 3 x 224 x 224 (bs, ch, w, h)
        x1 = self.down_1(img)                      # bs x 64 x 224 x 224
        x2 = self.max_pool(x1)                     # bs x 64 x 112 x 112
        x3 = self.down_2(x2)                       # bs x 128 x 112 x 112
        x4 = self.max_pool(x3)                     # bs x 128 x 56 x 56
        x5 = self.down_3(x4)                       # bs x 256 x 56 x 56
        x6 = self.max_pool(x5)                     # bs x 256 x 28 x 28
        x7 = self.down_4(x6)                       # bs x 512 x 28 x 28
        x8 = self.max_pool(x7)                     # bs x 512 x 14 x 14
        x9 = self.down_5(x8)                       # bs x 1024 x 14 x 14

                                    # bs x 512 x 28 x 28 && x7 --> bs x 512 x 28 x 28
        x = self.up_1(x9)
        x = self.up_conv_1(torch.cat([x, x7], 1))  # bs x 512 x 28 x 28
        x = self.up_2(x)                           # bs x 256 x 56 x 56
        x = self.up_conv_2(torch.cat([x, x5], 1))  # bs x 256 x 56 x 56
        x = self.up_3(x)                           # bs x 128 x 112 x 112
        x = self.up_conv_3(torch.cat([x, x3], 1))  # bs x 128 x 112 x 112
        x = self.up_4(x)                           # bs x 64 x 224 x 224
        x = self.up_conv_4(torch.cat([x, x1], 1))  # bs x 64 x 224 x 224

        x = self.output(x)
        x = self.output_activation(x)              # bs x 1 x 224 x 224

        return x
