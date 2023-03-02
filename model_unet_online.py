######################################################################################################################################
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet #
######################################################################################################################################
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, args, input_channels, output_channels):
        super(DoubleConv, self).__init__()

        ######################################################################################################
        # https://datascience.stackexchange.com/questions/46407/conv-bias-or-not-with-instance-normalization #
        ######################################################################################################

        if args.use_batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        elif args.use_double_batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        elif args.use_inst_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU(inplace=True)
            )

        elif args.use_double_inst_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU(inplace=True)
            )

        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(args, in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(args, feature*2, feature))

        self.bottleneck = DoubleConv(args, features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=1, padding=0)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # if x.shape != skip_connection.shape:
            #     x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        # x = self.output_activation(x)

        return x
