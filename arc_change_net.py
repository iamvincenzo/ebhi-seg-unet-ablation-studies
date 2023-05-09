#######################################################################################################################################
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet #
######################################################################################################################################

import os
import torch
import torch.nn as nn

""" Class for double-convolution model.  """
class DoubleConv(nn.Module):
    """ Initialize configurations. """
    def __init__(self, args, input_channels, output_channels):
        super(DoubleConv, self).__init__()

        #######################################################################################################
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

    """ Method used to get the convolutional layer. """
    def forward(self, x):
        return self.conv(x)


""" Class for customizable UNet architecture. """
class UNET(nn.Module):
    """ Initialize configurations. """
    def __init__(self, args, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.args = args

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(self.args, in_channels, feature))
            in_channels = feature

        # up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(self.args, feature*2, feature))

        # bottleneck of UNET
        self.bottleneck = DoubleConv(self.args, features[-1], features[-1]*2)
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, padding=0)
        
        self.output_activation = nn.Sigmoid()

        if self.args.weights_init == True:
            self.initialize_weights()

        # saving the model before the training process
        self.save_initial_weights_distribution()

    """ Method used to define the computations needed to create the 
        output of the neural network given its input. """
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

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        # remember: arc_change_net can work both 
        # with bcewl_loss and dc_loss/jac_loss
        if self.args.loss != 'bcewl_loss':
            x = self.output_activation(x)

        return x
    
    """ Method used to initialize the weights of the network. """
    def initialize_weights(self):
        print('\nPerforming weights initialization...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    """ Helper method used to save some data used in ablation studies. """
    def save_initial_weights_distribution(self):
        if (not(self.args.global_ablation) and not(self.args.selective_ablation) and 
            not(self.args.all_one_by_one)):        
            check_path = os.path.join(self.args.checkpoint_path, self.args.model_name + '_before_training.pth')
            torch.save(self.state_dict(), check_path)
            print('\nModel saved (before training)!\n')
