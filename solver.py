import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import UNet
from arc_change_net import UNET
from metrics import dc_loss, jac_loss, binary_jac, binary_acc, binary_prec, binary_rec, binary_f1s
from plotting_utils import set_default, add_gradient_hist, add_metric_hist, plot_check_results, kernels_viewer, activations_viewer

""" Solver for training and testing. """
class Solver(object):
    def __init__(self, train_loader, test_loader, device, writer, args):
        """ Initialize configurations. """
        self.args = args
        self.model_name = 'ebhi-seg_u-net_{}.pth'.format(self.args.model_name)

        if self.args.pretrained_net == True:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
            print(f'\nPretrained model implementation selected:\n\n {model}')
        elif self.args.arc_change_net == True:
            model = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            print(f'\nOnline model implementation selected:\n\n {model}')
        else:
            model = UNet(self.args)
            print(f'\nStandard model implementation selected:\n\n {model}')

        # define the model
        self.net = model.to(device)

        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model(device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss
            print(f'\nDC_loss selected!\n')
        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss
            print(f'\nJAC_loss selected!\n')
        elif self.args.loss == 'bcewl_loss' and self.args.arc_change_net == True:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f'\nBCEWithLogitsLoss selected!\n')

        # choose optimizer
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.args.lr, betas=(0.9, 0.999))

        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device

        self.writer = writer

        # visualize the model we built on tensorboard
        images, _, _ = next(iter(self.train_loader)) # images = images.to(device) ???
        self.writer.add_graph(self.net, images.to(self.device))
        self.writer.close()

        set_default() # setting fig-style

    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print('\nModel saved!\n')

    def load_model(self, device):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path, map_location=torch.device(device)))
        print('\nModel loaded!\n')

    """ Helper function used to binarize a tensor (mask)
        in order to compute accuracy, precision, recall, f1-score. """
    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf

    """ Helper function used to train the model with 
        early stopping implementatinon. """
    def train(self):
        # keep track of average training and test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []

        # trigger for earlystopping
        earlystopping = False

        self.net.train()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            # record the training and test losses for each batch in this epoch
            train_losses = []
            test_losses = []

            loop = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader), leave=False)

            for batch, (images, targets, _) in loop:
                images = images.to(self.device)
                targets = targets.to(self.device)  # the ground truth mask

                # zero the parameter gradients
                self.optimizer.zero_grad()  # self.net.zero_grad() ???

                # forward + backward + optimize
                pred = self.net(images)
                loss = self.criterion(pred, targets)
                loss.backward()

                # plot gradient histogram distribution ????
                if (batch % self.args.print_every == self.args.print_every - 1) or (batch == 0 and epoch == 0):
                    self.writer.add_figure('gradients_ebhi-seg', add_gradient_hist(
                        self.net), global_step=epoch * len(self.train_loader) + batch)

                self.optimizer.step()

                train_losses.append(loss.item())

                # used to check model improvement
                self.check_results(batch, epoch)

                if batch % self.args.print_every == self.args.print_every - 1 or (batch == 0 and epoch == 0):
                    self.save_model()  # save at the each (print_every - 1)

                    # Test model
                    if self.args.bs_test == 1:
                        dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, prec_class_test, rec_class_test, f1s_class_test = self.test(
                            test_losses)
                    else:
                        self.test(test_losses)

                    batch_avg_train_loss = np.average(train_losses)
                    batch_avg_test_loss = np.average(test_losses)

                    avg_train_losses.append(batch_avg_train_loss)
                    avg_test_losses.append(batch_avg_test_loss)

                    # general dc in train/test
                    batch_avg_train_dc = np.average(
                        [1 - x for x in train_losses])
                    batch_avg_test_dc = np.average(
                        [1 - x for x in test_losses])

                    print(f'\ntrain_loss: {batch_avg_train_loss:.5f} ' +
                          f'valid_loss: {batch_avg_test_loss:.5f}\n')

                    self.writer.add_scalar('batch_avg_train_loss', batch_avg_train_loss,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_loss', batch_avg_test_loss,
                                           global_step=epoch * len(self.train_loader) + batch)

                    self.writer.add_scalar('batch_avg_train_dc', batch_avg_train_dc,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_dc', batch_avg_test_dc,
                                           global_step=epoch * len(self.train_loader) + batch)

                    if self.args.bs_test == 1:
                        print(
                            f'Dice for class {[np.average(x) for x in dc_class_test]}')
                        print(
                            f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                        print(
                            f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                        print(
                            f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                        print(
                            f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                        print(
                            f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                        print(
                            f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

                        # log a Matplotlib Figure showing the metric for each class
                        self.writer.add_figure('accuracy_histo', add_metric_hist([np.average(
                            x) for x in acc_class_test], 'Accuracy'), global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('precision_histo', add_metric_hist([np.average(
                            x) for x in prec_class_test], 'Precision'), global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('recall_histo', add_metric_hist([np.average(
                            x) for x in rec_class_test], 'Recall'), global_step=epoch * len(self.train_loader) + batch)

                    print(
                        f'\nGloabl-step: {epoch * len(self.train_loader) + batch}')

                    train_losses = []
                    test_losses = []

                    if epoch > self.args.early_stopping:  # Early stopping with a patience of 1 and a minimum of N epochs
                        if avg_test_losses[-1] >= avg_test_losses[-2]:
                            print('\nEarly Stopping Triggered With Patience 1')
                            self.save_model()  # save before stop training
                            earlystopping = True
                    if earlystopping:
                        break

            if earlystopping:
                break

            self.save_model()  # save at the end of each epoch

        # if self.args.pretrained_net == False:
        #     self.kernel_analisys() # for debugging
        if self.args.arc_change_net == False:
            print(f'\nStarting kernel activations analysis!\n')
            self.activation_analisys()

        self.writer.flush()
        self.writer.close()
        print('Finished Training!\n')


    """ Helper function used to evaluate the model on the test set. """
    def test(self, test_losses):
        self.net.eval()  # put net into evaluation mode

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # compute some metrics performance for each class on the test set
            if self.args.bs_test == 1:
                dc_class_test = list([[], [], [], [], [], []])
                jac_cust_class_test = list([[], [], [], [], [], []])
                jac_class_test = list([[], [], [], [], [], []])
                acc_class_test = list([[], [], [], [], [], []])
                prec_class_test = list([[], [], [], [], [], []])
                rec_class_test = list([[], [], [], [], [], []])
                f1s_class_test = list([[], [], [], [], [], []])

            test_loop = tqdm(enumerate(self.test_loader),
                             total=len(self.test_loader), leave=False)

            for _, (test_images, test_targets, test_labels) in test_loop:
                test_images = test_images.to(self.device)
                test_targets = test_targets.to(self.device)
                test_pred = self.net(test_images.detach())

                # general loss
                test_loss = self.criterion(test_pred, test_targets).item()
                test_losses.append(test_loss)

                if self.args.bs_test == 1:
                    # function that binarize an image and then convert it to a tensor
                    maskf, predf = self.binarization_tensor(
                        test_targets, test_pred)

                    # per class metrics
                    dc_class_test[test_labels].append(1 - test_loss)
                    jac_cust_class_test[test_labels].append(
                        1 - jac_loss(test_pred, test_targets).item())
                    jac_class_test[test_labels].append(
                        binary_jac(maskf, predf))
                    acc_class_test[test_labels].append(
                        binary_acc(maskf, predf))
                    prec_class_test[test_labels].append(
                        binary_prec(maskf, predf))
                    rec_class_test[test_labels].append(
                        binary_rec(maskf, predf))
                    f1s_class_test[test_labels].append(
                        binary_f1s(maskf, predf))

        self.net.train()  # put again the model in trainining-mode

        if self.args.bs_test == 1:
            return dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, prec_class_test, rec_class_test, f1s_class_test

    """ Helper function used to visualize show some samples at
        the first batch of each epoch to check model improvements. """
    def check_results(self, batch, epoch):
        with torch.no_grad():
            if batch == 1:
                self.net.eval()

                (img, mask, label) = next(iter(self.test_loader))
                img = img.to(self.device)
                mask = mask.to(self.device)
                mask = mask[0]
                pred = self.net(img)

                self.writer.add_figure('check_results', plot_check_results(
                    img[0], mask, pred[0], label[0], self.args), global_step=epoch * len(self.train_loader) + batch)

                self.net.train()


    """ Helper function used to visualize CNN kernels. """
    def kernel_analisys(self):

        layer_list = [self.net.down_1, self.net.down_2, self.net.down_3,
                      self.net.down_4, self.net.down_5, self.net.up_conv_1,
                      self.net.up_conv_2, self.net.up_conv_3,
                      self.net.up_conv_4, self.net.output]

        kernels_viewer(layer_list, self.net.output, self.writer)


    """ Helper function used to visualize CNN activations. """
    def activation_analisys(self):

        if self.args.pretrained_net == True:
            layer_list = [self.net.encoder1, self.net.encoder2, self.net.encoder3,
                          self.net.encoder4, self.net.bottleneck, self.net.decoder4,
                          self.net.decoder3, self.net.decoder2,
                          self.net.decoder1, self.net.conv]
            out_l = self.net.conv
        else:
            layer_list = [self.net.down_1, self.net.down_2, self.net.down_3,
                          self.net.down_4, self.net.down_5, self.net.up_conv_1,
                          self.net.up_conv_2, self.net.up_conv_3,
                          self.net.up_conv_4, self.net.output]
            out_l = self.net.output

        (img, _, _) = next(iter(self.test_loader))
        img = img.to(self.device)

        activations_viewer(layer_list, self.net, self.writer, img, out_l)
