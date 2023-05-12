import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.utils.prune as prune
from plotting_utils import plot_check_results
import torchvision.transforms.functional as TF
from metrics import jac_loss, binary_jac, binary_acc
from metrics import binary_prec, binary_rec, binary_f1s


""" Class used to perform ablation studies based 
    on sensitivity analysis using different methods, such as 
    global_unstructured, l1_unstructured, random_structured, and random_unstructured. """
class AblationStudies(object):
    """ Initialize configurations. """
    def __init__(self, args, model_name, train_loader, test_loader, model, criterion, device, writer):
        self.args = args
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.device = device
        self.writer = writer

        # parameter used to distinguish saved runs in TensorBoard
        # such as in case of "all_one_by_one" study
        self.t = f"{datetime.now():%d%m%Y-%H%M%S}"

        mpl.rcParams.update(mpl.rcParamsDefault)

        # dictionary used to store collected results during the study
        self.my_dic_ablation_results = {}

    """ Method used to determine the number of zeros in the tensor 
        (module), in order to determine the local percentage of ablation. """
    def measure_module_sparsity(self, module, weight=True, bias=False, use_mask=False):
        num_zeros = 0
        num_elements = 0

        if use_mask == True:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight == True:
                    # get the total weights that are set to 0 in buffer 
                    num_zeros += torch.sum(buffer == 0).item()
                    # get the total number of weights in buffer
                    num_elements += buffer.nelement()
                if "bias_mask" in buffer_name and bias == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
        else:
            for param_name, param in module.named_parameters():
                if "weight" in param_name and weight == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()
                if "bias" in param_name and bias == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    """ Method used to determine the number of zeros in the 
        model, in order to determine the global percentage of ablation. """
    def measure_global_sparsity(self, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False):
        num_zeros = 0
        num_elements = 0

        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

            elif isinstance(module, torch.nn.Linear):
                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=linear_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    """ This method unfreezes pruned weights, removes original 
        weights and masks, and deletes the corresponding parameter and buffer 
        entries. The pruned parameter 'name' remains pruned, while the original 
        parameter 'name' + '_orig' and the mask buffer 'name' + '_mask' are deleted. """
    def remove_parameters(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass
            elif isinstance(module, torch.nn.Linear):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass

        return self.model

    """ Method used to save collected results into a json file. """
    def save_abl_results(self):
        with open('./abl_statistics/ablation_results_' + self.args.model_name + "_" + self.args.run_name + 
                  '_' + f"{datetime.now():%d%m%Y-%H%M%S}" + '.json', 'w') as f:
            json.dump(self.my_dic_ablation_results, f)

    """ Method used to save pruned models. """
    def save_abl_model(self):
        check_path = os.path.join(self.args.checkpoint_path, 'pruned_' + self.model_name)            
        torch.save(self.model.state_dict(), check_path)
        print('\nModel saved...\n')

    """ Method used to prune the model iteratively at 
        the variation of the percentage of pruned parameters. """
    def iterative_pruning(self, grouped_pruning=False):
        # at each iteration a certain percentage of model's weights is set to zero
        for i in range(self.args.num_iterations):
            print('\nPruning {}/{}'.format(i + 1, self.args.num_iterations))
            print('\nPruning...')

            """ Globally prunes tensors corresponding to all parameters in parameters_to_prune 
                by applying the specified pruning_method. 
                Modifies modules in place by:
                    1. adding a named buffer called 'name' + '_mask' corresponding to the binary 
                        mask applied to the parameter 'name' by the pruning method;
                    2. replacing the parameter name by its pruned version, while the original 
                        (unpruned) parameter is stored in a new parameter named name + '_orig'.

                    L1Unstructured: prune (currently unpruned) units 
                        in a tensor by zeroing out the ones with the lowest L1-norm. """
            if grouped_pruning == True:
                print('\nGrouped...')
                parameters_to_prune = []
                for _, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        parameters_to_prune.append((module, 'weight'))

                prune.global_unstructured(parameters_to_prune,
                                          pruning_method=prune.L1Unstructured,
                                          amount=self.args.conv2d_prune_amount)
                
                """ Pruning each module by conv2d_prune_amount/linear_prune_amount %. """
            else:
                for _, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module,
                                              name='weight',
                                              amount=self.args.conv2d_prune_amount)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module,
                                              name='weight',
                                              amount=self.args.linear_prune_amount)
                        
            num_zeros, num_elements, sparsity = self.measure_global_sparsity(weight=True, 
                bias=False, conv2d_use_mask=True, linear_use_mask=False)
            
            print(f'Global Sparsity: {sparsity:.2f}\n')
 
            self.my_dic_ablation_results['sparsity-global-' + str(i + 1)] = str(sparsity)

            # test the model: 
            # check the performance after the pruning
            self.test(i)

            self.save_abl_results()
            self.my_dic_ablation_results = {}

        # at the end remove the mask and the original weights
        self.remove_parameters()
        # and save the pruned model
        self.save_abl_model()

    """ Method used to prune only the modules passed as parameters. """
    def selective_pruning(self, mod_name_list=['downs.0.conv.0']):
        for i in range(self.args.num_iterations):
            print('\nSelective Pruning {}/{}'.format(i + 1, self.args.num_iterations))
            print('\nStarting pruning...')
            print(f'\non: {mod_name_list}...\n')

            """ Prunes tensor corresponding to parameter called name in module by 
                removing the specified amount of (currently unpruned) channels along 
                the specified dim selected at random.                             
                    - random_structured prunes entire filters/channels in a 
                        structured way (i.e.; following some predefined structure);
                    - random_unstructured prunes individual weights in an 
                        unstructured way.
                    Both methods use random sparsity patterns. """
            for module_name, module in self.model.named_modules():
                for mod_name in mod_name_list:
                    if mod_name == module_name and isinstance(module, torch.nn.Conv2d):
                        if self.args.random_structured == True:
                            # pruning is being applied along the channel dimension of 
                            # the convolutional layer. This means that entire channels or filters 
                            # will be removed from the layer, rather than individual weights.
                            # (i.e.; it reduces the feature-map, number of neurons)
                            prune.random_structured(module,
                                                    name='weight',
                                                    amount=self.args.conv2d_prune_amount,
                                                    dim=0)
                        elif self.args.random_unstructured == True:
                            prune.random_unstructured(module, 
                                                      name='weight',
                                                      amount=self.args.conv2d_prune_amount)
                                                                        
                        module_num_zeros, module_num_elements, sparsity = self.measure_module_sparsity(
                            module, weight=True, bias=False, use_mask=True)

                        print(f'\nmod_name: {mod_name}, num_zeros: {module_num_zeros}, '
                              f'num_elements: {module_num_elements}, sparsity: {sparsity}')

                        self.my_dic_ablation_results['sparsity-Conv2d-' + mod_name + '-' + str(i + 1)] = str(sparsity)

                    elif mod_name == module_name and isinstance(module, torch.nn.Linear):
                        if self.args.random_structured == True:
                            prune.random_structured(module,
                                                    name='weight',
                                                    amount=self.args.linear_prune_amount,
                                                    dim=0)
                        elif self.args.random_unstructured == True:
                            prune.random_unstructured(module, 
                                                    name='weight',
                                                    amount=self.args.linear_prune_amount)

                        module_num_zeros, module_num_elements, sparsity = self.measure_module_sparsity(
                            module, weight=True, bias=False, use_mask=True)
                        
                        print(f'\nmod_name: {mod_name}, num_zeros: {module_num_zeros}, '
                              f'num_elements: {module_num_elements}, sparsity: {sparsity}')

                        self.my_dic_ablation_results['sparsity-Linear' + mod_name + '-' + str(i + 1)] = str(sparsity)
            
            # debugging: print some info after to see sparse tensor
            if self.args.plt_weights_distr == True:
                self.plot_weights_distribution(mod_name_list)
          
            # test the model:
            # check the performance after the pruning
            self.test(i)

            self.save_abl_results()
            self.my_dic_ablation_results = {}

        # at the end remove the mask and the original weights
        self.remove_parameters()
        # and save the pruned model
        self.save_abl_model()

    """ Method used to binarize a tensor (mask)
        in order to compute binary accuracy, precision, recall, f1-score. 
        The convert('1') function creates a thresholded image where all pixels with values 
        less than 128 are set to 0, and all pixels with values greater than or equal to 128 
        are set to 1. """
    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        # np.squeeze(mask) 
        # --> [1, 3, 224, 224] --> [3, 224, 224]
        # convert: 
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf

    """ Method used to test the performance of the model after pruning. """
    def test(self, itr):
        print(f'\nPerforming validation-test after pruning...\n')

        test_losses = []

        # put net into evaluation mode
        self.model.eval()  

        # no need to calculate the gradients for outputs
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

            # used for creating a terminal progress bar
            test_loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=True)

            for batch_test, (test_images, test_targets, test_labels) in test_loop:
                # put data on correct device
                test_images = test_images.to(self.device)
                test_targets = test_targets.to(self.device)
                test_pred = self.model(test_images.detach())

                # general loss
                test_loss = self.criterion(test_pred, test_targets).item()
                test_losses.append(test_loss)

                # used to visually-check the performance of the model
                self.check_results(batch_test, itr)

                if self.args.bs_test == 1:
                    # function that binarize an image and then convert it to a tensor
                    maskf, predf = self.binarization_tensor(test_targets, test_pred)

                    # computation of per class metrics
                    dc_class_test[test_labels].append(1 - test_loss)
                    jac_cust_class_test[test_labels].append(1 - jac_loss(test_pred, test_targets).item())
                    jac_class_test[test_labels].append(binary_jac(maskf, predf))
                    acc_class_test[test_labels].append(binary_acc(maskf, predf))
                    prec_class_test[test_labels].append(binary_prec(maskf, predf))
                    rec_class_test[test_labels].append(binary_rec(maskf, predf))
                    f1s_class_test[test_labels].append(binary_f1s(maskf, predf))

            batch_avg_test_loss = np.average(test_losses)

            # saving some data in a dictionary (1)                    
            self.my_dic_ablation_results['avg_test_losses'] = str(batch_avg_test_loss)

            print(f'\nvalid_loss: {batch_avg_test_loss:.5f}\n')

            if self.args.bs_test == 1:
                print(f'Dice for class {[np.average(x) for x in dc_class_test]}')
                print(f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                print(f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                print(f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                print(f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                print(f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                print(f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

                # saving some data in a dictionary (2)                    
                self.my_dic_ablation_results['dc_class_test_mean'] = [str(np.average(x)) for x in dc_class_test]
                self.my_dic_ablation_results['jac_cust_class_test_mean'] = [str(np.average(x)) for x in jac_cust_class_test]
                self.my_dic_ablation_results['jac_class_test_mean'] = [str(np.average(x)) for x in jac_class_test]
                self.my_dic_ablation_results['acc_class_test_mean'] = [str(np.average(x)) for x in acc_class_test]
                self.my_dic_ablation_results['prec_class_test_mean'] = [str(np.average(x)) for x in prec_class_test]
                self.my_dic_ablation_results['rec_class_test_mean'] = [str(np.average(x)) for x in rec_class_test]
                self.my_dic_ablation_results['f1s_class_test_mean'] = [str(np.average(x)) for x in f1s_class_test]

    """ Method used to visulize the performance of the model after pruning."""
    def check_results(self, batch, itr):
        with torch.no_grad():
            # values chosen to have a visualization 
            # rate appropriate to the size of the test set
            if batch % 50 == 49:
                self.model.eval()
                # img[0], mask, pred[0], label[0] beacuse in 
                # case of batch-size > 1 prevent plot_check_results-crash
                (img, mask, label) = next(iter(self.test_loader))
                img = img.to(self.device)
                mask = mask.to(self.device)
                mask = mask[0]
                pred = self.model(img)

                self.writer.add_figure('ablation_check_results_' + self.t + '_' + str(itr), 
                                       plot_check_results(img[0], mask, pred[0], label[0], self.args), 
                                       global_step=len(self.train_loader) + batch)

    # debugging-methods
    ###########################################################################################################################

    """ Method used to plot CNN kernels. """
    def plot_kernels(self, tensor):
        # one kernel per column        
        num_cols=tensor.shape[0]
        num_kernels = tensor.shape[0]
        num_rows = 1

        fig = plt.figure(figsize=(num_cols, num_rows))

        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows, num_cols, i+1)
            ax1.imshow(tensor[i])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        # plt.show()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    """ Method used to plot CNN kernels. """
    def plot_weights_distribution(self, mod_name_list):
        for mod in mod_name_list:
            for module_name, module in self.model.named_modules():
                if(mod == module_name):
                    tensor = module.weight.detach()
                    
                    # uncomment the following line 
                    # if you want only the first 64 batch
                    if tensor.shape[0] > 64:
                        tensor = tensor[:64]
                    
                    tensor = tensor - tensor.min()
                    tensor = tensor / tensor.max()
                    
                    tensor = tensor.numpy()
                    
                    print(tensor.shape)
                    
                    self.plot_kernels(tensor)

                    break
        print('\n')

    ###########################################################################################################################
