##############################################################
# https://leimao.github.io/article/Neural-Networks-Pruning/ #
#############################################################

######################################################
# https://github.com/leimao/PyTorch-Pruning-Example #
#####################################################

import os
import json
import torch
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
# from matplotlib.lines import Line2D
import torch.nn.utils.prune as prune
import torchvision.transforms.functional as TF
from plotting_utils import plot_check_results
from metrics import jac_loss, binary_jac, binary_acc, binary_prec, binary_rec, binary_f1s



class AblationStudies(object):
    def __init__(self, args, model_name, train_loader, test_loader, model, criterion, device):
        super(AblationStudies, self).__init__()

        self.args = args
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.device = device


    def measure_module_sparsity(self, module, weight=True, bias=False, use_mask=False):
        num_zeros = 0
        num_elements = 0

        if use_mask == True:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight == True:
                    num_zeros += torch.sum(buffer == 0).item()
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


    def iterative_pruning_finetuning(self, num_epochs_per_iteration=10, grouped_pruning=False):
        for i in range(self.args.num_iterations):

            print('Pruning and Finetuning {}/{}'.format(i + 1, self.args.num_iterations))

            print('\nPruning...')

            if grouped_pruning == True:
                # Global pruning
                # I would rather call it grouped pruning.
                parameters_to_prune = []
                for _, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        parameters_to_prune.append((module, 'weight'))
                
                # select a pruning technique
                prune.global_unstructured(parameters_to_prune,
                                          pruning_method=prune.L1Unstructured,
                                          amount=self.args.conv2d_prune_amount)
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
                        
            # test the model
            self.test()


            num_zeros, num_elements, sparsity = self.measure_global_sparsity(weight=True, bias=False,
                                                                             conv2d_use_mask=True, linear_use_mask=False)
            
            print(f'Global Sparsity: {sparsity:.2f}\n')

            """
            # print(model.conv1._forward_pre_hooks)

            print("\nFine-tuning...")

            # retrain model

            train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        learning_rate=learning_rate * (learning_rate_decay**i),
                        num_epochs=num_epochs_per_iteration)

            _, eval_accuracy = evaluate_model(model=model,
                                            test_loader=test_loader,
                                            device=device,
                                            criterion=None)

            classification_report = create_classification_report(
                model=model, test_loader=test_loader, device=device)

            num_zeros, num_elements, sparsity = measure_global_sparsity(
                model,
                weight=True,
                bias=False,
                conv2d_use_mask=True,
                linear_use_mask=False)

            print("Test Accuracy: {:.3f}".format(eval_accuracy))
            print("Classification Report:")
            print(classification_report)
            print("Global Sparsity:")
            print("{:.2f}".format(sparsity))

            model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
            model_filepath = os.path.join(model_dir, model_filename)
            save_model(model=model,
                    model_dir=model_dir,
                    model_filename=model_filename)
            model = load_model(model=model,
                            model_filepath=model_filepath,
                            device=device)

        return model
        """


    def selective_pruning(self, mod_name_list=['downs.0.conv.0'], # , 'downs.0.conv.3' 
                          num_epochs_per_iteration=10, grouped_pruning=False):
        
        for i in range(self.args.num_iterations):

            print('Selective Pruning and Finetuning {}/{}'.format(i + 1, self.args.num_iterations))

            print('\nPruning...')

            for module_name, module in self.model.named_modules():
                for mod_name in mod_name_list:
                    if mod_name == module_name and isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, 
                                            name='weight', 
                                            amount=self.args.conv2d_prune_amount)
                        
            # test the model
            self.test()       

            # num_zeros, num_elements, sparsity = self.measure_global_sparsity(weight=True, bias=False,
                                                                             # conv2d_use_mask=True, linear_use_mask=False)
            
            # print(f'Global Sparsity: {sparsity:.2f}\n')

            # retrain model + ecc. ecc.
            # to do


    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf


    def test(self):
        print(f'\nPerforming validation-test after pruning...\n')

        test_losses = []
        my_dic_ablation_results = {}

        self.model.eval()  # put net into evaluation mode

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

            for batch_test, (test_images, test_targets, test_labels) in test_loop:
                test_images = test_images.to(self.device)
                test_targets = test_targets.to(self.device)
                test_pred = self.model(test_images.detach())

                # general loss
                test_loss = self.criterion(test_pred, test_targets).item()
                test_losses.append(test_loss)

                # used to check model improvement
                self.check_results(batch_test)

                if self.args.bs_test == 1:
                    # function that binarize an image and then convert it to a tensor
                    maskf, predf = self.binarization_tensor(test_targets, test_pred)

                    # per class metrics
                    dc_class_test[test_labels].append(1 - test_loss)
                    jac_cust_class_test[test_labels].append(1 - jac_loss(test_pred, test_targets).item())
                    jac_class_test[test_labels].append(binary_jac(maskf, predf))
                    acc_class_test[test_labels].append(binary_acc(maskf, predf))
                    prec_class_test[test_labels].append(binary_prec(maskf, predf))
                    rec_class_test[test_labels].append(binary_rec(maskf, predf))
                    f1s_class_test[test_labels].append(binary_f1s(maskf, predf))

            batch_avg_test_loss = np.average(test_losses)

            """ Saving some data in a dictionary (1). """                    
            my_dic_ablation_results['avg_test_losses'] = str(batch_avg_test_loss)

            print(f'\nvalid_loss: {batch_avg_test_loss:.5f}\n')

            if self.args.bs_test == 1:
                print(f'Dice for class {[np.average(x) for x in dc_class_test]}')
                print(f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                print(f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                print(f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                print(f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                print(f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                print(f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

                """ Saving some data in a dictionary (2). """                    
                my_dic_ablation_results['dc_class_test_mean'] = [str(np.average(x)) for x in dc_class_test]
                my_dic_ablation_results['jac_cust_class_test_mean'] = [str(np.average(x)) for x in jac_cust_class_test]
                my_dic_ablation_results['jac_class_test_mean'] = [str(np.average(x)) for x in jac_class_test]
                my_dic_ablation_results['acc_class_test_mean'] = [str(np.average(x)) for x in acc_class_test]
                my_dic_ablation_results['prec_class_test_mean'] = [str(np.average(x)) for x in prec_class_test]
                my_dic_ablation_results['rec_class_test_mean'] = [str(np.average(x)) for x in rec_class_test]
                my_dic_ablation_results['f1s_class_test_mean'] = [str(np.average(x)) for x in f1s_class_test]

            
            check_path = os.path.join(self.args.checkpoint_path, 'pruned_' + self.model_name)
            torch.save(self.model.state_dict(), check_path)
            print('\nModel saved!\n')

            """ Saving some statistics. """
            with open('./statistics/my_dic_ablation_results_' + self.args.model_name + 
                      '_' + datetime.datetime.now().strftime('%d%m%Y-%H%M%S') + '.json', 'w') as f:
                json.dump(my_dic_ablation_results, f)

            # print(f'Different conv-filters removed from each conv-layer: \n\n')
            # for x in self.removed_info:
            #     print(x)

        # self.model.train()  # put again the model in trainining-mode


    def check_results(self, batch):
        with torch.no_grad():
            if batch % 50 == 49:
                self.model.eval()

                (img, mask, label) = next(iter(self.test_loader))
                img = img.to(self.device)
                mask = mask.to(self.device)
                mask = mask[0]
                pred = self.model(img)

                fig = plot_check_results(
                    img[0], mask, pred[0], label[0], self.args)

                plt.show(block=False)
                plt.pause(2) #(10)
                plt.close()


###########################################################################################################################

    """
    def add_gradient_hist_mod(self, net):
        ave_grads = []
        layers = []
        for n, p in net.named_parameters():
            if ('bias' not in n):
                layers.append(n)
                if p.requires_grad:  # indicates whether a variable is trainable
                    ave_grad = np.abs(p.detach().numpy()).mean()
                else:
                    ave_grad = 0
                ave_grads.append(ave_grad)

        layers = [layers[i].replace(".weight", "") for i in range(len(layers))]

        fig = plt.figure(figsize=(12, 12))
        plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.legend([Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        return fig
    """

    
###########################################################################################################################