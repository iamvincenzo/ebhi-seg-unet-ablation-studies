#########################################
# BEST NETWORK-CONFIGURATION SELECTION #
########################################

import os
import json
import datetime
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import bar_plotting, area_plotting, line_plotting #, set_default


""" Helper function used to get cmd parameters. """
def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    ###################################################################
    parser.add_argument('--model_name', type=str, default="unet_final_2t",
                        help='name of the model to be saved/loaded')
    parser.add_argument('--result_analysis_name', type=str, default="unet_final_2t_abl_results",
                        help='to do ???')
    ###################################################################

    # best-configuration
    ###################################################################
    parser.add_argument('--get_best_net_config', action='store_true', # default=True,
                        help='get the best network configuration')
    parser.add_argument('--num_conf_per_it', type=int, default=3,
                        help='number of json files per run')
    parser.add_argument('--train_statistics_path', type=str,
                        default='./statistics', help='path were to get model statistics')
    ###################################################################
        
    # ablation
    ###################################################################
    parser.add_argument('--compare_ablation_results', action='store_true', default=True,
                        help='compare ablation results')    
    parser.add_argument('--abl_statistics_path', type=str,
                        default='./abl_statistics', help='path were to get model statistics')
    parser.add_argument('--save_imgs_path', type=str,
                        default='./data/abl_results_images/', help='path were to get model statistics')
    ###################################################################
    
    
    return parser.parse_args()


""" Helper function used to get files-name. """
def get_files_name(path):
    files = glob(path)

    return files


""" Helper function used to get the best network configuration. """
def get_best_net_config(args):
    print('\n\nGetting the best network configuration...')
    path = args.train_statistics_path + '/*.json'
    files = get_files_name(path)
    files = [f for f in files if 'ablation' not in f]

    N = args.num_conf_per_it  # number of iteration per configuration
    idx = 0
    mean_test_losses = []
    mean_test_accs = []
    mean_test_precs = []
    mean_test_recs = []

    for i in range(0, len(files), N):
        np_array_accs_list = []
        np_array_precs_list = []
        np_array_recs_list = []
        sum_train = 0.0
        sum_test = 0.0

        for j in range(N):
            with open(files[i + j]) as f:
                my_dict = json.load(f)
                my_dict['name'] = files[i + j]

                # statistics calculation: test_loss & train_loss
                sum_train += float(my_dict['avg_train_losses'][-1])
                sum_test += float(my_dict['avg_test_losses'][-1])

                # statistics calculation: accuracy
                l0 = np.array(my_dict['acc_class_test_mean'])
                l0 = l0.astype(np.float32)
                np_array_accs_list.append(l0)

                # statistics calculation: precision
                l1 = np.array(my_dict['prec_class_test_mean'])
                l1 = l1.astype(np.float32)
                np_array_precs_list.append(l1)

                # statistics calculation: recall
                l2 = np.array(my_dict['rec_class_test_mean'])
                l2 = l2.astype(np.float32)
                np_array_recs_list.append(l2)

        """ The name is only representative: at the end you have to choose between 
            configurations with the same name with the appropriate random seed. """
        mean_test_losses.append((my_dict['name'], sum_test/N, sum_train/N, idx))

        """ The final per-class-accuracy is computed as the element-wise-mean 
            of N list (corresponding to the N configurations) of per-class-accuracy. 
            Same operation for the per-class-precision and per-class-recall. """ 
        mean_test_accs.append(np.mean(np_array_accs_list, axis=0))
        mean_test_precs.append(np.mean(np_array_precs_list, axis=0))
        mean_test_recs.append(np.mean(np_array_recs_list, axis=0))

        # ITA:
        # l'inserimento dei dati nelle liste mean_test_losses, mean_test_accs, mean_test_precs e mean_test_recs
        # avviene in maniera sequenziale: quindi all'item alla posizione i-esima sono associati una certa
        # loss, per-class-accuracy-list, per-class-precision-list, per-class-recall-list. Tuttavia, al termine, 
        # la lista delle loss viene ordinata al fine di determinare la configurazione migliore. 
        # Ma, dopo l'ordinamento, non c'è più corrispondenza tra la posizione dell'item nella lista delle loss
        # e la sua posizione nelle altre liste. Quindi, prima di ordinare la lista, ad ogni item si associa il
        # vecchio id-sequenziale che viene poi utilizzato per reperire le liste corrispondenti.    

        idx += 1 # id is used to get the corresponding item in the various lists of accuracy, precision, recall

    """ After sorting there is no longer a correspondence between the 
        position of the item and the corresponding values in the lists
        of accuracy, precision and recall. """
    mean_test_losses.sort(key=lambda tup: tup[1])

    return mean_test_losses, mean_test_accs, mean_test_precs, mean_test_recs


""" Helper function used to ??? """
def compare_ablation_results(args, writer):
    print('\n\nGetting ablation studies statistics results...\n')
    
    path = args.abl_statistics_path + '/*.json'
    files = get_files_name(path)
    abl_files = [f for f in files if 'ablation' in f] #and args.model_name in f]
    train_file = [f for f in files if 'ablation' not in f and args.model_name in f]

    my_dict_train = {}
    my_dict_abl = {}

    # Training collected results
    with open(train_file[0]) as f:
        my_dict_train = json.load(f)

    j = 1
    types = ['global_ablation', 'glob_ablation_grouped', 
            'selective_ablation_single_mod', 'selective_ablation_double_mod', 'all_one_by_one']
    # Ablation collected results
    for file in abl_files:
        with open(file) as f:
            my_dict_abl = json.load(f)

            """ Statistics calculation
            print('Test-losses differences:')
            # if my_dict_abl_loss < my_dict_train_loss --> diff < 0 (good)
            # if my_dict_abl_loss > my_dict_train_loss --> diff > 0 (not good)
            # my_dict_abl_loss == my_dict_train_loss --> diff == 0 
            diff = float(my_dict_abl['avg_test_losses']) - float(my_dict_train['avg_test_losses'][-1])
            var1 = float(my_dict_train['avg_test_losses'][-1])
            var2 = float(my_dict_abl['avg_test_losses'])
            print(f'my_dict_abl: {var2:.4f}, my_dict_train: {var1:.4f}, abs_diff: {diff}')
            
            print('\nAccuracy differences:')
            """

            metrics = ['acc_class_test_mean'] #, 'prec_class_test_mean', 'rec_class_test_mean']

            exp = ""
            for t in types:
                if t in file:
                    exp = t
            
            for metric in metrics:
                l0 = np.array(my_dict_train[metric])
                l0 = l0.astype(np.float32)
                l1 = np.array(my_dict_abl[metric])
                l1 = l1.astype(np.float32)
                l2 = np.absolute(l0 - l1)

                title = list(my_dict_abl.keys())[0]
                val = my_dict_abl[title]
                title1 = title + " = " + val
 
                ax = bar_plotting(l0, l1, l2, metric)
                plt.title(title1)
                # plt.show()
                plt.draw()
                plt.savefig(args.save_imgs_path + str(j) + '_' + exp + '_' + title + '_bp.png', bbox_inches='tight', dpi=1000)      
                plt.close()  # close the figure when done         

                ax = area_plotting(l0, l1, l2, metric, False)
                plt.title(title1)
                # plt.show()
                plt.draw()
                plt.savefig(args.save_imgs_path + str(j) + '_' + exp + '_' + title + '_ap.png', bbox_inches='tight', dpi=1000)
                plt.close()  # close the figure when done

                ax = line_plotting(l0, l1, metric)
                plt.title(title1)
                # plt.show()
                plt.draw()
                plt.savefig(args.save_imgs_path + str(j) + '_' + exp + '_' + title + '_lp.png', bbox_inches='tight', dpi=1000)
                plt.close()  # close the figure when done

                j += 1


        # print('\n')


def main(args):
    if args.get_best_net_config == True:
        mean_test_losses, mean_test_accs, mean_test_precs, mean_test_recs = get_best_net_config(args)
        for idx, val in enumerate(mean_test_losses):
            print(f'\n{idx+1}) Configuration: {val[0]}, \n \
                mean-test-loss: {val[1]:.4f}, mean-train-loss: {val[2]:.4f}, abs-diff: {np.abs(val[1] - val[2]):.4f} \n \
                mean-test per-class-accuracy: {mean_test_accs[val[3]]}, total: {np.mean(mean_test_accs[val[3]]):.4f} \n \
                mean-test per-class-precision: {mean_test_precs[val[3]]}, total: {np.mean(mean_test_precs[val[3]]):.4f} \n \
                mean-test per-class-recall: {mean_test_recs[val[3]]}, total: {np.mean(mean_test_recs[val[3]]):.4f}\n')
    
    elif args.compare_ablation_results == True:
         # tensorboard specifications
        log_folder = './runs/' + args.result_analysis_name + '_' + \
            datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
        writer = SummaryWriter(log_folder)
        compare_ablation_results(args, writer)


if __name__ == "__main__":
    # set_default()
    args = get_args()
    print(f'\n{args}')
    if not os.path.isdir(args.save_imgs_path):
        os.makedirs(args.save_imgs_path)
    main(args)    
        
    
""" The selection of the best configuration is based on:
        - mean-test-loss;
        - difference in absolute value between mean-test-loss and mean-train-loss;
        - mean-accuracy, mean-precision, mean-recall.
        
    Output of the top-three best-configurations:

        >  1) Configuration: unet_3_0_2t,
              mean-test-loss: 0.0536, mean-train-loss: 0.0408, abs-diff: 0.0128 
              mean-test per-class-accuracy: [0.9623189  0.96565896 0.9605729  0.9154892  0.89932853 0.9408622 ], total: 0.9407 
              mean-test per-class-precision: [0.96945447 0.9783523  0.97588116 0.9515314  0.93830997 0.9434865 ], total: 0.9595 
              mean-test per-class-recall: [0.9676378  0.9580894  0.96404046 0.9274319  0.91293246 0.9604456 ], total: 0.9484

        >  2) Configuration: unet_13_0_2t, 
              mean-test-loss: 0.0560, mean-train-loss: 0.0466, abs-diff: 0.0095 
              mean-test per-class-accuracy: [0.9597576  0.9654646  0.95879984 0.90839267 0.8935342  0.9448583 ], total: 0.9385 
              mean-test per-class-precision: [0.9671772  0.9744418  0.974854   0.9533973  0.9409496  0.95258427], total: 0.9606 
              mean-test per-class-recall: [0.96462065 0.96160406 0.9628343  0.9156433  0.9021184  0.95946884], total: 0.9444

        >  3) Configuration: unet_4_0_2t, 
              mean-test-loss: 0.0576, mean-train-loss: 0.0526, abs-diff: 0.0050 
              mean-test per-class-accuracy: [0.9571473  0.96196645 0.95727676 0.9050472  0.89431095 0.92810273], total: 0.9340
              mean-test per-class-precision: [0.9668333 0.9768634 0.9731913 0.9419367 0.9312933 0.9322795], total: 0.9537
              mean-test per-class-recall: [0.9624381  0.9533414  0.9623093  0.92224884 0.91217774 0.95272416], total: 0.9442
    
    The selected best-configuration is the 2nd. """
