#########################################
# BEST NETWORK-CONFIGURATION SELECTION #
########################################

import os
import json
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from plotting_utils import bar_plotting, area_plotting
# , line_plotting, set_default


""" Helper function used to get cmd parameters. """
def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    ###################################################################
    parser.add_argument('--model_name', type=str, default="unet_final_2t",
                        help='name of the model to be saved/loaded')
    ###################################################################

    # best-configuration
    ###################################################################
    parser.add_argument('--get_best_net_config', action='store_true', default=True,
                        help='get the best network configuration')
    parser.add_argument('--num_conf_per_it', type=int, default=3,
                        help='number of json files per run')
    parser.add_argument('--train_statistics_path', type=str,
                        default='./statistics', help='path were to get model statistics')
    ###################################################################
        
    # ablation
    ###################################################################
    parser.add_argument('--compare_ablation_results', action='store_true', # default=True,
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

    # number of iteration per configuration
    N = args.num_conf_per_it 
    idx = 0
    mean_test_losses = []
    mean_test_dcs = []
    mean_test_jacs = []
    mean_test_accs = []
    mean_test_precs = []
    mean_test_recs = []

    # calcolo delle prestazioni medie dei len(files)/N esperimenti differenti
    for i in range(0, len(files), N):
        np_array_dcs_list = []
        np_array_jacs_list = []
        np_array_accs_list = []
        np_array_precs_list = []
        np_array_recs_list = []
        sum_train = 0.0
        sum_test = 0.0

        # per ogni simulazione relativa allo stesso esperimento
        for j in range(N):
            with open(files[i + j]) as f:
                my_dict = json.load(f)
                my_dict['name'] = files[i + j]

                # statistics calculation: test_loss & train_loss:
                # so considerano solo gli utlimi valori assunti dall loss
                # in training e testing
                sum_train += float(my_dict['avg_train_losses'][-1])
                sum_test += float(my_dict['avg_test_losses'][-1])

                # statistics calculation: accuracy
                l0 = np.array(my_dict['acc_class_test_mean'])
                l0 = l0.astype(np.float32)
                # [np.([..., .., ...]),  ..., N]
                np_array_accs_list.append(l0)

                # statistics calculation: precision
                l1 = np.array(my_dict['prec_class_test_mean'])
                l1 = l1.astype(np.float32)
                np_array_precs_list.append(l1)

                # statistics calculation: recall
                l2 = np.array(my_dict['rec_class_test_mean'])
                l2 = l2.astype(np.float32)
                np_array_recs_list.append(l2)

                # statistics calculation: dice
                l3 = np.array(my_dict['dc_class_test_mean'])
                l3 = l3.astype(np.float32)
                np_array_dcs_list.append(l3)

                # statistics calculation: iou
                l4 = np.array(my_dict['jac_class_test_mean'])
                l4 = l4.astype(np.float32)
                np_array_jacs_list.append(l4)

        """ The name is only representative: at the end you have to choose 
            between configurations with the same name with the appropriate random seed. """
        # best configuration related to the capability 
        # of generalization of the model (validation-loss/train-loss)
        mean_test_losses.append((my_dict['name'], sum_test/N, sum_train/N, idx))

        """ The final per-class-accuracy is computed as the element-wise-mean 
            of N list (corresponding to the N configurations) of per-class-accuracy. 
            Same operation for the per-class-precision and per-class-recall. """
        mean_test_dcs.append(np.mean(np_array_dcs_list, axis=0))
        mean_test_jacs.append(np.mean(np_array_jacs_list, axis=0))
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

    return mean_test_losses, mean_test_dcs, mean_test_jacs, mean_test_accs, mean_test_precs, mean_test_recs

""" Helper function used to compare ablation results. """
def compare_ablation_results(args):
    print('\n\nGetting ablation studies statistics results...\n')
    
    path = args.abl_statistics_path + '/*.json'
    files = get_files_name(path)
    abl_files = [f for f in files if 'ablation' in f]
    train_file = [f for f in files if 'ablation' not in f and args.model_name in f]

    my_dict_train = {}
    my_dict_abl = {}

    # training collected results
    with open(train_file[0]) as f:
        my_dict_train = json.load(f)

    metrics = ['acc_class_test_mean'] #, 'prec_class_test_mean', 'rec_class_test_mean']

    # ('file-name', 'op-name', drop, accuracy)
    drops_max_per_class = [('h', 'h', 0, 0), ('h', 'h', 0, 0), ('h', 'h', 0, 0), 
                           ('h', 'h', 0, 0), ('h', 'h', 0, 0), ('h', 'h', 0, 0)]
    
    drop_max_per = 'acc_class_test_mean' # metric used for selection

    j = 1
    types = ['global_ablation', 'glob_ablation_grouped', 'a_o_b_o', 'al_ol_bl_ol',
             'selective_ablation_single_mod', 'selective_ablation_double_mod', 'all_one_by_one']
    
    # Computing drop-max in accuracy per class
    for file in abl_files:
        with open(file) as f:
            my_dict_abl = json.load(f)

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

                # computing drop-max (worst) per class (accuracy)
                if metric == drop_max_per:
                    for i in range(len(drops_max_per_class)):
                        if drops_max_per_class[i][2] < l2[i]:
                            drops_max_per_class[i] = (file, title, l2[i], l1[i])

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

                """
                ax = line_plotting(l0, l1, metric)
                plt.title(title1)
                # plt.show()
                plt.draw()
                plt.savefig(args.save_imgs_path + str(j) + '_' + exp + '_' + title + '_lp.png', bbox_inches='tight', dpi=1000)
                plt.close()  # close the figure when done
                """

                j += 1
    
    for _, drp in enumerate(drops_max_per_class):
        print(drp)
    

    """ print (graphs) only the worst 
    # Computing graphs for each drop-max in accuracy per class
    for (file, _, _, _) in drops_max_per_class:
        with open(file) as f:
            my_dict_abl = json.load(f)
            
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

                # ax = line_plotting(l0, l1, metric)
                # plt.title(title1)
                # # plt.show()
                # plt.draw()
                # plt.savefig(args.save_imgs_path + str(j) + '_' + exp + '_' + title + '_lp.png', bbox_inches='tight', dpi=1000)
                # plt.close()  # close the figure when done

                j += 1           
    """

""" Helper function used to run the simulation. """
def main(args):
    if args.get_best_net_config == True:
        (mean_test_losses, mean_test_dcs, mean_test_jacs, 
         mean_test_accs, mean_test_precs, mean_test_recs) = get_best_net_config(args)
        for idx, val in enumerate(mean_test_losses):
            print(f'\n{idx+1}) Configuration: {val[0]}, \n'
                  f'mean-test-loss: {val[1]:.4f}, mean-train-loss: {val[2]:.4f}, abs-diff: {np.abs(val[1] - val[2]):.4f} \n'
                  f'mean-test per-class-dice: {mean_test_dcs[val[3]]}, total: {np.mean(mean_test_dcs[val[3]]):.4f} \n'
                  f'mean-test per-class-iou: {mean_test_jacs[val[3]]}, total: {np.mean(mean_test_jacs[val[3]]):.4f} \n'
                  f'mean-test per-class-accuracy: {mean_test_accs[val[3]]}, total: {np.mean(mean_test_accs[val[3]]):.4f} \n'
                  f'mean-test per-class-precision: {mean_test_precs[val[3]]}, total: {np.mean(mean_test_precs[val[3]]):.4f} \n'
                  f'mean-test per-class-recall: {mean_test_recs[val[3]]}, total: {np.mean(mean_test_recs[val[3]]):.4f}\n')
    
    elif args.compare_ablation_results == True:
        compare_ablation_results(args)

""" Starting the simulation. """
if __name__ == "__main__":
    # set_default()
    args = get_args()    
    print(f'\n{args}')

    if not os.path.isdir(args.save_imgs_path):
        os.makedirs(args.save_imgs_path)

    main(args)
