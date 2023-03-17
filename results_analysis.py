import json
import numpy as np
from glob import glob

#########################################
# BEST NETWORK-CONFIGURATION SELECTION #
########################################

""" Helper function used to get files-name. """
def get_files_name(path):
    files = glob(path)

    return files


""" Helper function used to get 
    the best network configuration. """
def main(path):
    files = get_files_name(path)

    N = 3  # number of iteration per configuration
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


if __name__ == "__main__":
    path = './statistics/*.json'
    mean_test_losses, mean_test_accs, mean_test_precs, mean_test_recs = main(path)

    for idx, val in enumerate(mean_test_losses):
        print(f'\n{idx+1}) Configuration: {val[0]}, \n \
              mean-test-loss: {val[1]:.4f}, mean-train-loss: {val[2]:.4f}, abs-diff: {np.abs(val[1] - val[2]):.4f} \n \
              mean-test per-class-accuracy: {mean_test_accs[val[3]]}, total: {np.mean(mean_test_accs[val[3]]):.4f} \n \
              mean-test per-class-precision: {mean_test_precs[val[3]]}, total: {np.mean(mean_test_precs[val[3]]):.4f} \n \
              mean-test per-class-recall: {mean_test_recs[val[3]]}, total: {np.mean(mean_test_recs[val[3]]):.4f}\n')
        
    
    """ The selection of the best configuration is based on:
            - mean-test-loss;
            - difference in absolute value between mean-test-loss and mean-train-loss;
            - mean-accuracy, mean-precision, mean-recall.
            
        Output of the top-three best-configurations:

            >   1) Configuration: unet_3_0_2t,
                mean-test-loss: 0.0536, mean-train-loss: 0.0408, abs-diff: 0.0128 
                mean-test per-class-accuracy: [0.9623189  0.96565896 0.9605729  0.9154892  0.89932853 0.9408622 ], total: 0.9407 
                mean-test per-class-precision: [0.96945447 0.9783523  0.97588116 0.9515314  0.93830997 0.9434865 ], total: 0.9595 
                mean-test per-class-recall: [0.9676378  0.9580894  0.96404046 0.9274319  0.91293246 0.9604456 ], total: 0.9484

            >  2) Configuration: unet_10_1_2t, 
               mean-test-loss: 0.0537, mean-train-loss: 0.0507, abs-diff: 0.0030 
               mean-test per-class-accuracy: [0.9607262  0.96570706 0.9593335  0.9279091  0.9061057  0.9376357 ], total: 0.9429 
               mean-test per-class-precision: [0.9698933  0.97131944 0.9687198  0.94207096 0.91854763 0.932326  ], total: 0.9505 
               mean-test per-class-recall: [0.9625911  0.9608024  0.9652941  0.94738054 0.93374807 0.9594771 ], total: 0.9549

            >  3) Configuration: unet_13_0_2t, 
               mean-test-loss: 0.0576, mean-train-loss: 0.0482, abs-diff: 0.0094 
               mean-test per-class-accuracy: [0.95846385 0.96584886 0.95859265 0.9265111  0.8954797  0.9469705 ], total: 0.9420 
               mean-test per-class-precision: [0.96732664 0.9744399  0.97126913 0.94663805 0.9108305  0.95278716], total: 0.9539 
               mean-test per-class-recall: [0.95924073 0.9580772  0.9619231  0.9409527  0.92701036 0.95817775], total: 0.9509
        
        The selected best-configuration is the 2nd. """
