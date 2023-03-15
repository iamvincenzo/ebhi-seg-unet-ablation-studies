import json
import numpy as np
from glob import glob

######################################################################
# THE BEST NETWROK-CONFIGURATION IS SELECTED BASED ON THE TEST-LOSS #
#####################################################################

""" Helper function used to get files-name """
def get_files_name(path):
    files = glob(path)
    return files


""" Helper function used to get 
    the best network configuration. """
def main(path):
    files = get_files_name(path)

    N = 3 # number of iteration per configuration
    mean_test_losses = []
    mean_test_accs = []    

    for i in range(0, len(files), N):
        np_array_list = []
        sum = 0.0
        for j in range(N):
            with open(files[i + j]) as f:
                my_dict = json.load(f)
                my_dict['name'] = files[i + j]
                
                # statistics calculation: test_loss
                sum += float(my_dict['avg_test_losses'][-1])

                l0 = np.array(my_dict['acc_class_test_mean'])
                l0 = l0.astype(np.float32)
                np_array_list.append(l0)       

        """ The name is only representative: at the end you have to choose between 
            configurations with the same name with the appropriate random seed. """ 
        mean_test_losses.append((my_dict['name'], sum/3))  
        mean_test_accs.append(np.mean(np_array_list, axis=0))


    """ Selecting the best-network-configuration: 
        based on test_loss. """
    best_loss = 10
    best_item = ()
    best_idx = 0
    for idx, item in enumerate(mean_test_losses):        
        if (item[1] < best_loss):
            best_item = item
            best_loss = item[1]
            best_idx = idx


    return best_item, mean_test_accs[best_idx]
    
    

if __name__ == "__main__":
    path = './statistics/*.json'
    best_item, mean_test_accs = main(path)

    print('\nBEST-RESULTS: \n')
    print(best_item)
    print(mean_test_accs, '\n')
