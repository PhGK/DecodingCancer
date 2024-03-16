import torch as tc
from data import combine_data
from settings_tt import setting
from training import train_test
from NN import Simple_Model
import os
import pandas as pd
from LRP import calculate_LRP_simple
from crossant import crossvalidate


def main():
    if not os.path.exists('./results/data/'):
        os.makedirs('./results/data/')
    data_coll = combine_data(setting, current_test_split = 0, splits = 5)

    ################
    #construct model
    ################

    model = Simple_Model(data_coll, setting)
    print(model)

    #################
    #crossvalidate
    #################
    crossvalidate(model, data_coll, setting, train_on_all=True, train_on_parts=True)



if __name__ == '__main__':
    print('lets go')
    main()
    
