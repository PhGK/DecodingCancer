from training import train_test, train_test_individual_cancers, save_predictions
from LRP import calculate_LRP_simple
import pandas as pd
import numpy as np
import copy
import torch as tc


def crossvalidate(model, data_coll, setting, train_on_all=True, train_on_parts=True):
    tc.save(model.state_dict(), './results/raw_params.pt')

    for i in range(data_coll.splits):
        data_coll.change_test_set(i)
        
        if train_on_all:
            model = train_test(model, data_coll, setting, fold=i)
            save_predictions(model, data_coll, setting, fold=i)
        
            if model.classname == 'Simple Model':
                calculate_LRP_simple(model, data_coll, setting, fold=i)

        
        if train_on_parts:
            train_test_individual_cancers(model, data_coll, setting, fold=i)

