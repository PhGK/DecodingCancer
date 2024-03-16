from torch.utils.data import DataLoader
import torch as tc
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from NN import Simple_Model



def reverse_feature_expansion(frame):

    rev_features = frame[frame['therapy_diagnostics'].str.endswith('_rev')] 
    rev_features['therapy_diagnostics_antero'] = rev_features['therapy_diagnostics'].str.split('_rev').str[0].copy()
    rev_features = rev_features.rename(columns = {'LRP': 'LRP_rev', 'input_score':'input_score_rev'})
    antero_features = frame[~frame['therapy_diagnostics'].str.endswith('_rev')] 
    antero_features = antero_features.rename(columns = {'LRP': 'LRP_antero', 'therapy_diagnostics': 'therapy_diagnostics_antero', 'input_score':'input_score_antero'})

    frame_unexpanded = antero_features.merge(rev_features[['therapy_diagnostics_antero', 'LRP_rev', 'sample_name', 'input_score_rev']], 
                                                                how='left', on = ['therapy_diagnostics_antero', 'sample_name'])

    frame_unexpanded['LRP'] = frame_unexpanded[['LRP_antero', 'LRP_rev']].sum(axis=1, skipna=True)
    return frame_unexpanded


def np2pd(nparray, sample_names, feature_names):

    frame = pd.DataFrame(nparray[:,:], index = np.array(sample_names),columns = feature_names)
    frame['sample_name'] = sample_names
    long_frame = pd.melt(frame, id_vars = 'sample_name', var_name = 'therapy_diagnostics', value_name = 'LRP')
    
    return long_frame.reset_index(drop=True)


def calculate_LRP_simple(model, data_collection, setting, PATH = './results/LRP/', fold=None):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    device = tc.device(setting['LRP_device'])

    # this is really test+val set
    patdata_test, surv_test = data_collection.get_test_set()
    model.eval().to(device)

    patdata_test, surv_test = patdata_test.to(device), surv_test.to(device)

    R = model(patdata_test)

    sample_names = data_collection.get_test_names()
    predictions = pd.DataFrame({'sample_name': sample_names, 'prediction': R.clone().detach().cpu().numpy().squeeze()})

    input_relevance = model.relprop(R).cpu().detach().numpy()
    print(R.shape)
    LRP_scores_long = np2pd(input_relevance, sample_names, data_collection.f_feature_names)
    input_long = data_collection.get_input_long()

    LRP_scores_long_and_inputs = LRP_scores_long.merge(input_long, on = ['sample_name', 'therapy_diagnostics'], how = 'left')

    LRP_scores_long_and_inputs_unexpanded = reverse_feature_expansion(LRP_scores_long_and_inputs)

    risk_prediction = pd.DataFrame({'sample_name': sample_names,
                                    'risk_prediction_all': R.clone().detach().cpu().numpy().squeeze()})
    LRP_scores_long_and_inputs_unexpanded = LRP_scores_long_and_inputs_unexpanded.merge(risk_prediction, how = 'left')

    LRP_scores_long_and_inputs_unexpanded.to_csv(PATH + 'LRP_'+ str(model.classname) + '_scores_input_' + str(fold) + '.csv')

