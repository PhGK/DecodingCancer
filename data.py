import torch as tc
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from settings_tt import place
import random
from collections import Counter

np.random.seed(0)
class DataSimulator:
    def __init__(self):
        pass

    def simulate_sample(self, i):
        tc.manual_seed(i)
        therapy = tc.bernoulli(tc.ones(3)*0.5)
        diagnostics = tc.randn(10)

        sum1 = therapy[0] * diagnostics[0] + 2 * therapy[0] * diagnostics[1]
        sum2 = therapy[1] * diagnostics[2] if (diagnostics[2]> 0) else 0
        sum3 = therapy[1] * diagnostics[2] if (diagnostics[2]< -1.0) else 0
        sum4 = therapy[2] * diagnostics[3] * diagnostics[4]

        survival = sum1+sum2+sum3+sum4 + 0.01*tc.randn(1)
        return therapy, diagnostics, survival

    def simulate_data(self,n):
        therapy, diagnostics, survival = zip(*[self.simulate_sample(i) for i in range(n)])
        return tc.stack(therapy), tc.stack(diagnostics), tc.stack(survival)


def combine_data(setting, current_test_split, splits):

    if place == 'M':
        nsamples = 10000
        synthetic_data = pd.DataFrame(np.random.rand(nsamples,6)*2)

        synthetic_data.columns = ['somefeature1_' +str(col) for col in synthetic_data.columns[:5]] +  ['somefeature2_' + str(col) for col in synthetic_data.columns[5:]]
        synthetic_data=synthetic_data.sort_index(ascending=True)

        synthetic_data_rev = 1-synthetic_data.copy()
        synthetic_data_rev.columns = [str(col)+'_rev' for col in synthetic_data_rev.columns]
        full_data = pd.concat((synthetic_data, synthetic_data_rev), axis=1)
        full_data = synthetic_data.copy()
        random_numbers = np.random.rand(full_data.shape[0])
        full_data['Cancer_C0'] = (random_numbers<0.5) * 1.0
        full_data['Cancer_C1'] = (random_numbers>=0.5) * 1.0



        

        survival_data= pd.DataFrame({'duration': full_data.iloc[:,0]})


        #survival_data=survival_data.sort_index(ascending=True)

        survival_data['event'] = ((tc.rand_like(tc.tensor(survival_data['duration']))>0.2)*1.0).numpy()

        survival_data = survival_data.rename(columns = {setting['duration_name']: 'duration', setting['event_name']: 'event'})
        
        #######################

        cancer_type = pd.DataFrame(full_data[[str(col) for col in full_data.columns if str(col).startswith('Cancer_')]].idxmax(axis=1),
                      columns = ['cancer_type'], index = full_data.index)



    data_collection = Data_Collection(full_data, survival_data, cancer_type, current_test_split=current_test_split, splits=splits)

    return data_collection


class Data_Collection(Dataset):
    def __init__(self, full_data, survival_data, cancer_type, current_test_split, splits):
        self.splits = splits
        full_data = full_data.reindex(sorted(full_data.columns), axis=1)

        self.full_data = full_data
        self.survival_data = survival_data

        self.f_tensor = tc.tensor(full_data.to_numpy()).float()
        self.f_sample_names = np.array(full_data.index)
        self.f_feature_names = full_data.columns

        self.s_tensor = tc.tensor(survival_data.to_numpy()).float()
        self.s_sample_names = survival_data.index
        self.s_feature_names = survival_data.columns

        self.cancer_type = cancer_type
        print('cancer types:', self.cancer_type['cancer_type'])
        self.unique_cancer_types = self.cancer_type['cancer_type'].unique()

        self.nsamples = self.f_tensor.shape[0]
        self.f_nfeatures = self.f_tensor.shape[1]
        self.s_nfeatures = self.s_tensor.shape[1]

        tc.manual_seed(0)
        self.random_sequence = tc.randperm(self.nsamples)
        #self.test_splits = np.array_split(self.random_sequence, splits)

        #generate train and test ids for all splits
        self.test_splits = self.generate_stratified_test_sets(self.random_sequence, cancer_type['cancer_type'], splits)

        #select current training and test ids 
        self.change_test_set(current_test_split)

        self.unique_features_dict = self.make_dict(full_data)

        assert self.nsamples == self.s_tensor.shape[0], 'the input data have different sample lengths'
        assert all(self.f_sample_names == self.s_sample_names), 'samples (full, survival) have different names'


    def get_train_set(self):
        return self.f_tensor[self.training_ids,:], self.s_tensor[self.training_ids,:]

    def get_test_set(self):
        return self.f_tensor[self.test_ids,:], self.s_tensor[self.test_ids,:]
        

    def generate_stratified_test_sets(self, sequence, group_var, splits):
        unique_groups = group_var.unique()
        permuted_group_vars = np.array(group_var)[sequence]

        lists_of_group_ids = [sequence[permuted_group_vars==g] for g in unique_groups]
        lists_of_split_data = [np.array_split(group_seq, splits) for group_seq in lists_of_group_ids]

        random.seed(0)
        [random.shuffle(seq) for seq in lists_of_split_data]
        

        test_splits = []
        for i in range(splits):
            one_split = tc.cat([group_seq[i] for group_seq in lists_of_split_data])

            #permute one_split for fair test - val separation during training
            one_split_new = one_split[tc.randperm(one_split.shape[0])]

            
            test_splits.append(one_split_new)
        
        for test_ids in test_splits:
            print(Counter(list(np.array(group_var)[test_ids])))

        return test_splits

        

    def change_test_set(self, new_test_split):
        self.current_test_split = new_test_split
        #self.test_ids = self.random_sequence[self.test_splits[new_test_split]] #doppelt gemoppelt?
        self.test_ids = self.test_splits[new_test_split] #besser?

        self.training_ids = np.setdiff1d(self.random_sequence, self.test_ids, assume_unique=True)

        self.train_len = self.training_ids.shape[0]
        self.test_len = self.test_ids.shape[0]


    def get_test_names(self):
        return self.f_sample_names[self.test_ids]

    def get_train_names(self):
        return self.f_sample_names[self.training_ids]
        

    def get_cancer_types_test(self):
        return np.array(self.cancer_type)[self.test_ids].squeeze()

    def get_cancer_types_train(self):
        return np.array(self.cancer_type)[self.training_ids].squeeze()

    def get_input_long(self):
       full_data = self.full_data
       full_data['sample_name'] = full_data.index
       return pd.melt(full_data, id_vars = 'sample_name', var_name = 'therapy_diagnostics', value_name = 'input_score')


    def make_dict(self, dat):
        featureclass = [col.split('_')[0] for col in dat.columns]
        counts = np.unique(featureclass, return_counts=True)
        count_dict = {counts[0][i]:counts[1][i] for i,_ in enumerate(counts[0])}
        return count_dict



        


