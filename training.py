from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
import torch.nn as nn
import torch as tc
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchtuples as tt
import numpy as np
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import os
import pandas as pd
from NN import Simple_Model
from data import combine_data
import gc

def train_test(model, data_collection, setting, fold, PATH = './results/training/'):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    #assert that data_collection is in the correct fold
    assert fold == data_collection.current_test_split, 'fold not right'

    device = tc.device(setting['training_device'])

    #reset model to the same state for each training fold
    model.load_state_dict(tc.load('./results/raw_params.pt'))
    model.train().to(device)

    #get train and test/val sets
    patdata_train, surv_train =  data_collection.get_train_set()
    patdata_test_val, surv_test_val =  data_collection.get_test_set()


    #split testval into test and val... test is the first part!!
    tv_length = patdata_test_val.shape[0]

    patdata_test, surv_test = patdata_test_val[:tv_length//2,:], surv_test_val[:tv_length//2,:]
    patdata_val, surv_val = patdata_test_val[tv_length//2:,:], surv_test_val[tv_length//2:,:]

    ####################################################################################################
    #save test and validation names for later quality control: e.g. each sample is only in one test/val set
    samplenames_test_val = data_collection.get_test_names()
    test_names = samplenames_test_val[:tv_length//2]
    val_names = samplenames_test_val[tv_length//2:]

    test_name_frame= pd.DataFrame({'sample_name': test_names, 'type': 'test', 'fold': fold})
    val_name_frame= pd.DataFrame({'sample_name': val_names, 'type': 'val', 'fold': fold})
    train_name_frame= pd.DataFrame({'sample_name': data_collection.get_train_names(), 'type': 'train', 'fold': fold})
    name_frame = pd.concat([test_name_frame, val_name_frame, train_name_frame],axis=0)
    name_frame.to_csv('./results/training/name_frame.csv',mode='w' if fold==0 else 'a', header=fold==0)
    ######################################################################################################

    print('train size:', patdata_train.shape,'test_size:', patdata_test.shape)

    #prepare test and val data for pycox workflow
    test_data = (patdata_test, (surv_test[:,0], surv_test[:,1]))
    val_data = (patdata_val, (surv_val[:,0], surv_val[:,1]))

    coxph_model = CoxPH(model, tt.optim.Adam(setting['lr'], weight_decay=setting['weight_decay']), device = device)

    effective_epochs = []
    for exp, training_epochs in enumerate(setting['reduce_lr_epochs']):
        callbacks = [tt.callbacks.EarlyStopping(),tt.callbacks.ClipGradNorm(model, max_norm=1.0)]
        print('train for {} epochs with lr {}'.format(training_epochs, setting['lr']*10**(-exp)))
        coxph_model.optimizer.set_lr(setting['lr']*10**(-exp))
        log = coxph_model.fit(patdata_train, (surv_train[:,0], surv_train[:,1]), batch_size=setting['training_batch_size'], epochs=training_epochs, callbacks=callbacks,
                 val_data=val_data, val_batch_size = setting['training_batch_size'], verbose=1)
        effective_epochs.append(log.epoch)

    _ = coxph_model.compute_baseline_hazards()
    surv_pred = coxph_model.predict_surv_df(test_data[0])

    #compute concordance and integrated brier scores
    ev = EvalSurv(surv_pred, np.array(test_data[1][0]).squeeze(), np.array(test_data[1][1]).squeeze(), censor_surv='km')
    concordance = ev.concordance_td()
    time_grid = np.linspace(np.array(test_data[1][0]).squeeze().min(), np.array(test_data[1][0]).squeeze().max(), 100)
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)

    print('concordance:', concordance)
    print('integrated_brier_score:', integrated_brier_score)
    print('integrated_nbll:', integrated_nbll)


    concordance_scores = pd.DataFrame({'fold': [fold], 'conc_score': concordance, 'brier_score': integrated_brier_score, 'ncancers': test_data[0].shape[0]})
    concordance_scores.to_csv('./results/training/conc_scores.csv',mode='w' if fold==0 else 'a', header=fold==0)

    strat_conc_scores = conc_score_per_cancer(coxph_model, data_collection)
    strat_conc_scores['fold'] = fold
    strat_conc_scores.to_csv('./results/training/stratified_conc_scores.csv',mode='w' if fold==0 else 'a', header=fold==0)

    return model


def conc_score_per_cancer(trained_model, data_collection):
    patdata_test_val, surv_test_val =  data_collection.get_test_set()
    tv_length = patdata_test_val.shape[0]

    patdata_test, surv_test = patdata_test_val[:tv_length//2,:], surv_test_val[:tv_length//2,:]
    patdata_val, surv_val = patdata_test_val[tv_length//2:,:], surv_test_val[tv_length//2:,:] #not needed

    cancer_types_test_val = data_collection.get_cancer_types_test()
    cancer_types_test = cancer_types_test_val[:tv_length//2]
    cancer_types_val = cancer_types_test_val[tv_length//2:] #not needed

    unique_cancer_types_test = data_collection.unique_cancer_types
    results = []
    for cancer_type in unique_cancer_types_test:
        current_ids = cancer_types_test==cancer_type
        test_data_stratified = (patdata_test[current_ids,:], (surv_test[current_ids,0], surv_test[current_ids,1]))


        if (test_data_stratified[0].shape[0]<10) | (np.array(test_data_stratified[1][1]).sum() <= 5):
            continue

        surv_pred = trained_model.predict_surv_df(test_data_stratified[0])

        ev_strat = EvalSurv(surv_pred, np.array(test_data_stratified[1][0]).squeeze(), np.array(test_data_stratified[1][1]).squeeze(), censor_surv='km')

        concordance_strat = ev_strat.concordance_td()

        time_grid = np.linspace(np.array(test_data_stratified[1][0]).squeeze().min(), np.array(test_data_stratified[1][0]).squeeze().max(), 100)
        integrated_brier_score_strat = ev_strat.integrated_brier_score(time_grid)

        results.append(pd.DataFrame({'cancer_type': [cancer_type], 'concordance': concordance_strat,"integrated_brier":integrated_brier_score_strat, 
                                     'ncancers_test':test_data_stratified[0].shape[0]}))

        gc.collect()

    stratified_concordance_scores = pd.concat(results, axis=0)

    return stratified_concordance_scores

        
def train_test_individual_cancers(model, data_collection, setting, fold, PATH = './results/training/'):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    device = tc.device(setting['training_device'])
    model.train().to(device)

    patdata_train, surv_train =  data_collection.get_train_set()
    patdata_test_val, surv_test_val =  data_collection.get_test_set()

    tv_length = patdata_test_val.shape[0]

    patdata_test, surv_test = patdata_test_val[:tv_length//2,:], surv_test_val[:tv_length//2,:]
    patdata_val, surv_val = patdata_test_val[tv_length//2:,:], surv_test_val[tv_length//2:,:]


    cancer_types_test_val = data_collection.get_cancer_types_test()
    print(cancer_types_test_val)
    cancer_types_test = cancer_types_test_val[:tv_length//2]
    cancer_types_val = cancer_types_test_val[tv_length//2:]

    cancer_types_train = data_collection.get_cancer_types_train()
    unique_cancer_types_test = data_collection.unique_cancer_types
    results = []
    
    for cancer_type in unique_cancer_types_test:
        print(cancer_type)

        model.load_state_dict(tc.load('./results/raw_params.pt'))

        current_test_ids = cancer_types_test==cancer_type
        current_train_ids = cancer_types_train==cancer_type
        current_val_ids = cancer_types_val==cancer_type

        test_data_now = (patdata_test[current_test_ids,:], (surv_test[current_test_ids,0], surv_test[current_test_ids,1]))
        val_data_now = (patdata_val[current_val_ids,:], (surv_val[current_val_ids,0], surv_val[current_val_ids,1]))

        patdata_train_now = patdata_train[current_train_ids,:]
        surv_train_now = surv_train[current_train_ids,:]

        print('train_samples:', patdata_train_now.shape[0], 'val_samples:', val_data_now[0].shape[0], 'test_samples:', test_data_now[0].shape[0], 
              'train_ratio:', patdata_train_now.shape[0]/(val_data_now[0].shape[0] + test_data_now[0].shape[0] + patdata_train_now.shape[0]))

        if (test_data_now[0].shape[0]<10) | (np.array(test_data_now[1][1]).sum() <= 5) | (val_data_now[0].shape[0]<10) | (np.array(val_data_now[1][1]).sum() <= 5):
            continue
            
        coxph_model = CoxPH(model, tt.optim.Adam(setting['lr'], weight_decay=setting['weight_decay']), device = device)
        #coxph_model = CoxPH(model, tt.optim.SGD(setting['lr']*0.01, momentum=0.9), device = device)

        callbacks = [tt.callbacks.EarlyStopping(), tt.callbacks.ClipGradNorm(model, max_norm=1.0)]

        effective_epochs = []
        for exp, training_epochs in enumerate(setting['reduce_lr_epochs']):
            callbacks = [tt.callbacks.EarlyStopping(),  tt.callbacks.ClipGradNorm(model, max_norm=1.0)]
            print('train for {} epochs with lr {}'.format(training_epochs, setting['lr']*10**(-exp)))
            coxph_model.optimizer.set_lr(setting['lr']*10**(-exp))
            log = coxph_model.fit(patdata_train[current_train_ids,:], (surv_train[current_train_ids,0], surv_train[current_train_ids,1]),
                  batch_size=setting['training_batch_size'], epochs=training_epochs, callbacks=callbacks,
                     val_data=val_data_now, val_batch_size = setting['training_batch_size'], verbose=1)
            effective_epochs.append(log.epoch)

        _ = coxph_model.compute_baseline_hazards()
        surv_pred = coxph_model.predict_surv_df(test_data_now[0])

        ev = EvalSurv(surv_pred, np.array(test_data_now[1][0]).squeeze(), np.array(test_data_now[1][1]).squeeze(), censor_surv='km')
        concordance = ev.concordance_td()
        time_grid = np.linspace(np.array(test_data_now[1][0]).squeeze().min(), np.array(test_data_now[1][0]).squeeze().max(), 100)
        integrated_brier_score = ev.integrated_brier_score(time_grid)
        integrated_nbll = ev.integrated_nbll(time_grid)

        print('concordance:', concordance)
        print('integated_brier_score:', integrated_brier_score)
        print('integrated_nbll:', integrated_nbll)

        results.append(pd.DataFrame({'cancer_type': [cancer_type], 'concordance': concordance, 'integrated_brier':integrated_brier_score, 'ncancers_test':test_data_now[0].shape[0]}))

        del coxph_model
        del log
        del callbacks
        del ev
        tc.cuda.empty_cache()
        
        gc.collect()

    strat_conc_scores = pd.concat(results,axis=0)
    strat_conc_scores['fold'] = fold
    strat_conc_scores.to_csv('./results/training/stratified_conc_scores_individual.csv',mode='w' if fold==0 else 'a', header=fold==0)


def save_predictions(model, data_collection, setting,  fold, PATH = './results/training/'):
    device = tc.device(setting['training_device'])
    model.eval().to(device)
    
    patdata_test_val, surv_test_val =  data_collection.get_test_set()
    patdata_test_val, surv_test_val = patdata_test_val.to(device), surv_test_val.to(device)
    sample_names = data_collection.get_test_names()

    pred = model.forward(patdata_test_val).cpu().detach().numpy().squeeze()

    #maybe implement different risks here


    df = pd.DataFrame({'sample_name': sample_names, 'risk_prediction_all': pred})

    df.to_csv(PATH + 'risk_predictions.csv',mode='w' if fold==0 else 'a', header=fold==0)



