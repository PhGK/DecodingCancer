place = 'M'



if place == 'M':
    setting = {#data params
           'DATAPATH': '../some_data/',
           'therapy_file': 'therapy.csv',
           'diagnostics_file': 'diagnostics.csv',
           'survival_file': 'survival.csv',
           'duration_name': '0',
           'event_name': 'event',
 
            #training params

            'hidden_depth_simple': 0,   # this is the depth for the first two models (before product) as well as the model after product

            'factor_hidden_nodes': 100, # determines the width of first hidden layers -> width = factor_hidden_nodes * input_width
  
            'training_device': 'cuda:0',

            'training_batch_size': 64,
            'reduce_lr_epochs': [2],
            'lr': 1e-3,
            'dropout': 0.05,
            'input_dropout': 0.0,
            'weight_decay': 0,

            #LRP params
            'nepochs': 16,
            'batch_size': 8,

            'LRP_device': 'cuda:0',
            'LRP_batch_size': 1000}
