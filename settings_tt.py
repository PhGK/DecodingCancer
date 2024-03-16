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

            'hidden_depth_simple': 0,   

            'factor_hidden_nodes': 10, # determines the width ofhidden layers -> width = factor_hidden_nodes * input_width
  
            'training_device': 'cuda:0',

            'training_batch_size': 1024,
            'reduce_lr_epochs': [50,50,50],
            'lr': 1e-4,
            'dropout': 0.5,
            'input_dropout': 0.5,
            'weight_decay': 0,

            #LRP params
            'nepochs': 16,
            'batch_size': 8,

            'LRP_device': 'cuda:0',
            'LRP_batch_size': 1000}
