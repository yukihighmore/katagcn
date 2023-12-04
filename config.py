
def config(dataset_name):
    args = {}
    if dataset_name == 'PEMS03':
        # problem definition
        args['N'] = 358
        args['day_num'] = 30 + 31 + 30
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 0
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 8
        args['layers_num'] = 0
        # data loader
        args['graph_file'] = f'./dataset/03_weight.csv'
        args['data_file'] = f'./dataset/PEMS03.npz'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002

    if dataset_name == 'PEMS04':
        # problem definition
        args['N'] = 307
        args['day_num'] = 31 + 28
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 0
        args['use_weekends'] = False
        args['dilation'] = True
        # hyper-parameters
        args['emb_dim'] = 5
        args['layers_num'] = 1
        # data loader
        args['graph_file'] = f'./dataset/04_weight.csv'
        args['data_file'] = f'./dataset/PEMS04.npz'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.003

    if dataset_name == 'PEMS07':
        # problem definition
        args['N'] = 883
        args['day_num'] = 98
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 0
        args['use_weekends'] = False
        args['dilation'] = True
        # hyper-parameters
        args['emb_dim'] = 5
        args['layers_num'] = 2
        # data loader
        args['graph_file'] = f'./dataset/07_weight.csv'
        args['data_file'] = f'./dataset/PEMS07.npz'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002

    if dataset_name == 'PEMS08':
        # problem definition
        args['N'] = 170
        args['day_num'] = 62
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 6
        args['use_weekends'] = True
        args['dilation'] = True
        # hyper-parameters
        args['emb_dim'] = 5
        args['layers_num'] = 1
        # data loader
        args['graph_file'] = f'./dataset/08_weight.csv'
        args['data_file'] = f'./dataset/PEMS08.npz'
        # train
        args['reduce_patience'] = 10
        args['early_stop_patience'] = 20
        args['ini_lr'] = 0.003

    if dataset_name == 'PEMSD7(M)':
        # problem definition
        args['N'] = 228
        args['day_num'] = 31 + 30 - 17
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 5
        args['week_offset'] = 1
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 5
        args['layers_num'] = 3
        # data loader
        args['graph_file'] = f'./dataset/PeMSD7_W_228.csv'
        args['data_file'] = f'./dataset/PeMSD7_V_228.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002

    if dataset_name == 'PEMSD7(L)':
        # problem definition
        args['N'] = 1026
        args['day_num'] = 31 + 30 - 17
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 200
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 5
        args['week_offset'] = 1
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 3
        args['layers_num'] = 3
        # data loader
        args['graph_file'] = f'./dataset/PeMSD7_W_1026.csv'
        args['data_file'] = f'./dataset/PeMSD7_V_1026.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002



    return args