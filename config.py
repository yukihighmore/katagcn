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
        args['epoch'] = 300
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 0
        args['use_weekends'] = True
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 6
        args['layers_num'] = 1
        args['graph_emb'] = 358
        # data loader
        args['data_file'] = f'./dataset/PEMS03/PEMS03.npz'
        args['top_corr'] = f'./dataset/PEMS03/top_similarities.csv'
        # train
        args['reduce_patience'] = 15
        args['early_stop_patience'] = 25
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    if dataset_name == 'PEMS04':
        # problem definition
        args['N'] = 307
        args['day_num'] = 31 + 28
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 400
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 0
        args['use_weekends'] = False
        args['dilation'] = True
        # hyper-parameters
        args['emb_dim'] = 6
        args['layers_num'] = 2
        args['graph_emb'] = 307
        # data loader
        args['data_file'] = f'./dataset/PEMS04/PEMS04.npz'
        args['top_corr'] = f'./dataset/PEMS04/top_similarities.csv'
        # train
        args['reduce_patience'] = 10
        args['early_stop_patience'] = 25
        args['ini_lr'] = 0.003
        args['alpha'] = 1

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
        args['emb_dim'] = 6
        args['layers_num'] = 2
        args['graph_emb'] = 883
        # data loader
        args['data_file'] = f'./dataset/PEMS07/PEMS07.npz'
        args['top_corr'] = f'./dataset/PEMS07/top_similarities.csv'
        # train
        args['reduce_patience'] = 10
        args['early_stop_patience'] = 25
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    if dataset_name == 'PEMS08':
        # problem definition
        args['N'] = 170
        args['day_num'] = 62
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 400
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 6
        args['use_weekends'] = True
        args['dilation'] = True
        # hyper-parameters
        args['emb_dim'] = 6
        args['layers_num'] = 2
        args['graph_emb'] = 170
        # data loader
        args['data_file'] = f'./dataset/PEMS08/PEMS08.npz'
        args['top_corr'] = f'./dataset/PEMS08/top_similarities.csv'
        # train
        args['reduce_patience'] = 10
        args['early_stop_patience'] = 25
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    if dataset_name == 'PEMSD7(M)':
        # problem definition
        args['N'] = 228
        args['day_num'] = 31 + 30 - 17
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 300
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 5
        args['week_offset'] = 1
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 7
        args['layers_num'] = 3
        args['graph_emb'] = 228
        # data loader
        args['data_file'] = f'./dataset/PEMSD7(M)/PeMSD7_V_228.csv'
        args['top_corr'] = f'./dataset/PEMSD7(M)/top_similarities.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    if dataset_name == 'PEMSD7(L)':
        # problem definition
        args['N'] = 1026
        args['day_num'] = 31 + 30 - 17
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 300
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 5
        args['week_offset'] = 1
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 4
        args['layers_num'] = 3
        args['graph_emb'] = 1026
        # data loader
        args['data_file'] = f'./dataset/PEMSD7(L)/PeMSD7_V_1026.csv'
        args['top_corr'] = f'./dataset/PEMSD7(L)/top_similarities.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    if dataset_name == 'PEMS-BAY':
        # problem definition
        args['N'] = 325
        args['day_num'] = 31 + 28 + 31 + 30 + 31 + 30 - 1  # 去掉了 2017-03-12
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 300
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 1
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 3
        args['layers_num'] = 4
        args['graph_emb'] = 325
        # data loader
        args['data_file'] = f'./dataset/PEMS-BAY/PEMS-BAY.h5'
        args['top_corr'] = f'./dataset/PEMS-BAY/top_similarities.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 10
        args['ini_lr'] = 0.003
        args['alpha'] = 1

    if dataset_name == 'METR-LA':
        # problem definition
        args['N'] = 207
        args['day_num'] = 119
        args['n_past'] = 12
        args['n_pred'] = 12
        # fundamental
        args['batch_size'] = 64
        args['epoch'] = 300
        # construct model
        args['day_stamps'] = 288
        args['week_states'] = 7
        args['week_offset'] = 5
        args['use_weekends'] = False
        args['dilation'] = False
        # hyper-parameters
        args['emb_dim'] = 3
        args['layers_num'] = 1
        args['graph_emb'] = 207
        # data loader
        args['data_file'] = f'./dataset/METR-LA/METR-LA.h5'
        args['top_corr'] = f'./dataset/METR-LA/top_similarities.csv'
        # train
        args['reduce_patience'] = 5
        args['early_stop_patience'] = 15
        args['ini_lr'] = 0.002
        args['alpha'] = 1

    return args
