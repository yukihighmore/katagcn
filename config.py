from data_loader import *

n = 883
n_stamp = 288
n_day = 98
n_his = 12
n_pred = 12
n_frame = n_his + n_pred
n_c = 1
batch_size = 32
epoch = 200
emb_dim = 4
n_h = 2
graph_weight = f'./dataset/07_weight.csv'
data_file = f'./dataset/PEMS07.npz'
n_train, n_test, n_val = 0.7, 0.2, 0.1
print(f'Training configs: epoch:{epoch}, batch_size:{batch_size}, data_file:{data_file}, '
      f'graph_weight:{graph_weight}, emb_dim:{emb_dim}.')
# data_gen(file_path, data_config, n_route, n_frame=24, day_slot=288, day_num=91):
PeMS = data_gen(data_file, (n_train, n_val, n_test), n, n_his + n_pred, n_stamp, n_day)
print('>> Loading dataset successfully!<<')
