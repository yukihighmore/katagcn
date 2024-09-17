import numpy as np
import pandas as pd



def Scale(x):
    x_ = (x - np.mean(x)) / np.std(x)
    return x_


data = np.load('PEMS08.npz')
data_seq = data['data'][:, :, 0]
data_seq = Scale(data_seq)

# 相邻的节点对
pairs = pd.read_csv('PEMS08.csv')
a_seq = pairs['from']
b_seq = pairs['to']
pairs_num = len(a_seq)
top_K_num = int(pairs_num * 0.25)

# distance = lambda x, y: np.sqrt((x - y) ** 2)
pairs_similar = []
for i in range(pairs_num):
    a = data_seq[:, a_seq[i]]
    b = data_seq[:, b_seq[i]]
    r = np.corrcoef([a, b])[0, 1]
    tmp = [a_seq[i], b_seq[i], r]
    pairs_similar.append(tmp)

pairs_similar = np.array(pairs_similar)
r_list = pairs_similar[:, 2]
idxs = np.argsort(r_list)

pairs_similar_top = []
for i in range(pairs_num):
    if i > pairs_num - top_K_num:
        pairs_similar_top.append(pairs_similar[idxs[i]])

pairs_similar = pd.DataFrame(pairs_similar_top)
pairs_similar.to_csv('top_similarities.csv', index=None, header=None)


