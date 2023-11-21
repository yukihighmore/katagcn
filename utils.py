import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

def scaled_laplacian(file_path):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    W = pd.read_csv(file_path, header=None).values
    # n, d = np.shape(W)[0], np.sum(W, axis=1)
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    # L0 和 L1，其中 L0 是一个大小为 n x n 的单位矩阵，L1 是提供的拉普拉斯矩阵 L 的副本。
    # 这些矩阵用于在 cheb_poly_approx 函数中计算切比雪夫多项式。
    # 具体来说，L0 和 L1 是拉普拉斯矩阵的前两个切比雪夫多项式。
    # 其余的切比雪夫多项式是使用这两个矩阵作为递推关系来计算的，

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        # 最后一维拼接：其他维度的需要是相同的维度
        # (n, n, KS)
        return np.stack(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def GCN_approx(file_path, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    W = pd.read_csv(file_path, header=None).values
    A = W + np.identity(n)
    d = np.array(A.sum(1))
    d_hat = diags(np.power(d, -0.5).flatten()).toarray()
    L = d_hat.dot(A).dot(d_hat)

    return L


# Return the relative position relationship matrix
def get_nearest_vertex(file_path):
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    sort_index = np.argsort(W, axis=-1)
    n_vertex = len(W)
    srpe = np.zeros(shape=[n_vertex, n_vertex])
    for i in range(n_vertex):
        for j in range(n_vertex):
            srpe[i, sort_index[i, j]] = j
    return srpe


# Z-score normalization
def z_score(x, mean, std):
    return (x - mean) / std


# The inverse of function z_score()
def z_inverse(x, mean, std):
    return x * std + mean


# Mean absolute percentage error
def MAPE(y_true, y_pred, null_val=0):
    mask = np.not_equal(y_true, null_val).astype('float32')
    mape = np.abs((y_pred - y_true)) / (np.abs(y_true) + 1)
    mape = mask * mape
    non_zero_len = mask.sum()
    return np.sum(mape) / non_zero_len


# Mean squared error
def RMSE(v, v_):
    mse = tf.keras.metrics.MeanSquaredError()
    mse.update_state(v, v_)
    return np.sqrt(mse.result().numpy())


# Mean absolute error
def MAE(v, v_):
    mae = tf.keras.metrics.MeanAbsoluteError()
    mae.update_state(v, v_)
    return mae.result().numpy()


# Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
def evaluation(y, y_, y_stats):
    """
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param y_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    """
    dim = len(y_.shape)

    if dim == 2:
        # single_step case
        #v = z_inverse(y, y_stats['mean'], y_stats['std'])
        v_ = z_inverse(y_, y_stats['mean'], y_stats['std'])
        return np.array([MAPE(y, v_), MAE(y, v_), RMSE(y, v_)])
        # return np.array([MAPE(y, y_), MAE(y, y_), RMSE(y, y_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [ batch_size, time_step, n_route]
        # recursively call
        for i in range(y_.shape[1]):
            real = y[:, i, :]
            pre = y_[:, i, :]
            tmp_res = evaluation(real, pre, y_stats)
            tmp_list.append(tmp_res)
        return tmp_list
