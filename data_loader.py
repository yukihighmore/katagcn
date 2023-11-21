from utils import *


# Dataset class
class Dataset(object):
    def __init__(self, x_data, train_stats):
        self.__data = x_data
        self.train_mean = train_stats['mean']
        self.train_std = train_stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.train_mean, 'std': self.train_std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.train_mean + self.train_std


# Set value in sequence
def seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week):
    # initial (n_frame + 1, n_route + 2)
    tmp_seq = np.zeros((len_seq, n_frame + 1, n_route + 2))
    # 1 static features： route
    features_static = np.arange(1, n_route + 1)
    features_static = features_static.reshape((1, n_route))  # 1, n_route
    # 2 dynamic features: week, day
    for sta in range(offset, offset + len_seq):
        end = sta + n_frame
        # data_tmp -> 24
        data_tmp = data_seq[sta:end, :]
        # tmp -> n_frame, n_route
        features = np.reshape(data_tmp, [n_frame, n_route])
        # feature -> n_frame + 1, n_route
        features_with_route = np.append(features_static, features, axis=0)
        # week -> n_frame + 1, 1
        stamp_in_week = day_in_week[sta]
        feature_week = np.full((n_frame + 1, 1), stamp_in_week)
        # day -> n_frame + 1, 1
        stamp_in_day = time_ind[sta]
        feature_day = np.full((n_frame + 1, 1), stamp_in_day)
        feature_time = np.append(feature_week, feature_day, axis=1)
        feature_all = np.append(feature_time, features_with_route, axis=1)
        tmp_seq[sta - offset, :, :] = feature_all
    return tmp_seq


# Return the sequence(train/test/val) with value
def seq_gen(seq_ppt, data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num):
    """
    :param seq_ppt: str, the type of target date sequence.
    :param data_config: units, the ratio of train, test, val.
    :param data_seq: np.ndarray, source data / time-series.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 12 (3 /15 min, 6 /30 min, 9 /45 min, 12/60 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    """
    n_train, n_val, n_test = data_config
    slot_nums = day_slot * day_num

    len_train = int(n_train * slot_nums)
    len_val = int(n_val * slot_nums)
    if seq_ppt == 'train':
        len_seq = len_train
        offset = 0
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week)
    if seq_ppt == 'val':
        len_seq = len_val
        offset = len_train
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week)
    if seq_ppt == 'test':
        org = n_val * day_slot * day_num
        res_train = n_train * slot_nums - float(len_train)
        res_val = n_test * slot_nums - float(len_val)
        len_seq = int(org + res_train + res_val) - n_frame + 1
        offset = len_train + len_val
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week)

    return assignment_seq


# Divide the data into train, val and test sequences
def data_gen(file_path, data_config, n_route, n_frame=24, day_slot=288, day_num=62):
    """
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param day_num: int, the number of days.
    :param C_0: int, the size of input channel.
    :return: dict, dataset that contains training, validation and test with stats.
    """

    # generate training, validation and test data
    try:
        # open_file
        data = np.load(file_path)
        data_seq = data['data'][:, :, 0]
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    L, N = data_seq.shape


    # numerical time_in_day
    time_ind = [i % 288 for i in range(L)]
    time_ind = np.array(time_ind)

    # numerical day_in_week
    day_in_week = [(i // 288) % 7 for i in range(L)]
    day_in_week = np.array(day_in_week)

    seq_train = seq_gen('train', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num)
    train = seq_train[:, 1:, 2:]

    seq_test = seq_gen('test', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num)
    test = seq_test[:, 1:, 2:]

    seq_val = seq_gen('val', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num)
    val = seq_val[:, 1:, 2:]

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(train), 'std': np.std(train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route].
    x_train = z_score(train, x_stats['mean'], x_stats['std'])
    x_val = z_score(val, x_stats['mean'], x_stats['std'])
    x_test = z_score(test, x_stats['mean'], x_stats['std'])

    # attach time
    x_y_train = np.concatenate((x_train[:, :12, :], train[:, 12:, :]), axis=1)
    data_train = np.concatenate((seq_train[:, 1:, :2], x_y_train), axis=2)
    data_train = np.concatenate((seq_train[:, :1, :], data_train), axis=1)
    # 测试集
    x_y_test = np.concatenate((x_test[:, :12, :], test[:, 12:, :]), axis=1)
    data_test = np.concatenate((seq_test[:, 1:, :2], x_y_test), axis=2)
    data_test = np.concatenate((seq_test[:, :1, :], data_test), axis=1)
    # 验证集
    x_y_val = np.concatenate((x_val[:, :12, :], val[:, 12:, :]), axis=1)
    data_val = np.concatenate((seq_val[:, 1:, :2], x_y_val), axis=2)
    data_val = np.concatenate((seq_val[:, :1, :], data_val), axis=1)

    datas = {'train': data_train, 'val': data_val, 'test': data_test}
    dataset = Dataset(datas, x_stats)
    return dataset
