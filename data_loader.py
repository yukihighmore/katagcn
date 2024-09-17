from utils import *
import h5py

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
def seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week, use_weekend):
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
        if use_weekend:
            if stamp_in_week % 7 == 0 or stamp_in_week % 7 == 1:
                feature_week[0, 0] = 0
            else:
                feature_week[0, 0] = 1
        # day -> n_frame + 1, 1
        stamp_in_day = time_ind[sta]
        feature_day = np.full((n_frame + 1, 1), stamp_in_day)
        feature_time = np.append(feature_week, feature_day, axis=1)
        feature_all = np.append(feature_time, features_with_route, axis=1)
        tmp_seq[sta - offset, :, :] = feature_all
    return tmp_seq


# Return the sequence(train/test/val) with value
def seq_gen(seq_ppt, data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num, use_weekend):

    n_train, n_val, n_test = data_config
    slot_nums = day_slot * day_num

    len_train = int(n_train * slot_nums)
    len_val = int(n_val * slot_nums)
    if seq_ppt == 'train':
        len_seq = len_train
        offset = 0
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week, use_weekend)
    if seq_ppt == 'val':
        len_seq = len_val
        offset = len_train
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week, use_weekend)
    if seq_ppt == 'test':
        org = n_val * day_slot * day_num
        res_train = n_train * slot_nums - float(len_train)
        res_val = n_test * slot_nums - float(len_val)
        len_seq = int(org + res_train + res_val) - n_frame + 1
        offset = len_train + len_val
        assignment_seq = seq_assignment(len_seq, n_frame, n_route, offset, data_seq, time_ind, day_in_week, use_weekend)

    return assignment_seq


# Divide the data into train, val and test sequences
def data_gen(file_path, data_config, n_route, day_num, P, M, day_slot=288, week_slot=7, week_offset=0, use_weekend=False, pems_bay=False):
    # generate training, validation and test data
    try:
        # open_file
        if file_path[-1] == 'v':
            data_seq = pd.read_csv(file_path, header=None).values
        else:
            if file_path[-4:] == 'Y.h5':
                data = h5py.File(file_path, 'r')
                data_seq = data['speed']['block0_values'][:]
            else:
                if file_path[-4:] == 'A.h5':
                    data = h5py.File(file_path, 'r')
                    data_seq = data['df']['block0_values'][:]
                else:
                    data = np.load(file_path)
                    data_seq = data['data'][:, :, 0]
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    L, N = data_seq.shape
    n_frame = P + M
    if pems_bay == True:
        jieduan = 288 * (31 + 28 + 11)
        time_ind_1 = [i % day_slot for i in range(jieduan)]
        time_ind_2 = [i % day_slot for i in range(jieduan + 288 -12, L)]
        time_ind = time_ind_1 + time_ind_2
        time_ind = np.array(time_ind)

        day_in_week_1 = [((i // 288) + week_offset) % week_slot for i in range(jieduan)]
        day_in_week_2 = [((i // 288) + week_offset) % week_slot for i in range(jieduan + 288 -12, L)]
        day_in_week = day_in_week_1 + day_in_week_2
        day_in_week = np.array(day_in_week)
        data_seq_1 = data_seq[:jieduan, :]
        data_seq_2 = data_seq[jieduan + 288 - 12:, :]
        data_seq = np.concatenate((data_seq_1, data_seq_2), axis=0)

    else:
        # numerical time_in_day
        time_ind = [i % day_slot for i in range(L)]
        time_ind = np.array(time_ind)

        # numerical day_in_week
        day_in_week = [((i // 288) + week_offset) % week_slot for i in range(L)]
        day_in_week = np.array(day_in_week)



    seq_train = seq_gen('train', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num, use_weekend)
    train = seq_train[:, 1:, 2:]

    seq_test = seq_gen('test', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num, use_weekend)
    test = seq_test[:, 1:, 2:]

    seq_val = seq_gen('val', data_config, data_seq, time_ind, day_in_week, n_frame, n_route, day_slot, day_num,use_weekend)
    val = seq_val[:, 1:, 2:]

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(train), 'std': np.std(train)}
    print('train mean:')
    print(np.mean(train))
    print('train std:')
    print(np.std(train))
    print('val mean:')
    print(np.mean(val))
    print('val std:')
    print(np.std(val))
    print('test mean:')
    print(np.mean(test))
    print('test std:')
    print(np.std(test))

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route].
    x_train = z_score(train, x_stats['mean'], x_stats['std'])
    x_val = z_score(val, x_stats['mean'], x_stats['std'])
    x_test = z_score(test, x_stats['mean'], x_stats['std'])

    # attach time
    x_y_train = np.concatenate((x_train[:, :P, :], train[:, P:, :]), axis=1)
    data_train = np.concatenate((seq_train[:, 1:, :2], x_y_train), axis=2)
    data_train = np.concatenate((seq_train[:, :1, :], data_train), axis=1)
    # 测试集
    x_y_test = np.concatenate((x_test[:, :P, :], test[:, P:, :]), axis=1)
    data_test = np.concatenate((seq_test[:, 1:, :2], x_y_test), axis=2)
    data_test = np.concatenate((seq_test[:, :1, :], data_test), axis=1)
    # 验证集
    x_y_val = np.concatenate((x_val[:, :P, :], val[:, P:, :]), axis=1)
    data_val = np.concatenate((seq_val[:, 1:, :2], x_y_val), axis=2)
    data_val = np.concatenate((seq_val[:, :1, :], data_val), axis=1)

    datas = {'train': data_train, 'val': data_val, 'test': data_test}
    dataset = Dataset(datas, x_stats)
    return dataset
