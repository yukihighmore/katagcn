import tensorflow as tf

tf.config.run_functions_eagerly(True)
from config import *
from data_loader import *
from layers import *
from model import *
from tensorflow import keras

if __name__ == "__main__":
    # change the datssets here as 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMSD7(M)', 'PEMSD7(L)'
    dateset_name = 'PEMS08'
    args = config(dateset_name)

    # data transfer
    N, L, P, M = args['N'], args['day_num'], args['n_past'], args['n_pred']
    sod, sow, w_o, u_w = args['day_stamps'], args['week_states'], args['week_offset'], args['use_weekends']
    Data = args['data_file']
    # ratio
    n_train, n_test, n_val = 0.7, 0.2, 0.1

    PeMS = data_gen(Data, (n_train, n_val, n_test), N, L, P, M, sod, sow, w_o, u_w)
    print('>> Loading dataset successfully!<<')

    # dataloader
    batch_size = args['batch_size']
    train_data = PeMS.get_data('train')
    his_data = tf.constant(train_data[:, 0:P + 1, :], dtype=tf.float32)
    pre_data = tf.constant(train_data[:, P + 1:, 2:], dtype=tf.float32)

    his_train_data = tf.data.Dataset.from_tensor_slices(his_data)
    pre_train_data = tf.data.Dataset.from_tensor_slices(pre_data)
    train_dataset = tf.data.Dataset.zip((his_train_data, pre_train_data)).shuffle(buffer_size=3).batch(
        batch_size).cache()
    val_data = PeMS.get_data('val')
    his_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, 0:P + 1, :], dtype=tf.float32))
    pre_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, P + 1:, 2:], dtype=tf.float32))
    val_dataset = tf.data.Dataset.zip((his_val_data, pre_val_data)).shuffle(buffer_size=3).batch(batch_size).cache()
    test_data = PeMS.get_data('test')
    his_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, 0:P + 1, :], dtype=tf.float32))
    pre_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, P + 1:, 2:], dtype=tf.float32))
    test_dataset = tf.data.Dataset.zip((his_test_data, pre_test_data)).shuffle(buffer_size=3).batch(batch_size).cache()

    emb_dim, mlp, dilation, Graph = args['emb_dim'], args['layers_num'], args['dilation'], args['graph_file']

    mm = TaKaGCN(graph_dim=N, n_his=P, n_pre=M, emb_dim=emb_dim, mlp=mlp, day_stamps=sod, week_states=sow,
                 week_flag=u_w, dilation=dilation, Graph=Graph, std=PeMS.get_stats()['std'], mean=PeMS.get_stats()['mean'])

    epoch = args['epoch']

    inputs = keras.Input(shape=(P + 1, N + 2))
    outputs = mm(inputs)
    model = CustomModel(inputs=inputs, outputs=outputs)

    model_name = dateset_name + '_' + str(emb_dim) + '_' + str(mlp) + '.keras'
    # period=1 verbose=1
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_name, monitor='val_mean_absolute_error', mode='min',
                                        save_best_only=True,
                                        save_weights_only=False),
        keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5,
                                          patience=args['reduce_patience'],
                                          mode='min', cooldown=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.0001,
                                      patience=args['early_stop_patience'])
    ]
    # tf.keras.losses.mean_absolute_error
    loss = tf.keras.losses.MeanAbsoluteError(
        reduction="sum_over_batch_size", name="mean_absolute_error"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args['ini_lr'], beta_1=0.9, beta_2=0.999, epsilon=None,
                                           decay=0.0,
                                           amsgrad=False),
        loss=loss,
        metrics=tf.keras.metrics.MeanAbsoluteError(), run_eagerly=True)
    model.fit(
        train_dataset,
        epochs=epoch,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    # test
    print('load best model!')
    _custom_objects = {
        "MLP": MLP,
        "iconv": kagconv,
        "talinear": talinear,
        "TaKaGCN": TaKaGCN,
        "CustomModel": CustomModel
    }
    test_model = keras.models.load_model(model_name, custom_objects=_custom_objects)

    y_pred = []
    y_true = []
    for element in test_dataset.as_numpy_iterator():
        x, y = element
        y_ = test_model(x, training=False)
        y_true.append(y)
        y_pred.append(y_)
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)
    evl = evaluation(y_true, y_pred, PeMS.get_stats())

    mape = 0
    mae = 0
    rmse = 0
    for step in range(12):
        #print(f'{evl[step][0]:7.3%},{evl[step][1]:4.3f},{evl[step][2]:6.3f}')
        print(f'The steps {step} MAPE is: {evl[step][0]:7.3%}, MAE is:{evl[step][1]:4.3f}, RMSE is: {evl[step][2]:6.3f}')
        mape = mape + evl[step][0]
        mae = mae + evl[step][1]
        rmse = rmse + evl[step][2]
    mape = mape / 12
    mae = mae / 12
    rmse = rmse / 12
    #print(f'{mape:7.3%},{mae:4.3f},{rmse:6.3f}')
    print(f'The average MAPE is: {mape:7.3%}, MAE is: {mae:4.3f}, RMSE is: {rmse:6.3f}')

    #print("Mission accomplished, dad")
