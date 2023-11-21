import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)
from layers import *
from model import *
from tensorflow import keras

# Set all GPUs to apply for graphics storage only when needed
# 将所有 GPU 设置为仅在需要时申请显存
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

if __name__ == '__main__':
    # dataloader
    train_data = PeMS.get_data('train')
    his_data = tf.constant(train_data[:, 0:13, :], dtype=tf.float32)
    pre_data = tf.constant(train_data[:, 13:, 2:], dtype=tf.float32)

    his_train_data = tf.data.Dataset.from_tensor_slices(his_data)
    pre_train_data = tf.data.Dataset.from_tensor_slices(pre_data)
    train_dataset = tf.data.Dataset.zip((his_train_data, pre_train_data)).shuffle(buffer_size=3).batch(
        batch_size).cache()
    val_data = PeMS.get_data('val')
    his_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, 0:13, :], dtype=tf.float32))
    pre_val_data = tf.data.Dataset.from_tensor_slices(tf.constant(val_data[:, 13:, 2:], dtype=tf.float32))
    val_dataset = tf.data.Dataset.zip((his_val_data, pre_val_data)).shuffle(buffer_size=3).batch(batch_size).cache()
    test_data = PeMS.get_data('test')
    his_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, 0:13, :], dtype=tf.float32))
    pre_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(test_data[:, 13:, 2:], dtype=tf.float32))
    test_dataset = tf.data.Dataset.zip((his_test_data, pre_test_data)).shuffle(buffer_size=3).batch(batch_size).cache()

    mm = TDGNN(graph_dim=n, n_his=n_his, n_pre=n_pred,
               emb_dim=emb_dim, n_stamp=n_stamp, graph_weight=graph_weight)

    inputs = keras.Input(shape=(n_his + 1, n + 2))
    outputs = mm(inputs)
    model = CustomModel(inputs=inputs, outputs=outputs)
    # period=1 verbose=1
    callbacks = [
        keras.callbacks.ModelCheckpoint('pems07.keras', monitor='val_mean_absolute_error', mode='min',
                                        save_best_only=True,
                                        save_weights_only=False),
        keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=5,
                                          mode='min', cooldown=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.0001, patience=10)
    ]
    # tf.keras.losses.mean_absolute_error
    loss = tf.keras.losses.MeanAbsoluteError(
        reduction="sum_over_batch_size", name="mean_absolute_error"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
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
        "TDGNN": TDGNN,
        "CustomModel": CustomModel
    }
    test_model = keras.models.load_model('pems07.keras', custom_objects=_custom_objects)
    batch_nums = len(test_dataset)
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
    print(evl)
    # evl = result / batch_nums
    mape = 0
    mae = 0
    rmse = 0
    for step in range(12):
        print(f'{evl[step][0]:7.3%},{evl[step][1]:4.3f},{evl[step][2]:6.3f}')
        mape = mape + evl[step][0]
        mae = mae + evl[step][1]
        rmse = rmse + evl[step][2]
    mape = mape / 12
    mae = mae / 12
    rmse = rmse / 12
    print(f'{mape:7.3%},{mae:4.3f},{rmse:6.3f}')
    
    print("Mission accomplished, dad")
