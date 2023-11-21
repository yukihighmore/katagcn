import tensorflow as tf

tf.config.run_functions_eagerly(True)
from utils import *
from config import *

seed = 2023

# TDGNN Module
class TDGNN(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, n_stamp, graph_weight, name="TDGNN",
                 **kwargs):
        super(TDGNN, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.n_stamp = n_stamp
        self.graph_weight = graph_weight
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # get relative position matrix
        hop_arr = get_nearest_vertex(graph_weight)
        seq_hop_arr = np.zeros(graph_dim * graph_dim)
        for i in range(graph_dim * graph_dim):
            k = int(i / graph_dim)
            j = i % graph_dim
            seq_hop_arr[i] = k * graph_dim + hop_arr[k, j]
        self.SDist = tf.constant(seq_hop_arr, dtype=tf.int32)
        # rd
        self.srpe = self.add_weight(name=self.name + '_SRPE', shape=(graph_dim * graph_dim, 1),
                                    initializer=tf.keras.initializers.glorot_normal(seed))
        self.relu_g = tf.keras.layers.ReLU()
        # ad
        self.sape = self.add_weight(name=self.name + '_SAPE', shape=(graph_dim, n_his, emb_dim),
                                    initializer=tf.keras.initializers.glorot_normal(seed))
        # at
        self.dow_emb = self.add_weight(name=self.name + '_day_of_week', shape=(7, graph_dim, emb_dim),
                                       initializer=tf.keras.initializers.glorot_normal(seed))
        self.tod_emb = self.add_weight(name=self.name + '_time_of_day', shape=(n_stamp, graph_dim, emb_dim),
                                       initializer=tf.keras.initializers.glorot_normal(seed))
        # connect
        self.connect_weight = self.add_weight(name=self.name + '_connect_weight_',
                                              shape=[n_his, graph_dim, 3 * emb_dim, emb_dim],
                                              initializer=tf.keras.initializers.glorot_normal(seed))
        self.connect_bias = self.add_weight(name=self.name + '_connect_bias_' + self.name,
                                            shape=[n_his, graph_dim, emb_dim],
                                            initializer=tf.keras.initializers.Zeros(),
                                            regularizer=tf.keras.regularizers.L2(1e-3))
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.3, name=self.name + '_relu')

        # 动态
        # 时间依赖
        self.dy_weight_1 = self.add_weight(name=self.name + '_dw1',
                                           shape=[7, n_his, emb_dim, 2 * emb_dim, n_his],
                                           initializer=tf.keras.initializers.glorot_normal(seed),
                                           trainable=True)
        self.dy_offset_1 = self.add_weight(name=self.name + '_do1', shape=[7, graph_dim, 2 * emb_dim, n_his],
                                           initializer=tf.keras.initializers.Zeros(),
                                           trainable=True)
        self.dy_weight_2 = self.add_weight(name=self.name + '_dw2',
                                           shape=[n_stamp, n_his, emb_dim, 2 * emb_dim, n_his],
                                           initializer=tf.keras.initializers.glorot_normal(seed),
                                           trainable=True)
        self.dy_offset_2 = self.add_weight(name=self.name + '_do2', shape=[n_stamp, graph_dim, 2 * emb_dim, n_his],
                                           initializer=tf.keras.initializers.Zeros(),
                                           trainable=True)
        self.relulast = tf.keras.layers.LeakyReLU(alpha=0.3, name=self.name + '_relulast')

        self.drop = tf.keras.layers.Dropout(name=self.name + '_drop', rate=0.5)

        # GLU
        self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, 2 * emb_dim, graph_dim))
        self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, 2 * emb_dim, n_his))

        self.weight1 = self.add_weight(name=self.name + '_w1', shape=[graph_dim, n_his, n_pre],
                                       initializer=tf.keras.initializers.glorot_normal(seed),
                                       trainable=True)
        self.weight2 = self.add_weight(name=self.name + '_w2', shape=[graph_dim, n_his, n_pre],
                                       initializer=tf.keras.initializers.glorot_normal(seed),
                                       trainable=True)
        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None)

    def call(self, inputs, training=True):
        batch_size = tf.shape(inputs)[0]
        n_route = tf.cast(inputs[:, 0, 2:], dtype='int32')
        s_week = tf.cast(inputs[:, 1:, 0], dtype='int32')
        s_day = tf.cast(inputs[:, 1:, 1], dtype='int32')
        week = tf.cast(inputs[:, 0, 0], dtype='int32')
        date = tf.cast(inputs[:, 0, 1], dtype='int32')
        # inputs -> batch_size, n_his + 1, graph_dim + 2, channels
        x_3 = inputs[:, 1:, 2:]
        x_4 = tf.expand_dims(x_3, axis=-1)
        # 嵌入向量
        x_embdings = self.x2embding(x_4)
        # x_embdings -> b, t, g, e
        relation_pos_emb = tf.gather(self.srpe, self.SDist)
        relation_pos_emb = tf.reshape(relation_pos_emb, shape=[self.graph_dim, self.graph_dim])
        relation_pos_emb = self.relu_g(tf.matmul(relation_pos_emb, relation_pos_emb, transpose_b=True))
        x_rel_emb = tf.einsum('vg, btge->btve', relation_pos_emb, x_embdings)
        pos_emb = tf.gather(self.sape, n_route, axis=0)
        # b, g, t, e -> b, t, g, e
        pos_emb = tf.transpose(pos_emb, perm=[0, 2, 1, 3])
        # b, t, g, e
        dow = tf.gather(self.dow_emb, s_week, axis=0)
        tod = tf.gather(self.tod_emb, s_day, axis=0)
        time_emb = dow + tod

        # x_list -> b, t, g, 3e
        x_list = tf.concat([x_rel_emb, pos_emb, time_emb], axis=-1)
        x_embdings = tf.einsum('btgf,tgfe->btge', x_list, self.connect_weight)
        x_embdings = x_embdings + self.connect_bias
        x_embdings = self.relu(x_embdings)

        # x_embdings -> b, t, g, e -> b, g, e, t
        x_embdings = tf.transpose(x_embdings, perm=[0, 2, 3, 1])
        # 时刻线性变换
        weight_1 = tf.gather(self.dy_weight_1, week)
        bais_1 = tf.gather(self.dy_offset_1, week)
        weight_2 = tf.gather(self.dy_weight_2, date)
        bais_2 = tf.gather(self.dy_offset_2, date)
        # i -> e, j -> t
        x_ = tf.einsum('bget,bteij->bgij', x_embdings, weight_1)
        x_ = tf.add(x_, bais_1)
        x__ = tf.einsum('bget,bteij->bgij', x_embdings, weight_2)
        x__ = tf.add(x__, bais_2)
        x_embdings = (x_ + x__) * 0.5
        x_embdings = self.relulast(x_embdings)

        # x_embdings -> b, g, 2e, t -> b, t, g, 2e
        x_embdings = tf.transpose(x_embdings, perm=[0, 3, 1, 2])

        x_embdings = self.drop(x_embdings, training=training)
        x_3 = tf.transpose(x_3, perm=[0, 2, 1])
        # GLU
        # tge -> b, t, g, 2e
        tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
        tg = tf.einsum('gev,btge->btv', self.tge2tg, tge)
        tg2gt = tf.transpose(tg, perm=[0, 2, 1])
        gt1 = (tg2gt + x_3) * tf.sigmoid(tg2gt)
        # gte -> b, g, t, 2e
        gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        gt = tf.einsum('teo,bgte->bgo', self.gte2gt, gte)
        gt2 = (gt + x_3) * tf.sigmoid(gt)

        # x -> b, t, g -> b, g, t
        x1 = tf.einsum('gio,bgi->bgo', self.weight1, gt1)
        x2 = tf.einsum('gio,bgi->bgo', self.weight2, gt2)

        x = tf.concat([x1, x2], axis=-1)
        x = self.fully(x)
        output = tf.transpose(x, perm=[0, 2, 1])

        return output

    def get_config(self):
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "n_pre": self.n_pre,
                  "emb_dim": self.emb_dim, "n_stamp": self.n_stamp,
                  "graph_weight": self.graph_weight}
        base_config = super(TDGNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
