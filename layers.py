import tensorflow as tf

#tf.config.run_functions_eagerly(True)
from utils import *

seed = 2023


class MLP(tf.keras.layers.Layer):
    def __init__(self, n_pre, emb_dim, layer_num, name="MLP", **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.layer_num = layer_num
        self.relus = {}
        self.linears = {}
        for i in range(layer_num):
            if i == layer_num - 1:
                self.relus[str(i)] = tf.keras.layers.Dense(name=self.name + 'relu_' + str(i), units=emb_dim * n_pre,
                                                           activation='relu')
                self.linears[str(i)] = tf.keras.layers.Dense(name=self.name + 'line_' + str(i), units=n_pre,
                                                             activation=None)
            else:
                self.relus[str(i)] = tf.keras.layers.Dense(name=self.name + 'relu_' + str(i), units=emb_dim * n_pre,
                                                           activation='relu')
                self.linears[str(i)] = tf.keras.layers.Dense(name=self.name + 'line_' + str(i), units=emb_dim * n_pre,
                                                             activation=None)

    def call(self, inputs):
        if self.layer_num > 0:
            for i in range(self.layer_num):
                inputs = self.relus[str(i)](inputs)
                inputs = self.linears[str(i)](inputs)

        return inputs

    def get_config(self):
        config = {"n_pre": self.n_pre, "emb_dim": self.emb_dim, "layer_num": self.layer_num}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class kagconv(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, emb_dim, day_stamps, week_states, graph_emb, name="kagconv",
                 **kwargs):
        super(kagconv, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.emb_dim = emb_dim
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.graph_emb = graph_emb
        # learnable ralative position embdding
        #self.srpe = self.add_weight(name=self.name + '_SRPE', shape=(graph_dim * graph_dim, emb_dim),
                                    #initializer=tf.keras.initializers.glorot_normal(seed))
        #self.srpe = self.add_weight(name=self.name + '_SRPE', shape=(graph_dim, emb_dim),
         #                           initializer=tf.keras.initializers.glorot_normal(seed))
        self.srpe = self.add_weight(name=self.name + '_SRPE', shape=(graph_dim, graph_emb, emb_dim),
                                    initializer=tf.keras.initializers.glorot_normal(seed))
        self.relu_g = tf.keras.layers.ReLU()
        # position embdding
        self.sape = self.add_weight(name=self.name + '_SAPE', shape=(graph_dim, n_his, emb_dim),
                                    initializer=tf.keras.initializers.glorot_normal(seed))
        # stamp on a day embdding
        self.dow_emb = self.add_weight(name=self.name + '_day_of_week', shape=(week_states, graph_dim, emb_dim),
                                       initializer=tf.keras.initializers.glorot_normal(seed))
        # state on a week embdding
        self.tod_emb = self.add_weight(name=self.name + '_time_of_day', shape=(day_stamps, graph_dim, emb_dim),
                                       initializer=tf.keras.initializers.glorot_normal(seed))
        # connect
        self.connect_weight = self.add_weight(name=self.name + '_connect_weight',
                                              shape=[n_his, graph_dim, 3 * emb_dim, emb_dim],
                                              initializer=tf.keras.initializers.glorot_normal(seed))
        self.connect_bias = self.add_weight(name=self.name + '_connect_bias',
                                            shape=[n_his, graph_dim, emb_dim],
                                            initializer=tf.keras.initializers.Zeros())
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.3, name=self.name + '_relu')
        #self.drop = tf.keras.layers.Dropout(name=self.name + '_drop', rate=0.5)

    def call(self, inputs, n_route, s_week, s_day):
        # x_embdings -> b, t, g, e
        #relation_pos_emb = tf.gather(self.srpe, self.SDist)
        #relation_pos_emb = tf.reshape(relation_pos_emb, shape=[self.graph_dim, self.graph_dim, self.emb_dim])
        relation_pos_emb = self.srpe
        relation_pos_emb_T = tf.transpose(relation_pos_emb, perm=[1, 0, 2])
        relation_pos = tf.einsum('vge, gie->vie', relation_pos_emb, relation_pos_emb_T)
        relation_pos = self.relu_g(relation_pos)
        x_rel_emb = tf.einsum('vge, btge->btve', relation_pos, inputs)
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
        outputs = self.relu(x_embdings)
        #outputs = self.drop(outputs)

        return outputs

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "emb_dim": self.emb_dim,
                  "day_stamps": self.day_stamps, "week_states": self.week_states, "graph_emb": self.graph_emb}
        base_config = super(kagconv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class talinear(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag, name="talinear",
                 **kwargs):
        super(talinear, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.emb_dim = emb_dim
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.dilation = dilation
        self.week_flag = week_flag
        if dilation:
            if week_flag:
                self.week_weight_space = self.add_weight(name=self.name + '_wws',
                                                         shape=[2, n_his, emb_dim, emb_dim, n_his],
                                                         initializer=tf.keras.initializers.glorot_normal(seed),
                                                         trainable=True)
                self.week_bias_space = self.add_weight(name=self.name + '_wbs',
                                                       shape=[2, graph_dim,  emb_dim, n_his],
                                                       initializer=tf.keras.initializers.Zeros(),
                                                       trainable=True)
                self.drop = tf.keras.layers.Dropout(name=self.name + '_drop', rate=0.5)
            else:
                self.week_weight_space = self.add_weight(name=self.name + '_wws',
                                                         shape=[week_states, n_his, emb_dim,  emb_dim, n_his],
                                                         initializer=tf.keras.initializers.glorot_normal(seed),
                                                         trainable=True)
                self.week_bias_space = self.add_weight(name=self.name + '_wbs',
                                                       shape=[week_states, graph_dim,  emb_dim, n_his],
                                                       initializer=tf.keras.initializers.Zeros(),
                                                       trainable=True)
                self.drop = tf.keras.layers.Dropout(name=self.name + '_drop', rate=0.5)
            self.day_weight_space = self.add_weight(name=self.name + '_dws',
                                                    shape=[day_stamps, n_his, emb_dim,  emb_dim, n_his],
                                                    initializer=tf.keras.initializers.glorot_normal(seed),
                                                    trainable=True)
            self.day_bias_space = self.add_weight(name=self.name + '_dbs',
                                                  shape=[day_stamps, graph_dim,  emb_dim, n_his],
                                                  initializer=tf.keras.initializers.Zeros(),
                                                  trainable=True)
            self.relulast = tf.keras.layers.LeakyReLU(alpha=0.3, name=self.name + '_relulast')

        else:
            if week_flag:
                self.day_weight_space = self.add_weight(name=self.name + '_dws',
                                                        shape=[day_stamps, n_his, emb_dim,  emb_dim, n_his],
                                                        initializer=tf.keras.initializers.glorot_normal(seed),
                                                        trainable=True)
                self.day_bias_space = self.add_weight(name=self.name + '_dbs',
                                                      shape=[day_stamps, graph_dim, emb_dim, n_his],
                                                      initializer=tf.keras.initializers.Zeros(),
                                                      trainable=True)
                self.relulast = tf.keras.layers.LeakyReLU(alpha=0.3, name=self.name + '_relulast')
            else:
                self.day_weight_space = self.add_weight(name=self.name + '_dws',
                                                    shape=[day_stamps, emb_dim, n_his],
                                                    initializer=tf.keras.initializers.glorot_normal(seed),
                                                    trainable=True)
                self.day_bias_space = self.add_weight(name=self.name + '_dbs',
                                                  shape=[day_stamps, graph_dim, emb_dim, n_his],
                                                  initializer=tf.keras.initializers.Zeros(),
                                                  trainable=True)
                self.relulast = tf.keras.layers.ReLU(name=self.name + '_relulast')

            self.ln = tf.keras.layers.LayerNormalization()

            self.drop = tf.keras.layers.Dropout(name=self.name + '_drop', rate=0.8)


    def call(self, inputs, week, date):
        if self.dilation:
            week_weight = tf.gather(self.week_weight_space, week)
            week_bias = tf.gather(self.week_bias_space, week)
            date_weight = tf.gather(self.day_weight_space, date)
            date_bias = tf.gather(self.day_bias_space, date)
            # i -> e, j -> t
            x_ = tf.einsum('bget,bteij->bgij', inputs, week_weight)
            x_ = tf.add(x_, week_bias)
            x__ = tf.einsum('bget,bteij->bgij', inputs, date_weight)
            x__ = tf.add(x__, date_bias)
            x_embdings = (x_ + x__) * 0.5
            x_embdings = self.relulast(x_embdings)
            x_embdings = self.drop(x_embdings)
        else:
            date_weight = tf.gather(self.day_weight_space, date)
            date_bias = tf.gather(self.day_bias_space, date)
            if self.week_flag:
                x__ = tf.einsum('bget,bteij->bgij', inputs, date_weight)
                x__ = tf.add(x__, date_bias)
                x__ = self.ln(x__)
                x_embdings = self.relulast(x__)
            else:
                x__ = tf.einsum('bget,bet->bget', inputs, date_weight)
                x__ = tf.add(x__, date_bias)
                x__ = self.ln(x__)
                x_embdings = self.relulast(x__)
            x_embdings = self.drop(x_embdings)
        return x_embdings

    def get_config(self):
        # graph_dim, n_his, emb_dim, day_stamps, week_states, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "emb_dim": self.emb_dim,
                  "day_stamps": self.day_stamps, "week_states": self.week_states, "dilation": self.dilation,
                  "week_flag": self.week_flag}
        base_config = super(talinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
# TaKaGCN(graph_dim=N, n_his=P, n_pre=M, emb_dim=emb_dim, mlp=mlp, day_stamps=sod, week_states=sow,
#                  week_flag=u_w, dilation=dilation, Graph=Graph)
class TaKaGCN(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph,
                 std, mean, name="TaKaGCN", **kwargs):
        super(TaKaGCN, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.Graph = Graph
        self.std = std
        self.mean = mean
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, Graph, name=self.name + '_kagconv_')

        # time linear
        self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
                                 name=self.name + '_taline')

        # GLU

        self.semlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_stmlp_')
        self.temlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_temlp_')

        if dilation:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, 2 * emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, 2 * emb_dim, n_his))
        else:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))

        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None, use_bias=False)

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

        x_embdings = self.kagconv(x_embdings, n_route, s_week, s_day)

        # x_embdings -> b, t, g, e -> b, g, e, t
        x_embdings = tf.transpose(x_embdings, perm=[0, 2, 3, 1])
        x_embdings = self.talinear(x_embdings, week, date)
        # x_embdings -> b, g, e, t -> b, t, g, e
        x_embdings = tf.transpose(x_embdings, perm=[0, 3, 1, 2])

        # GLU
        if self.dilation:
            # tge -> b, t, g, 2e
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
            # gte -> b, g, t, 2e
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        else:
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])

        x_3 = tf.transpose(x_3, perm=[0, 2, 1])
        x1 = self.semlp(x_3)
        x2 = self.temlp(x_3)

        tg = tf.einsum('gev,btge->btv', self.tge2tg, tge)
        tg2gt = tf.transpose(tg, perm=[0, 2, 1])
        gt1 = (tg2gt + x1) * tf.sigmoid(tg2gt)
        gt = tf.einsum('teo,bgte->bgo', self.gte2gt, gte)
        gt2 = (gt + x2) * tf.sigmoid(gt)

        x = tf.concat([gt1, gt2], axis=-1)
        x = self.fully(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        output = z_inverse(x, self.mean, self.std)
        return output

    def get_config(self):
        # graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "n_pre": self.n_pre, "emb_dim": self.emb_dim,
                  "mlp": self.mlp, "day_stamps": self.day_stamps, "week_states": self.week_states,
                  "week_flag": self.week_flag,
                  "dilation": self.dilation, "Graph": self.Graph, "std": self.std, "mean": self.mean}
        base_config = super(TaKaGCN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''

# no KaGCN
class wokagconv(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph,
                 std, mean, name="wokagconv", **kwargs):
        super(wokagconv, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.Graph = Graph
        self.std = std
        self.mean = mean
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        # self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, Graph, name=self.name + '_kagconv_')

        # time linear
        self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
                                 name=self.name + '_taline')

        # GLU

        self.semlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_stmlp_')
        self.temlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_temlp_')

        if dilation:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, 2 * emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, 2 * emb_dim, n_his))
        else:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))

        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None, use_bias=False)

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

        # x_embdings = self.kagconv(x_embdings, n_route, s_week, s_day)

        # x_embdings -> b, t, g, e -> b, g, e, t
        x_embdings = tf.transpose(x_embdings, perm=[0, 2, 3, 1])
        x_embdings = self.talinear(x_embdings, week, date)
        # x_embdings -> b, g, e, t -> b, t, g, e
        x_embdings = tf.transpose(x_embdings, perm=[0, 3, 1, 2])

        # GLU
        if self.dilation:
            # tge -> b, t, g, 2e
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
            # gte -> b, g, t, 2e
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        else:
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])

        x_3 = tf.transpose(x_3, perm=[0, 2, 1])
        x1 = self.semlp(x_3)
        x2 = self.temlp(x_3)

        tg = tf.einsum('gev,btge->btv', self.tge2tg, tge)
        tg2gt = tf.transpose(tg, perm=[0, 2, 1])
        gt1 = (tg2gt + x1) * tf.sigmoid(tg2gt)
        gt = tf.einsum('teo,bgte->bgo', self.gte2gt, gte)
        gt2 = (gt + x2) * tf.sigmoid(gt)

        x = tf.concat([gt1, gt2], axis=-1)
        x = self.fully(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        output = z_inverse(x, self.mean, self.std)
        return output

    def get_config(self):
        # graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "n_pre": self.n_pre, "emb_dim": self.emb_dim,
                  "mlp": self.mlp, "day_stamps": self.day_stamps, "week_states": self.week_states,
                  "week_flag": self.week_flag,
                  "dilation": self.dilation, "Graph": self.Graph, "std": self.std, "mean": self.mean}
        base_config = super(wokagconv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# no TaLinear
class wotalinear(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph,
                 std, mean, name="wokagconv", **kwargs):
        super(wotalinear, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.Graph = Graph
        self.std = std
        self.mean = mean
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, Graph, name=self.name + '_kagconv_')

        # time linear
        # self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
        #                        name=self.name + '_taline')

        # GLU

        self.semlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_stmlp_')
        self.temlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_temlp_')

        if dilation:
            self.expand_dim = tf.keras.layers.Dense(2 * emb_dim)
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, 2 * emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, 2 * emb_dim, n_his))
        else:
            self.expand_dim = tf.keras.layers.Dense(emb_dim)
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))

        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None, use_bias=False)

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

        x_embdings = self.kagconv(x_embdings, n_route, s_week, s_day)

        # x_embdings -> b, t, g, e -> b, g, e, t
        # x_embdings = tf.transpose(x_embdings, perm=[0, 2, 3, 1])
        # x_embdings = self.talinear(x_embdings, week, date)
        # x_embdings -> b, g, e, t -> b, t, g, e
        # x_embdings = tf.transpose(x_embdings, perm=[0, 3, 1, 2])

        x_embdings = self.expand_dim(x_embdings)

        # GLU
        if self.dilation:
            # tge -> b, t, g, 2e
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
            # gte -> b, g, t, 2e
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        else:
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])

        x_3 = tf.transpose(x_3, perm=[0, 2, 1])
        x1 = self.semlp(x_3)
        x2 = self.temlp(x_3)

        tg = tf.einsum('gev,btge->btv', self.tge2tg, tge)
        tg2gt = tf.transpose(tg, perm=[0, 2, 1])
        gt1 = (tg2gt + x1) * tf.sigmoid(tg2gt)
        gt = tf.einsum('teo,bgte->bgo', self.gte2gt, gte)
        gt2 = (gt + x2) * tf.sigmoid(gt)

        x = tf.concat([gt1, gt2], axis=-1)
        x = self.fully(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        output = z_inverse(x, self.mean, self.std)
        return output

    def get_config(self):
        # graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "n_pre": self.n_pre, "emb_dim": self.emb_dim,
                  "mlp": self.mlp, "day_stamps": self.day_stamps, "week_states": self.week_states,
                  "week_flag": self.week_flag,
                  "dilation": self.dilation, "Graph": self.Graph, "std": self.std, "mean": self.mean}
        base_config = super(wotalinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# no STGLU
class wostglu(tf.keras.layers.Layer):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph,
                 std, mean, name="wostglu", **kwargs):
        super(wostglu, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.Graph = Graph
        self.std = std
        self.mean = mean
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, Graph, name=self.name + '_kagconv_')

        # time linear
        self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
                                 name=self.name + '_taline')

        # GLU
        self.res = MLP(n_pre, emb_dim, mlp, name=self.name + '_res_')
        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=1, activation='relu', use_bias=False)

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

        x_embdings = self.kagconv(x_embdings, n_route, s_week, s_day)

        # x_embdings -> b, t, g, e -> b, g, e, t
        x_embdings = tf.transpose(x_embdings, perm=[0, 2, 3, 1])
        x_embdings = self.talinear(x_embdings, week, date)
        # x_embdings -> b, g, e, t

        x_3 = tf.transpose(x_3, perm=[0, 2, 1])
        x1 = self.res(x_3)
        if self.dilation:
            x = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        else:
            x = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])
        x = self.fully(x)
        x1 = tf.transpose(x1, perm=[0, 2, 1])
        x = tf.reshape(x, shape=[batch_size, self.n_his, self.graph_dim]) + x1
        output = z_inverse(x, self.mean, self.std)
        return output

    def get_config(self):
        # graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, Graph
        config = {"graph_dim": self.graph_dim, "n_his": self.n_his, "n_pre": self.n_pre, "emb_dim": self.emb_dim,
                  "mlp": self.mlp, "day_stamps": self.day_stamps, "week_states": self.week_states,
                  "week_flag": self.week_flag,
                  "dilation": self.dilation, "Graph": self.Graph, "std": self.std, "mean": self.mean}
        base_config = super(wostglu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
