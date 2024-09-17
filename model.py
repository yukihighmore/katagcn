from utils import *
from layers import *
from tensorflow import keras
import keras.backend as kb
import time


def cosine(emb_a, emb_b):
    fz = kb.sum((emb_a * emb_b), axis=1)
    fm = kb.sqrt(kb.sum(kb.square(emb_a), axis=1)) * kb.sqrt(kb.sum(kb.square(emb_b), axis=1))
    return fz / fm


class KaTaGCN(keras.Model):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, graph_emb,
                 std, mean, top_corr, alpha, name="KaTaGCN", **kwargs):
        super(KaTaGCN, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.graph_emb = graph_emb
        self.std = std
        self.mean = mean
        self.top_corr = top_corr
        self.alpha = alpha
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, graph_emb,
                               name=self.name + '_kagconv_')

        # time linear
        self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
                                 name=self.name + '_taline')

        # GLU
        self.semlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_stmlp_')
        self.temlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_temlp_')

        if dilation:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))
        else:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))

        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None, use_bias=False)

    @tf.function
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
            tge = tf.reshape(x_embdings,
                             shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            # gte -> b, g, t, 2e
            gte = tf.reshape(x_embdings,
                             shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])
        else:
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])
        '''

        x_embdings = tf.concat([x_embdings, x_embdings_g], axis=-1)

        tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
        gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        '''
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

    def get_loss(self, y, y_pred):
        loss = tf.keras.losses.MeanAbsoluteError(
            reduction="sum_over_batch_size", name="mean_absolute_error"
        )
        losse = loss(y, y_pred)
        alpha = self.alpha
        relation_pos_emb = self.kagconv.srpe
        relation_pos_emb_T = tf.transpose(relation_pos_emb, perm=[1, 0, 2])
        relation_pos = tf.einsum('vge, gie->vie', relation_pos_emb, relation_pos_emb_T)
        #relation_pos = self.kagconv.relu_g(relation_pos)

        s_t = pd.read_csv(self.top_corr, index_col=None, header=None).values
        s_t_n = len(s_t)
        emb_a = []
        for i in range(s_t_n):
            emb_a.append(relation_pos[int(s_t[i, 0]), int(s_t[i, 1]), :])
        emb_a = tf.stack(emb_a)
        k = kb.max(kb.abs(emb_a)) // 10 + 1
        s_t_e_s = kb.mean(emb_a / (10 ** k))
        #s_t_e_s = kb.mean(emb_a)

        weightPunish = alpha * (- s_t_e_s)
        losse = losse + weightPunish
        return losse

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        #time_start = time.time()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # regular_loss = self.get_loss(y, y_pred)
            #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = self.get_loss(y, y_pred)
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        mae = self.metrics[1].result()
        #time_end = time.time()
        #print(time_end - time_start)
        #return {m.name: m.result() for m in self.metrics}
        return {'last_batch_loss': loss, 'mae': mae}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # y_stats = PeMS.get_stats()
        # y = z_inverse(y, y_stats['mean'], y_stats['std'])

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        #mae = self.metrics[0].result()
        return {m.name: m.result() for m in self.metrics}


class CustomModel(keras.Model):
    def __init__(self, graph_dim, n_his, n_pre, emb_dim, mlp, day_stamps, week_states, week_flag, dilation, graph_emb,
                 std, mean, top_corr, alpha, name="TaKaGCN", **kwargs):
        super(CustomModel, self).__init__(name=name, **kwargs)
        self.graph_dim = graph_dim
        self.n_his = n_his
        self.n_pre = n_pre
        self.emb_dim = emb_dim
        self.mlp = mlp
        self.day_stamps = day_stamps
        self.week_states = week_states
        self.week_flag = week_flag
        self.dilation = dilation
        self.graph_emb = graph_emb
        self.std = std
        self.mean = mean
        self.top_corr = top_corr
        self.alpha = alpha
        self.x2embding = tf.keras.layers.Conv2D(filters=emb_dim, kernel_size=1, name="x2embding" + "_" + self.name)
        # kagcn
        self.kagconv = kagconv(graph_dim, n_his, emb_dim, day_stamps, week_states, graph_emb,
                               name=self.name + '_kagconv_')

        # time linear
        self.talinear = talinear(graph_dim, n_his, emb_dim, day_stamps, week_states, dilation, week_flag,
                                 name=self.name + '_taline')

        # GLU
        self.semlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_stmlp_')
        self.temlp = MLP(n_pre, emb_dim, mlp, name=self.name + '_temlp_')

        if dilation:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))
        else:
            self.tge2tg = self.add_weight(name=self.name + '_tge2tg', shape=(graph_dim, emb_dim, graph_dim))
            self.gte2gt = self.add_weight(name='gte2gt' + self.name, shape=(n_his, emb_dim, n_his))

        self.fully = tf.keras.layers.Dense(name=self.name + '_full', units=n_pre, activation=None, use_bias=False)

    @tf.function
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
            tge = tf.reshape(x_embdings,
                             shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            # gte -> b, g, t, 2e
            gte = tf.reshape(x_embdings,
                             shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])
        else:
            tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, self.emb_dim])
            gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, self.emb_dim])
        '''

        x_embdings = tf.concat([x_embdings, x_embdings_g], axis=-1)

        tge = tf.reshape(x_embdings, shape=[batch_size, self.n_his, self.graph_dim, 2 * self.emb_dim])
        gte = tf.reshape(x_embdings, shape=[batch_size, self.graph_dim, self.n_his, 2 * self.emb_dim])
        '''
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

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # y_stats = PeMS.get_stats()
            # y = z_inverse(y, y_stats['mean'], y_stats['std'])
            # y_pred = z_inverse(y_pred, y_stats['mean'], y_stats['std'])
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # y_stats = PeMS.get_stats()
        # y = z_inverse(y, y_stats['mean'], y_stats['std'])

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    s_t = pd.read_csv('dataset/pairs_top.csv', index_col=None, header=None).values
