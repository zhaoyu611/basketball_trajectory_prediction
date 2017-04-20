import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util_MDN import *


class Model():
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.crd_num = 4  # including coordinate X, Y, Z and game clock
        self.hidden_size = args.hidden_size
        self.y_out_dim = 2  # the predict output should be one-hot(hit, miss),
        self.hidden_layers = args.hidden_layers
        self.dist = args.dist
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        # two for distribution over hit&miss, params for distribution
        # parameters
        self.mixtures = 3  # num of mixture denesity netowrks
        self.use_MDN = False  # if use MDN model

        self.X = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_len, self.crd_num], name="input_data")
        self.y = tf.placeholder(dtype=tf.int64, shape=[
                                self.batch_size], name="ground_truth")
        self.drop_out = tf.placeholder(dtype=tf.float32, name="drop_out")
        self.W_out = tf.Variable(tf.random_normal(
            [self.hidden_size, self.y_out_dim], stddev=0.01), name="W_out")
        self.b_out = tf.Variable(tf.constant(
            0.1, shape=[self.y_out_dim]), name="b_out")

        # input_data is a [seq_len] length list, each has shape
        # [self.batch_size, crd_num]
        self.input_data = tf.unpack(self.X, axis=1)

    def LSTM_model(self):
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(
                self.hidden_size, use_peepholes=True)

            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_layers)

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_out)
            outputs, _ = tf.nn.rnn(cell, self.input_data, dtype=tf.float32)
            # outputs is a list of seq_len length, each has shape [batch_size,
            # hidden_size]
            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out
        return outputs

    def bidir_LSTM_model(self):
        with tf.name_scope("bidir_LSTM") as scope:
            assert self.hidden_size % 2 == 0, "hidden_size must be even number for bidir-LSTM"
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size / 2)
            cells_fw = [cell] * self.hidden_layers
            cells_bw = [cell] * self.hidden_layers

            pre_layer = self.input_data
            for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
                with vs.variable_scope("cell{}".format(i)) as cell_scope:
                    pre_layer, _, _ = tf.nn.bidirectional_rnn(
                        cell_fw, cell_bw, pre_layer, dtype=tf.float32)
            outputs = pre_layer
            outputs[-1] = tf.nn.dropout(outputs[-1], self.drop_out)
            # outputs is a list of seq_len length, each has shape [batch_size,
            # hidden_size]
            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out
            return outputs

    def CNN_model(self):
        """Here we have 2 Conv layers, followed by LSTM layers

        """
        with tf.name_scope('Conv_layer') as scope:
            # reshape the input data for Conv_LSTM
            conv_inputs = tf.reshape(
                self.X, [-1, self.seq_len, 1, self.crd_num])
            conv_W = {
                # 5x1 conv, crd_num inputs, 32 outputs
                'wc1': tf.Variable(tf.random_normal([1, 1, self.crd_num, 32])),
                # 5x1 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([1, 1, 32, 64])),
                'wo': tf.Variable(tf.random_normal([64, 2]))
            }
            conv_b = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bo': tf.Variable(tf.random_normal([2]))
            }
            def conv2d(X, W, b, stride=1):
                X = tf.nn.conv2d(X, W, [1, stride, stride, 1], padding='SAME')
                X = tf.nn.bias_add(X, b)
                return tf.nn.relu(X)

            def maxpool2d(X, k=2):
                return tf.nn.max_pool(X, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

            conv1 = conv2d(conv_inputs, conv_W['wc1'], conv_b['bc1'])
            # now conv1 has shape [self.batch_size, seq_len, 1, 32]
            conv1 = maxpool2d(conv1)
            # now conv1 has shape [self.batch_size, seq_len/k, 1, 32]

            conv2 = conv2d(conv1, conv_W['wc2'], conv_b['bc2'])
            # now conv2 has shape [self.batch_size, seq_len/k, 1, 64]
            conv2 = maxpool2d(conv2)
            # now conv2 has shape [self.batch_size, seq_len/k/k, 1, 64]
            conv_outputs = tf.squeeze(conv2, 2)
            # now conv_outputs has shape [self.batch_size, seq_len/k/k, 64]
            outputs = tf.unpack(conv_outputs, axis=1)
            # now outputs are a list of seq_len/k/k length, each has shape
            # [self.batch_size, 64]
            outputs = outputs[-1]
            # now outputs is the last in the list, has shape [self.batch_size,
            # 64]
            outputs = tf.matmul(outputs, conv_W['wo']) + conv_b['bo']
            # now outputs has shape [self.batch_size, 2]
            self.y_pred = outputs

    def Conv_LSTM_model(self):
        with tf.name_scope('Conv_layer') as scope:
            # reshape the input data for Conv_LSTM
            conv_inputs = tf.reshape(
                self.X, [-1, self.seq_len, 1, self.crd_num])
            conv_W = {
                # 5x1 conv, crd_num inputs, 32 outputs
                'wc1': tf.Variable(tf.random_normal([1, 1, self.crd_num, 32])),
                # 5x1 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([1, 1, 32, 64])),
                'wo': tf.Variable(tf.random_normal([64, 2]))
            }
            conv_b = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bo': tf.Variable(tf.random_normal([2]))
            }
            def conv2d(X, W, b, stride=1):
                X = tf.nn.conv2d(X, W, [1, stride, stride, 1], padding='SAME')
                X = tf.nn.bias_add(X, b)
                return tf.nn.relu(X)

            def maxpool2d(X, k=2):
                return tf.nn.max_pool(X, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

            conv1 = conv2d(conv_inputs, conv_W['wc1'], conv_b['bc1'])
            # now conv1 has shape [self.batch_size, seq_len, 1, 32]
            conv1 = maxpool2d(conv1)
            # now conv1 has shape [self.batch_size, seq_len/k, 1, 32]

            conv2 = conv2d(conv1, conv_W['wc2'], conv_b['bc2'])
            # now conv2 has shape [self.batch_size, seq_len/k, 1, 64]
            conv2 = maxpool2d(conv2)
            # now conv2 has shape [self.batch_size, seq_len/k/k, 1, 64]
            conv_outputs = tf.squeeze(conv2, 2)
            # now conv_outputs has shape [self.batch_size, seq_len/k/k, 64]
            conv_outputs_list = tf.unpack(conv_outputs, axis=1)
            # now outputs are a list of seq_len/k/k length, each has shape
            # [self.batch_size, 64]

        # stack LSTM layers
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_layers)

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_out)
            outputs, _ = tf.nn.rnn(cell, conv_outputs_list, dtype=tf.float32)
            print outputs
            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out

    def MDN_model(self, LSTM_type='LSTM'):
        """ define mixture denisty network
            argument: 
                LSTM_type: [str] use 'LSTM' or  'BLSTM' before the MDN 
        """
        self.use_MDN = True
        if LSTM_type == 'LSTM':
            outputs = self.LSTM_model()
        elif LSTM_type == 'BLSTM':
            outputs = self.bidir_LSTM_model()
        else:
            raise "You should specify the right model before run MDN"
        
        with tf.name_scope("Output_MDN") as scope:
            params = 8  # 7+theta
            # Two for distribution over hit&miss, params for distribution
            # parameters
            output_units = self.mixtures * params
            W_o = tf.Variable(tf.random_normal(
                [self.hidden_size, output_units], stddev=0.01))
            b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))
            # For comparison with XYZ, only up to last time_step
            # --> because for final time_step you cannot make a prediction
            outputs_tensor = tf.concat(0, outputs[:-1])
            # is of size [batch_size*seq_len by output_units]
            h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)

        with tf.name_scope('MDN_over_next_vector') as scope:
            # Next two lines are rather ugly, But its the most efficient way to
            # reshape the data
            h_xyz = tf.reshape(h_out_tensor, (self.seq_len - 1, self.batch_size, output_units))
            # transpose to [batch_size, output_units, sl-1]
            h_xyz = tf.transpose(h_xyz, [1, 2, 0])
            # x_next = tf.slice(x,[0,0,1],[batch_size,3,sl-1])  #in size [batch_size,
            # output_units, sl-1]
            MDN_X = tf.transpose(self.X, [0, 2, 1])
            x_next = tf.sub(MDN_X[:, :3, 1:], MDN_X[:, :3, :self.seq_len - 1])
            # From here any, many variables have size [batch_size, mixtures, sl-1]
            xn1, xn2, xn3 = tf.split(1, 3, x_next)
            self.mu1, self.mu2, self.mu3, self.s1, self.s2, self.s3, self.rho, self.theta = tf.split(
                1, params, h_xyz)

            # make the theta mixtures
            # softmax all the theta's:
            max_theta = tf.reduce_max(self.theta, 1, keep_dims=True)
            self.theta = tf.sub(self.theta, max_theta)
            self.theta = tf.exp(self.theta)
            normalize_theta = tf.inv(tf.reduce_sum(self.theta, 1, keep_dims=True))
            self.theta = tf.mul(normalize_theta, self.theta)

            # Deviances are non-negative and tho between -1 and 1
            self.s1 = tf.exp(self.s1)
            self.s2 = tf.exp(self.s2)
            self.s3 = tf.exp(self.s3)
            self.rho = tf.tanh(self.rho)

            # probability in x1x2 plane
            px1x2 = tf_2d_normal(xn1, xn2, self.mu1, self.mu2,
                                 self.s1, self.s2, self.rho)
            px3 = tf_1d_normal(xn3, self.mu3, self.s3)
            px1x2x3 = tf.mul(px1x2, px3)

            # Sum along the mixtures in dimension 1
            px1x2x3_mixed = tf.reduce_sum(tf.mul(px1x2x3, self.theta), 1)
            print('You are using %.0f mixtures' % self.mixtures)
            # at the beginning, some errors are exactly zero.
            loss_seq = -tf.log(tf.maximum(px1x2x3_mixed, 1e-20))
            self.cost_seq = tf.reduce_mean(loss_seq)


    def Evaluating(self):
        with tf.name_scope("evaluating") as scope:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.y_pred, self.y)
            self.cost = tf.reduce_mean(self.loss)

            if self.use_MDN:  # if use MDN, then add cost_seq to cost
                self.cost += self.cost_seq

            # global_step = tf.Variable(0, trainable=False)
            # lr = tf.train.exponential_decay(self.learning_rate, global_step, 14000, 0.95, staircase=True)
            # self.train_op = tf.train.AdamOptimizer(
            #     learning_rate=self.learning_rate).minimize(self.cost)

            # self.train_op = tf.train.AdamOptimizer(
            #     learning_rate=self.learning_rate).minimize(self.cost)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, tvars), 1)
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(
                self.learning_rate, global_step, 14000, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)

            self.correct_pred = tf.equal(tf.argmax(self.y_pred, 1), self.y)

            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, tf.float32))

            # calculate training parameters number

            self.numel = tf.reduce_sum([tf.size(var) for var in tvars])
