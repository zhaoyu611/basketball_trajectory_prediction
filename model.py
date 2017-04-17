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

        self.X = tf.placeholder(dtype=tf.float32, shape=[
            None, self.seq_len, self.crd_num], name="input_data")
        self.y = tf.placeholder(dtype=tf.int64, shape=[
                                None], name="ground_truth")
        self.drop_out = tf.placeholder(dtype=tf.float32, name="drop_out")
        self.W_out = tf.Variable(tf.random_normal(
            [self.hidden_size, self.y_out_dim], stddev=0.01), name="W_out")
        self.b_out = tf.Variable(tf.constant(
            0.1, shape=[self.y_out_dim]), name="b_out")

        # input_data is a [seq_len] length list, each has shape [None, crd_num]
        self.input_data = tf.unpack(self.X, axis=1)

    def LSTM_model(self):
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_layers)

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_out)
            outputs, _ = tf.nn.rnn(cell, self.input_data, dtype=tf.float32)

            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out

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
            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out

    def Conv_LSTM_model(self):
        """Here we have 2 Conv layers, followed by LSTM layers

        """
        with tf.name_scope('Conv_layer') as scope:
            # reshape the input data for Conv_LSTM
            conv_inputs = tf.reshape(
                self.X, [-1, self.seq_len, 1, self.crd_num])
            conv_W = {
                # 5x1 conv, crd_num inputs, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 1, self.crd_num, 32])),
                # 5x1 conv, 32 inputs, 1024 outputs
                'wc2': tf.Variable(tf.random_normal([5, 1, 32, 1024]))
            }
            conv_b = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([1024]))
            }
            def conv2d(X, W, b, stride=1):
                X = tf.nn.conv2d(X, W, [1, stride, stride, 1], padding='SAME')
                X = tf.nn.bias_add(X, b)
                return tf.nn.relu(X)

            def maxpool2d(X, k=2):
                return tf.nn.max_pool(X, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

            conv1 = conv2d(conv_inputs, conv_W['wc1'], conv_b['bc1'])
            conv1 = maxpool2d(conv1)

            conv2 = conv2d(conv1, conv_W['wc2'], conv_b['bc2'])
            conv2 = maxpool2d(conv2)
            # now the conv2 has shape [None, seq_len/k/k, 1, 1024]
            conv_outputs = tf.squeeze(conv2, 2)
            conv_outputs = tf.unpack(conv_outputs, axis=1)
            # now the conv_outputs is a seq_len/k/k length list,
            # each has shape [None, 1024]
        # stack LSTM layers
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_layers)

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_out)
            outputs, _ = tf.nn.rnn(cell, conv_outputs, dtype=tf.float32)

            self.y_pred = tf.matmul(outputs[-1], self.W_out) + self.b_out

    def CNN_model(self):
        """Here we have 2 Conv layers, followed by LSTM layers

        """
        with tf.name_scope('Conv_layer') as scope:
            # reshape the input data for Conv_LSTM
            conv_inputs = tf.reshape(
                self.X, [-1, self.seq_len, 1, self.crd_num])
            conv_W = {
                # 5x1 conv, crd_num inputs, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 1, self.crd_num, 32])),
                # 5x1 conv, 32 inputs, 1024 outputs
                'wc2': tf.Variable(tf.random_normal([5, 1, 32, 1024])),
                'wo': tf.Variable(tf.random_normal([1024, 2]))
            }
            conv_b = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([1024])),
                'bo': tf.Variable(tf.random_normal([2]))
            }
            def conv2d(X, W, b, stride=1):
                X = tf.nn.conv2d(X, W, [1, stride, stride, 1], padding='SAME')
                X = tf.nn.bias_add(X, b)
                return tf.nn.relu(X)

            def maxpool2d(X, k=2):
                return tf.nn.max_pool(X, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

            conv1 = conv2d(conv_inputs, conv_W['wc1'], conv_b['bc1'])
            conv1 = maxpool2d(conv1)

            conv2 = conv2d(conv1, conv_W['wc2'], conv_b['bc2'])
            conv2 = maxpool2d(conv2)
            # now the conv2 has shape [None, seq_len/k/k, 1, 1024]
            conv_outputs = tf.squeeze(conv2, 2)
            print conv_outputs
            outputs = tf.unpack(conv_outputs, axis=1)
            print outputs
            outputs = outputs[-1]
            print outputs
            outputs = tf.matmul(outputs, conv_W['wo']) + conv_b['bo']
            self.y_pred = outputs
            print self.y_pred
            print '============'

            # now the conv_outputs is a seq_len/k/k length list,
            # each has shape [None, 1024]
        # stack LSTM layers

    def Evaluating(self):
        with tf.name_scope("evaluating") as scope:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.y_pred, self.y)
            self.cost = tf.reduce_mean(self.loss)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
            self.correct_pred = tf.equal(tf.argmax(self.y_pred, 1), self.y)
            print self.y
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, tf.float32))

            # calculate training parameters number
            tvars = tf.trainable_variables()
            self.numel = tf.reduce_sum([tf.size(var) for var in tvars])
