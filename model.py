import tensorflow as tf


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

        self.input_data = tf.unpack(self.X, axis=1)

    def LSTM_model(self):
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_layers)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.drop_out)
            output, _ = tf.nn.rnn(cell, self.input_data, dtype=tf.float32)
            self.y_pred = tf.matmul(output[-1], self.W_out) + self.b_out

    def Evaluating(self):
        with tf.name_scope("evaluating") as scope:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.y_pred, self.y)
            self.cost = tf.reduce_mean(self.loss)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
            self.correct_pred = tf.equal(tf.argmax(self.y_pred, 1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            #calculate training parameters number
            tvars = tf.trainable_variables()
            self.numel = tf.reduce_sum([tf.size(var) for var in tvars])