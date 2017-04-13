import numpy
import tensorflow as tf
from util_MDN import *
from tensorflow.python.framework import ops

class Model():
    def __init__(self, args):
        """Hyperparameters"""
        num_layers = args.Num_layers
        hidden_size = args.hidden_size
        batch_size = args.batch_size
        max_grad_norm = args.max_grad_norm
        sl = args.sl
        mixtures = args.mixtures
        crd = args.crd_num
        learning_rate = args.lerning_rate
        MDN = args.MDN

        self.sl = sl
        self.crd = crd
        self.batch_size = batch_size

        # define the input
        self.x = tf.placeholder(dtype=tf.float32, shape=[
                                batch_size, crd, sl], name="input_data")
        self.y_ = tf.placeholder(dtype=tf.int64, shape=[
                                 batch_size], name="ground_truth")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="drop_out")
        # define 2 LSTM layers
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.keep_prob)

            # initial state
            initial_state = cell.zero_state(batch_size, tf.float32)
            outputs = []
            self.states = []
            state = initial_state
            for time_step in range(sl):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(
                    self.x[:, :, time_step], initial_state)
                outputs.append(cell_output)
                self.states.append(state)
            final_state = state

            # outputs is now a list of length sl, with tensor [batch_size,
            # hidden_size]
        with tf.name_scope("SoftMax") as scope:
            final = outputs[-1]
            W_c = tf.Variable(tf.random_normal([hidden_size, 2], stddev=0.01))
            # shape of h_c: [batch_size, 2], one-hot for probability of hit and miss
            b_c = tf.Variable(tf.constant(0.1, shape=[2]))
            self.h_c = tf.matmul(final, W_c) + b_c
            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.h_c, self.y_)
            self.cost = tf.reduce_mean(loss)
            loss_sum = tf.scalar_summary("cross entropy_loss", self.cost)

        with tf.name_scope("Output_MDN") as scope:
            params = 8  # mu1, mu2, mu3, sig1, sig2, sig3, coref, theata
            # mean and std for x, y, z normal_distribution,
            # and coralation efficient and theata to judge whether hit
            output_units = mixtures * params
            W_o = tf.Variable(tf.random_normal(
                [hidden_size, output_units], stddev=0.01))
            b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))
            # for comparison with XYZ, only up to the last time_step
            outputs_tensor = tf.concat(0, outputs[:-1])
            # outputs_tensor is now tensor [(sl-1)*batch_size, hidden_size]
            h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)
            # outputs_tensor is now tensor [(sl-1)*batch_size, output_units]
        with tf.name_scope("MDN_over_next_vector") as scope:
            # reshape the h_xyz to have shape [batch_size, output_unit,sl-1]
            h_xyz = tf.reshape(
                h_out_tensor, (sl - 1, batch_size, output_units))
            h_xyz = tf.transpose(h_xyz, [1, 2, 0])
            # calculate the difference between the sequence, shape: [n, crd,
            # sl]
            x_next = tf.sub(self.x[:, :3, 1:], self.x[:, :3, :-1])
            xn1, xn2, xn3 = tf.split(1, 3, x_next)
            self.mu1, self.mu2, self.mu3, self.s1, self.s2, self.s3, self.rho, self.theata = tf.split(
                1, params, h_xyz)

            # softmax all the theata's
            max_theata = tf.reduce_mean(self.theata, 1, keep_dims=True)
            self.theata = tf.sub(self.theata, max_theata)
            self.theata = tf.exp(self.theata)
            normalize_theata = tf.inv(
                tf.reduce_sum(self.theata, 1, keep_dims=True))
            self.theata = tf.mul(normalize_theata, self.theata)

            # deviances are non-negative and tho between -1 and 1
            self.s1 = tf.exp(self.s1)
            self.s2 = tf.exp(self.s2)
            self.s3 = tf.exp(self.s3)
            self.rho =tf.tanh(self.rho)

            px1x2 = tf_2d_normal(xn1, xn2, self.mu1, self.mu2, self.s1, self.s2, self.rho)   #probability in x1x2 plane
            px3 = tf_1d_normal(xn3,self.mu3,self.s3)
            px1x2x3 = tf.mul(px1x2,px3)
            #sum along the mixtures in dim 1
            px1x2x3_mixed = tf.reduce_sum(tf.mul(px1x2x3, self.theata), 1)
            print "You are using {} mixtrues".format(mixtures)
            loss_seq = -tf.log(tf.maximum(px1x2x3_mixed, 1e20))
            self.cost_seq = tf.reduce_mean(loss_seq)
            self.cost_comb = self.cost
            #the magic line where both heads come together
            if MDN:    self.cost_comb  = self.cost_comb + self.cost_seq
        with tf.name_scope("train") as scope:
            tvars = tf.trainable_variables()
            
            #We clip the gradients to prevent explosion
            grads = tf.gradients(self.cost_comb, tvars)
            grads, _ = tf.clip_by_global_norm(grads,0.5)

            #Some decay on the learning rate
            global_step = tf.Variable(0,trainable=False)
            lr = tf.train.exponential_decay(learning_rate,global_step,14000,0.95,staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = zip(grads, tvars)
            self.train_step = optimizer.apply_gradients(gradients,global_step=global_step)
            # The following block plots for every trainable variable
            #  - Histogram of the entries of the Tensor
            #  - Histogram of the gradient over the Tensor
            #  - Histogram of the grradient-norm over the Tensor
            self.numel = tf.constant([[0]])
            for gradient, variable in gradients:
                if isinstance(gradient, ops.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient
                self.numel +=tf.reduce_sum(tf.size(variable))

        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(self.h_c, 1), self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        #Define one op to call all summaries
        self.merged =tf.merge_all_summaries()
