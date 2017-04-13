from dataloader import DataLoad
import argparse
import tensorflow as tf
import numpy as np
from model import *


def preprocess_data(args):
    direc = './data/'
    csv_file = 'seq_all.csv'
    center = np.array([5.25, 25.0, 10.0])
    height_low_bound = 11.
    dataLoader = DataLoad(direc, csv_file, center)  # create a dataloader class
    dataLoader.munge_data(height_low_bound, args.sl, args.dist_2_baket)
    dataLoader.center_data(center)
    dataLoader.test_valid_data_split(args.split_ratio)
    return dataLoader


def train(args):
    """ input all arguments, and train model
    """
    #=======step 1: preprocess data=========
    dl = preprocess_data(args)  # return the prerprocessed dataloader class
    #=======================================
    #===========step 2: construct model=====

    # reshape X_train, shape: [n, seq_len, coord] ---> [n, coord, seq_len]
    data_dict = dl.data
    X_train = np.transpose(data_dict['X_train'], [0, 2, 1])
    y_train = data_dict['y_train']
    X_test = np.transpose(data_dict['X_test'], [0, 2, 1])
    y_test = data_dict['y_test']
    N, crf, _ = X_train.shape
    Ntest = X_test.shape[0]
    epoch = np.floor(args.batch_size * args.max_interactions / dl.N)
    print "train with approximately {} epochs".format(epoch)

    model = Model(args)

    #=======================================
    #===========step 3: train model=========
    plot_every = 100
    perf_collect = np.zeros(
        (7, int(np.floor(args.max_interactions / plot_every))))
    sess = tf.Session()
    # initial settings for early stopping
    auc_ma = 0.0
    auc_best = 0.0

    sess.run(tf.initialize_all_variables())
    step = 0  # step is a counter for filling perf_collect
    i = 0
    early_stop = False
    args.max_interactions=3
    while i < args.max_interactions and not early_stop:
        if i%plot_every == 0:
            #check training performance
            if MDN:
                fetch=[model.accuracy, model.cost, model.cost_seq]
            else:
                fetch=[model.accuracy, model.cost]

            result= sess.run(fetch, feed_dict={model.x: })

        batch_ind = np.random.choice(N, args.batch_size, replace=False)

        train_acc,_=sess.run([model.accuracy, model.train_step], feed_dict={model.x: X_train[batch_ind], model.y_: y_train[batch_ind], model.keep_prob: args.drop_out})
        
        i += 1

    batch_ind_test = np.random.choice(Ntest, args.batch_size, replace=False)
    result = sess.run([model.accuracy, model.numel], feed_dict={model.x: X_test[batch_ind_test], model.y_: y_test[batch_ind_test], model.keep_prob: args.drop_out})
    print "accuracy result is {}, variabels num is {}".format(result[0], result[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MDN', type=bool, default=True,
                        help='whether to use MDN in model')
    parser.add_argument('--Num_layers', type=int, default=2,
                        help='numbers of layers for LSTM')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden cells in each layer')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--max_grad_norm', type=int,
                        default=1, help='bound of clipping gradient')
    parser.add_argument('--sl', type=int, default=12, help='sequence length')
    parser.add_argument('--mixtures', type=int,
                        default=3, help='number of MDNs')
    parser.add_argument('--lerning_rate', type=float,
                        default=0.05, help='learning rate')
    parser.add_argument('--drop_out', type=float,
                        default=0.95, help='drop out rate')
    parser.add_argument('--max_interactions', type=int, default=20000,
                        help='maximum number of training interactions')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='ratio of splitting training and testing data')
    parser.add_argument('--dist_2_baket', type=float,
                        default=5.0, help='distance from point to basket')
    parser.add_argument('--crd_num', type=int, default=4,
                        help='set the coordinate numbers, the default values is x, y, z and game_clock')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
