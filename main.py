import numpy as np
import tensorflow as tf
from dataloader import DataLoad
import argparse
from model import Model
# from matplotlib import pyplot as plt
import sklearn
from sklearn import metrics


def load_arg():

    paser = argparse.ArgumentParser()
    paser.add_argument("--hidden_layers", type=int,
                       default=2, help="number of hidden layer ")
    paser.add_argument("--seq_len", type=int, default=12,
                       help="sequence length")
    paser.add_argument("--dist", type=float, default=5.0,
                       help="distance from point to center")
    paser.add_argument("--hidden_size", type=int, default=64,
                       help="units num in each hidden layer")
    paser.add_argument("--drop_out", type=float, default=0.7,
                       help="drop out probability")
    paser.add_argument('--learning_rate', type=float, default=0.005,
                       help="learning_rate")
    paser.add_argument('--epoch', type=int, default=300,
                       help="epoch")
    paser.add_argument('--batch_size', type=int, default=64,
                       help="batch size")
    paser.add_argument('--model_before_MDN', type=str, default='LSTM',
                       help='the model before MDN, the input should be "LSTM" or "BLSTM"')

    args = paser.parse_args()
    return args


def main():
    #=======step 1: get args for model=======
    args = load_arg()
    #=======step 2: preprocess data==========
    direc = './data/'  # directory of data file
    csv_file = 'seq_all.csv'
    dl = DataLoad(direc, csv_file)
    dl.munge_data(height=11.0, seq_len=args.seq_len, dist=args.dist)
    basket_center = np.array([5.25, 25.0, 10.0])
    dl.center_data(center_cent=basket_center)
    sum_samples, num_train, num_test = dl.test_valid_data_split(ratio=0.8)
    print "--------------------------------------------------------------------"
    X_train = dl.data['X_train']
    y_train = dl.data['y_train']
    X_test = dl.data['X_test']
    y_test = dl.data['y_test']
    #=======step 3: construct model==========
    model = Model(args)
    #Now we construct model, there are LSTM_model, bidir_LSTM_model, 
    #CNN_model, Conv_LSTM_model, LSTM_MDN_model can be chosen.
    model.CNN_model()
    model.Evaluating()
    #=======step 4: start training===========

    train_cost_list = []
    test_cost_list = []
    test_AUC_list = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(args.epoch):
            for batch_num in range(num_train / args.batch_size):
                perm_ind = np.random.choice(
                    num_train, args.batch_size, replace=False)
                feed_dict = {model.X: X_train[perm_ind], model.y: y_train[
                    perm_ind], model.drop_out: args.drop_out}
                fetch = [model.train_op, model.accuracy,
                         model.cost]
                _, train_acc, train_cost = sess.run(
                    fetch, feed_dict=feed_dict)
            #=======step 5: start testing============
            train_cost_list.append(train_cost)
            # print "at {} epoch, the train accuracy is {}, the train cost is
            # {}".format(i, train_acc, train_cost)
            perm_ind = np.random.choice(
                num_test, args.batch_size, replace=False)
            feed_dict = {model.X: X_test[perm_ind], model.y: y_test[
                perm_ind], model.drop_out: 1.0}
            fetch = [model.accuracy, model.cost, model.y_pred, model.numel]
            test_acc, test_cost, y_pred, numel = sess.run(
                fetch, feed_dict=feed_dict)
            test_AUC = sklearn.metrics.roc_auc_score(
                y_test[perm_ind], y_pred[:, 1])
            print "at {} epoch, the training cost is {}, the training accuracy is {}".format(i, train_cost, train_acc)
            print "at {} epoch, the test cost is {}, the test accuracy is {}".format(i, test_cost, test_acc)
            print "at {} epoch, the test AUC is {}".format(i, test_AUC)
            print "------------------------------------------------------"
            test_AUC_list.append(test_AUC)
            test_cost_list.append(test_cost)

        best_AUC = max(test_AUC_list)
        best_AUC_ind = test_AUC_list.index(best_AUC)
        print "========================================================"
        print "Finally, the best test AUC is {} at {} epoch,".format(best_AUC, best_AUC_ind)
        print "Finally, the model has {} parameters".format(numel)
        # wirte result in local
        with open('result.txt', 'a') as f:
            f.write("the best test AUC is {} at {} epoch, the model has {} \
                parameters".format(best_AUC, best_AUC_ind, numel))
    #========step 5: draw results===============
    plot = False
    if plot:
        plt.figure()
        plt.plot(train_cost_list, 'r', label='train_cost')
        plt.plot(test_cost_list, '--r', label='test_cost')
        plt.legend()
        plt.figure()
        plt.plot(test_AUC_list, label='test_AUC')
        plt.show()


if __name__ == "__main__":
    main()
