import numpy as np
import tensorflow as tf
from dataloader import DataLoad
import argparse
from model import Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn import metrics
from scipy.stats import multivariate_normal


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
  paser.add_argument('--epoch', type=int, default=3,
                     help="epoch")
  paser.add_argument('--batch_size', type=int, default=64,
                     help="batch size")
  paser.add_argument('--model_type', type=str, default='BLSTM_MDN_model',
                     help='the model type should be LSTM_model, \
                       bidir_LSTM_model, CNN_model, Conv_LSTM_model, \
                       LSTM_MDN_model or BSLTM_MDN_model.')

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
  if args.model_type == 'LSTM_model':
    model.LSTM_model()
  elif args.model_type == 'bidir_LSTM_model':
    model.bidir_LSTM_model()
  elif args.model_type == 'CNN_model':
    model.CNN_model()
  elif args.model_type == 'Conv_LSTM_model':
    model.Conv_LSTM_model()
  elif args.model_type == 'LSTM_MDN_model':
    model.MDN_model('LSTM')
  elif args.model_type == 'BLSTM_MDN_model':
    model.MDN_model('BLSTM')
  else:
    print "please choose correct model type"
    return
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
    plot = True
    if plot:
      if args.model_type == 'LSTM_MDN_model' or 'BLSTM_MDN_model':

        val_dict = {model.X: X_test[perm_ind], model.y: y_test[
            perm_ind], model.drop_out: 1.0}
        batch = X_test[perm_ind]



        def plot_traj_MDN_mult(model,sess,val_dict,batch,sl_plot = 5, ind = -1):
          """Plots the trajectory. At given time-stamp, it plots the probability distributions
          of where the next point will be
          THIS IS FOR MULTIPLE MIXTURES
          input:
          - sess: the TF session
          - val_dict: a dictionary with which to evaluate the model
          - batch: the batch X_val[some_indices] that you feed into val_dict.
            we could also pick this from val-dict, but this workflow is cleaner
          - sl_plot: the time-stamp where you'd like to visualize
          - ind: some index into the batch. if -1, we'll pick a random one"""
          result = sess.run([model.mu1,model.mu2,model.mu3,model.s1,model.s2,model.s3,model.rho,model.theta],feed_dict=val_dict)
          batch = batch.transpose(0, 2, 1)
          batch_size,crd,seq_len = batch.shape

          assert ind < batch_size, 'Your index is outside batch'
          assert sl_plot < seq_len, 'Your sequence index is outside sequence'
          if ind == -1: ind = np.random.randint(0,batch_size)
          delta = 0.025  #Grid size to evaluate the PDF
          width = 1.0  # how far to evaluate the pdf?

          fig = plt.figure()
          ax = fig.add_subplot(221, projection='3d')
          ax.plot(batch[ind,0,:], batch[ind,1,:], batch[ind,2,:],'r')
          ax.scatter(batch[ind,0,sl_plot], batch[ind,1,sl_plot], batch[ind,2,sl_plot])
          ax.set_xlabel('x coordinate')
          ax.set_ylabel('y coordinate')
          ax.set_zlabel('z coordinate')


          # lower-case x1,x2,x3 are indezing the grid
          # upper-case X1,X2,X3 are coordinates in the mesh
          x1 = np.arange(-width, width+0.1, delta)
          x2 = np.arange(-width, width+0.2, delta)
          x3 = np.arange(-width, width+0.3, delta)
          X1,X2,X3 = np.meshgrid(x1,x2,x3,indexing='ij')
          XX = np.stack((X1,X2,X3),axis=3)

          PP = []

          mixtures = result[0].shape[1]
          for m in range(mixtures):
            mean = np.zeros((3))
            mean[0] = result[0][ind,m,sl_plot]
            mean[1] = result[1][ind,m,sl_plot]
            mean[2] = result[2][ind,m,sl_plot]
            cov = np.zeros((3,3))
            sigma1 = result[3][ind,m,sl_plot]
            sigma2 = result[4][ind,m,sl_plot]
            sigma3 = result[5][ind,m,sl_plot]
            sigma12 = result[6][ind,m,sl_plot]*sigma1*sigma2
            cov[0,0] = np.square(sigma1)
            cov[1,1] = np.square(sigma2)
            cov[2,2] = np.square(sigma3)
            cov[1,2] = sigma12
            cov[2,1] = sigma12
            rv = multivariate_normal(mean,cov)
            P = rv.pdf(XX)  #P is now in [x1,x2,x3]
            PP.append(P)
          # PP is now a list
          PP = np.stack(PP,axis=3)
          # PP is now in [x1,x2,x3,mixtures]
          #Multiply with the mixture
          theta_local = result[7][ind,:,sl_plot]
          ZZ = np.dot(PP,theta_local)
          #ZZ is now in [x1,x2,x3]

          print('The theta variables %s'%theta_local)


          #Every Z is a marginalization of ZZ.
          # summing over axis 2, gives the pdf over x1,x2
          # summing over axis 1, gives the pdf over x1,x3
          # summing over axis 0, gives the pdf over x2,x3
          ax = fig.add_subplot(2,2,2)
          X1, X2 = np.meshgrid(x1, x2)
          Z = np.sum(ZZ,axis=2)
          CS = ax.contour(X1, X2, Z.T)
          plt.clabel(CS, inline=1, fontsize=10)
          ax.set_xlabel('x coordinate')
          ax.set_ylabel('y coordinate')

          ax = fig.add_subplot(2,2,3)
          X1, X3 = np.meshgrid(x1, x3)
          Z = np.sum(ZZ,axis=1)
          CS = ax.contour(X1, X3, Z.T)
          plt.clabel(CS, inline=1, fontsize=10)
          ax.set_xlabel('x coordinate')
          ax.set_ylabel('Z coordinate')

          ax = fig.add_subplot(2,2,4)
          X2, X3 = np.meshgrid(x2, x3)
          Z = np.sum(ZZ,axis=0)
          CS = ax.contour(X2, X3, Z.T)
          plt.clabel(CS, inline=1, fontsize=10)
          ax.set_xlabel('y coordinate')
          ax.set_ylabel('Z coordinate')




        plot_traj_MDN_mult(model,sess,val_dict,batch)

      plt.figure()
      plt.plot(train_cost_list, 'r', label='train_cost')
      plt.plot(test_cost_list, '--r', label='test_cost')
      plt.legend()
      plt.figure()
      plt.plot(test_AUC_list, label='test_AUC')
      plt.show()


if __name__ == "__main__":
  main()
