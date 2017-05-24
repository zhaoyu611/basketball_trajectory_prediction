import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def plot_traj_MDN_mult(model, sess, val_dict, batch, sl_plot=5, ind=-1):
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
  result = sess.run([model.mu1, model.mu2, model.mu3, model.s1,
                     model.s2, model.s3, model.rho, model.theta], feed_dict=val_dict)
  batch = batch.transpose(0, 2, 1)
  batch_size, crd, seq_len = batch.shape

  assert ind < batch_size, 'Your index is outside batch'
  assert sl_plot < seq_len, 'Your sequence index is outside sequence'
  if ind == -1:
    ind = np.random.randint(0, batch_size)

  fig = plt.figure()
  ax = fig.add_subplot(221, projection='3d')
  ax.plot(batch[ind, 0, :], batch[ind, 1, :], batch[ind, 2, :], 'r')

  point = (batch[ind, 0, sl_plot], batch[
           ind, 1, sl_plot], batch[ind, 2, sl_plot])
  point_next = (batch[ind, 0, sl_plot + 1], batch[ind,
                                                  1, sl_plot + 1], batch[ind, 2, sl_plot + 1])

  ax.scatter(point[0], point[1], point[2])
  ax.scatter(point_next[0], point_next[1], point_next[2], color='r')
  # print point
  # print point_next
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('y coordinate')
  ax.set_zlabel('z coordinate')

  delta = 0.025  # Grid size to evaluate the PDF
  width = 1.0  # how far to evaluate the pdf?
  # lower-case x1,x2,x3 are indezing the grid
  # upper-case X1,X2,X3 are coordinates in the mesh
  x1 = np.arange(-width, width + 0.1, delta)
  x2 = np.arange(-width, width + 0.2, delta)
  x3 = np.arange(-width, width + 0.3, delta)

  ZZ, offset = cal_next_point_pdf(result, ind, sl_plot, [x1, x2, x3])

  # Every Z is a marginalization of ZZ.
  # summing over axis 2, gives the pdf over x1,x2
  # summing over axis 1, gives the pdf over x1,x3
  # summing over axis 0, gives the pdf over x2,x3
  ax = fig.add_subplot(2, 2, 2)

  X1, X2 = np.meshgrid(x1, x2)
  Z = np.sum(ZZ, axis=2)
  CS = ax.contour(X1 + point[0], X2 + point[1], Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.scatter(point[0], point[1])
  ax.scatter(point_next[0], point_next[1], color='r')
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('y coordinate')

  ax = fig.add_subplot(2, 2, 3)
  X1, X3 = np.meshgrid(x1, x3)
  Z = np.sum(ZZ, axis=1)
  CS = ax.contour(X1 + point[0], X3 + point[2], Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.scatter(point[0], point[2])
  ax.scatter(point_next[0], point_next[2], color='r')
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('Z coordinate')

  ax = fig.add_subplot(2, 2, 4)
  X2, X3 = np.meshgrid(x2, x3)
  Z = np.sum(ZZ, axis=0)
  CS = ax.contour(X2 + point[1], X3 + point[2], Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.scatter(point[1], point[2])
  ax.scatter(point_next[1], point_next[2], color='r')
  ax.set_xlabel('y coordinate')
  ax.set_ylabel('Z coordinate')

  #-----start generate multiplty trajectories--------
  point_list = []
  trajectory_num = 2
  start_point = np.array([batch[ind, 0, sl_plot], batch[
      ind, 1, sl_plot], batch[ind, 2, sl_plot]])

  for sl in range(sl_plot, seq_len - 1):
    _, offset = cal_next_point_pdf(result, ind, sl, [x1, x2, x3])
    start_point = start_point + offset
    point_list.append(start_point)
  print point_list
  point_array = np.stack(point_list, axis=1)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(point_array[0, :], point_array[1, :], point_array[2, :], 'r')
  ax.plot(batch[ind, 0, :], batch[ind, 1, :], batch[ind, 2, :], 'b')


def cal_next_point_pdf(result, ind, sl_plot, crd_bound):
  """ Given a point and the params of probality distribution of next point, 
      then calculate the probality distribution of next point position.
      args:
        result [list] the session result, include [mu1, mu2, mu3, s1, s2, s3, rho, theta]
        ind: [int] the index of batch, in oder to specify trajectory
        sl_plot: [int] the time stamp point, it should be no bigger than seq_len
        crd_bound: [list] the lower case and upper case of x1, x2, x3
      return: 
        pdf: [ndarray] the pdf of next point, which has shape [x1, x2, x3]
        offset: [ndarray] the next point offset of current point, which has shape (3,)
  """

  [x1, x2, x3] = crd_bound
  X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
  XX = np.stack((X1, X2, X3), axis=3)
  PP = []
  point_offset_list = []
  mixtures = result[0].shape[1]
  for m in range(mixtures):
    mean = np.zeros((3))
    mean[0] = result[0][ind, m, sl_plot]
    mean[1] = result[1][ind, m, sl_plot]
    mean[2] = result[2][ind, m, sl_plot]
    cov = np.zeros((3, 3))
    sigma1 = result[3][ind, m, sl_plot]
    sigma2 = result[4][ind, m, sl_plot]
    sigma3 = result[5][ind, m, sl_plot]
    sigma12 = result[6][ind, m, sl_plot] * sigma1 * sigma2
    cov[0, 0] = np.square(sigma1)
    cov[1, 1] = np.square(sigma2)
    cov[2, 2] = np.square(sigma3)
    cov[1, 2] = sigma12
    cov[2, 1] = sigma12
    rv = multivariate_normal(mean, cov)
    point_offset = np.random.multivariate_normal(mean, cov)
    P = rv.pdf(XX)  # P is now in [x1,x2,x3]
    PP.append(P)
    point_offset_list.append(point_offset)
  # PP is now a list

  PP = np.stack(PP, axis=3)

  point_offset_list = np.stack(point_offset_list, axis=1)
  # PP is now in [x1,x2,x3,mixtures]
  # Multiply with the mixture
  theta_local = result[7][ind, :, sl_plot]
  pdf = np.dot(PP, theta_local)
  offset = np.dot(point_offset_list, theta_local)

  #pdf is now in [x1,x2,x3]
  print('The theta variables %s' % theta_local)

  def get_bound(val_range, x):
    min_val = min(val_range)
    max_val = max(val_range)
    if x < min_val:
      x = min_val
    if x > max_val:
      x = max_val
    return x

  offset[0] = get_bound(x1, offset[0])
  offset[1] = get_bound(x2, offset[1])
  offset[2] = get_bound(x3, offset[2])

  return pdf, offset


def draw_multi_traj(start_point, offset, trajectory_num=3):
  """generate multiply trajectories and draw them
    args:
      point: [list] the x, y, z value of this point
      pdf: [ndarray] the probility distribution of x, y, z coordinates next point,
                      shape is [x1, x2, x3]
      trajectory_num: [int] generate trajectory num for each point

    return:
  """

  # x_val, y_val, z_val = point  # the x y z value the this point

  # def cal_next_point(point, pdf):
  #   """ calculate next point when given current point and pdf\
  #       args:
  #         point: [list] the x, y, z value of this point
  #         pdf:   [ndarray] the probility distribution of next point,
  #                           whose shape is [x1, x2, x3]
  #       return:
  #         next_point: [list] the x, y, z value of next point
  #   """
  #   # for i in range(len(point)):
  #   #   N = pdf.shape[i]
  #   #   accumulate = 0
  #   #   for j in range(N):
  #   #     accumulate+ = pdf

  # cal_next_point(point, pdf)
  next_point = start_point + offset
  return next_point
