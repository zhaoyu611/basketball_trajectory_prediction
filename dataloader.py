import numpy as np
import pandas as pd
from itertools import groupby
import tarfile
import os
# get the longest elements and its index


def return_large_true(ind_crit):
    """ return the longest sequnce and its index in the same ID
    args: #centering the sequences
        ind_crit: [ndarray] each element is 0 or 1
    return:
        best_elements: [ndarray] the longest sequence 
        best_i: [int] the longest sequence's start index
    """
    i = 0
    best_elements = 0
    best_i = 0
    for key, group in groupby(ind_crit, lambda x: x):

        number = next(group)
        elements = len(list(group)) + 1
        if number == 1 and elements > 1:
            if elements > best_elements:  # if the current sequence is longer than previous, then update
                best_elements = elements
                best_i = i
        i += elements
    return best_elements, best_i


class DataLoad():
    def __init__(self, direc, csv_file, center=np.array([5.25, 25.0, 10.0])):
        """Create a dataload class to load data from local and preprocess with it
        args:
            dirc: [str] the path of input data file
            csv_file: [str] the input data file name, and the file extentions should be '.csv'
        """
        assert direc[-1] == '/', 'Please provide a dicrectionary ending with a /'
        assert csv_file[-3:] == 'csv', 'Please confirm the file extentions'
        self.center = center
        self.csv_loc = direc + csv_file  # The location of the csv file
        self.data3 = []  # create a list to store the preprocessed data
        self.labels = []  # create a list to store the preprocessed labels
        self.data = {}  # create a dict to store splitted data and labels
        self.N = 0  # total number of sequences in preprocessed data
        self.iter_train = 0  # train iteration
        self.epochs = 0  # epochs for looping
        self.omit = 0  # omitted sequences number

        if not os.path.exists(self.csv_loc):
            with tarfile.open(direc+'seq_all.csv.tar.gz') as tar:
                tar.extract(csv_file, path=direc)


    def munge_data(self, height=11.0, seq_len=10.0, dist=3.0, verbose=False):
        """read data, omit useless sequences, and reshape data 
        args: 
            height: [float] the low bound to chop of data
            seq_len: [float] cut too long sequences to seq_len, and discard the unsatisfied ones.
            dist: the minimum distance between the point and center, discard these points who are samller than it
            verbose: [bool] whether to show some headers and other outputs while debugging
        return:
            self.data3: [ndarray] the preprocessed input data, whose shape is [self.N, seq_len, 3]. 
                        and the last dimension is x, y, z
            self.labels: [ndarray] the preprocessed input labels, whose shape is [self.N, 1]
        """
        #========step 1: read data=============
        #-----judgement for initial configuration---------
        if self.data3:
            print "You already have dat in this instance. Are you calling the function twice?"
        if height < 9.0:
            print "Please note that the height is measured from ground."
        df = pd.read_csv(self.csv_loc).sort(
            ['id', 'game_clock'], ascending=[1, 0])

        if verbose == True:  # showing some useful info
            print "the shape of data is {}".format(df.shape)
            test_data = df[df['id'] == '0021500001_102']
            print test_data.head(10)
        # convert data type and extract useful attributes
        df_arr = df.as_matrix(
            ['x', 'y', 'z', 'game_clock', 'EVENTMSGTYPE', 'rankc'])
        #======================================
        #========step 2: extract useful data==========
        row, col = df_arr.shape
#         print type(df_arr)
        start_ind = 0  # set start index
        if verbose:
            row = 142
        for i in range(1, row, 1):  # for each point
            if int(df_arr[i, 5]) == 1:
                end_ind = i
                # pick the points in the same ID
                seq = df_arr[start_ind:end_ind, :4]
                dist_xyz = np.linalg.norm(seq[:, :3] - self.center, axis=1)
                # set a critera to judge the sequence whether satisfy that
                # the distance from point to center bigger than dist and
                # point's height bigger than height
                ind_crit = np.logical_and(dist_xyz > dist, seq[:, 2] > height)
                if sum(ind_crit) == 0:
                    continue  # if no sequence satisfy the condition, then continue
                li, i = return_large_true(ind_crit)
                seq = seq[i:li + i, :]  # extract longest sequence from each ID

                # assume start time is 0
                try:
                    seq[:, 3] = seq[:, 3] - np.min(seq[:, 3])
                except:
                    print "A sequence didn't match the criteria"
                if seq.shape[0] >= seq_len:

                    self.data3.append(seq[-seq_len:])
                    self.labels.append(df_arr[start_ind, 4])
                else:
                    self.omit += 1
                start_ind = end_ind

        #======================================
        #========step 3: reshape data==========
        try:
            self.data3 = np.stack(self.data3, 0)
            self.labels = np.stack(self.labels, 0)
            self.labels = self.labels - 1  # convert labels' value from 1, 2 to 0, 1
            self.N = len(self.labels)
        except:
            print "Something is wrong when convert list to ndarray"
        print "After preprocess, we lost {} sequences in sum".format(self.omit)

    def center_data(self, center_cent=np.array([5.25, 25.0, 10.0])):
        """centering all data with new center_cent"""
        assert not isinstance(
            self.data3, list), "Please convert type to np.ndarray"
        assert isinstance(
            self.center, np.ndarray), "Please provide center as ndarray"
        self.data3[:, :, :3] -= center_cent
        self.center -= center_cent

    def test_valid_data_split(self, ratio=0.8):
        """split test and vlid data"""
        per_ind = np.random.permutation(self.N)  # shuffle the index
        # set the top ratio indexes as test index
        train_ind = per_ind[:int(ratio * self.N)]
        # set the left indexes as test index
        test_ind = per_ind[int(ratio * self.N):]
        self.data['X_train'] = self.data3[train_ind]
        self.data['y_train'] = self.labels[train_ind]
        self.data['X_test'] = self.data3[test_ind]
        self.data['y_test'] = self.labels[test_ind]
        num_train = self.data['X_train'].shape[0]
        num_test = self.data['X_test'].shape[0]
        sum_num = num_test + num_train
        print "we have {0} samples in sum, including {1} traing samples, and {2} test samples".format(sum_num, num_train, num_test)
        print "type of X_train is {0}, shape of X_train is (num_sample, seq_len, crd): {1}".format(type(self.data['X_train']), self.data['X_train'].shape)
        print "type of y_train is {0}, shape of y_train is (num_sample, ): {1}".format(type(self.data['y_train']), self.data['y_train'].shape)
        
        return sum_num, num_train, num_test


if __name__ == '__main__':
    dl = DataLoad('./data/', 'seq_all.csv')
