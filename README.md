# basketball_trajectory_prediction
This repo is an open source of paper : Applying bidirectional LSTM and Mixture Density Network for Basketball Trajectory Prediction.
I strongly recommend you to review Rajiv and Rob's repo at first.  the URL is https://github.com/RobRomijnders/RNN_basketball. 
I think they made cool job and details about basketball prediction. Also you can find their paper and referrences in the repo.
Based on their contribution, I set up a new repo, which proposed Bidirectional LSTM and Mixture Density Network (BLSTM-MDN) for the same prediction problem.
I did 2 jobs in the main, Hit or miss classification and trajecotry generating.
In the first job, users can choose one of models, including CNN, LSTM, BLSTM, LSTM-MDN and BLSTM-MDN. And trajectory genarating only works for LSTM-MDN and BLSTM-MDN.

# Setup
* TesnsorFlow 1.0 <br>
* sklearn <br>
* hyperopt <br>

# The files
* data: the original data is in 'seq_all.csv.tar.gz', and the 'seq_all.csv' is the unziped dataset.
* plot_staff: the scripts and final figures based on the models
* dataloader.py: data pre-process
* model.py: build model by TensorFlow
* util_MDN: utility functions for building model
* sample.py: functions used for generating trajectory
* main.py: main steps for classification and generating

# Run
Simply run file "main.py" in terminal with default argpases: python main.py
Here is the explanation of each argpase.
~~~python

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
  paser.add_argument('--epoch', type=int, default=1,
                     help="epoch")
  paser.add_argument('--batch_size', type=int, default=64,
                     help="batch size")
  paser.add_argument('--model_type', type=str, default='BLSTM_MDN_model',
                     help='the model type should be LSTM_model, \
                       bidir_LSTM_model, CNN_model, Conv_LSTM_model, \
                       LSTM_MDN_model or BLSTM_MDN_model.')
~~~
                       
If you want to generate some trajetories, please set "generate_trajectory" as True in code. Because it is False in default.
It should be noted that it only generates traejctory with BLSTM-MDN or LSTM-MDN.

# Contact me
Be free the ust the code for studying. But please contact me if you want for commercial applying. <br>
You are welcome to pull requests or issues. <br>
E-mail: zhaoyuafeu@gmail.com <br>
Facebook: zhaoyuafeu <br>




