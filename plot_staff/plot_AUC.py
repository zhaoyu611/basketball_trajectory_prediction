import cPickle
import matplotlib.pyplot as plt
import numpy as np

file_list = ['CNN_AUC.pkl', 'BLSTM_AUC.pkl',
             'LSTM_MDN_AUC.pkl', 'BLSTM_MDN_AUC.pkl']

CNN_AUC = cPickle.load(open("CNN_AUC.pkl", "rb"))
LSTM_MDN_AUC = cPickle.load(open("LSTM_MDN_AUC.pkl", "rb"))
BLSTM_AUC = cPickle.load(open("BLSTM_AUC.pkl", "rb"))
BLSTM_MDN_AUC = cPickle.load(open("BLSTM_MDN_AUC.pkl", "rb"))
print np.max(CNN_AUC)
print np.max(LSTM_MDN_AUC)
print np.max(BLSTM_AUC)
print np.max(BLSTM_MDN_AUC)
print np.argmax(CNN_AUC)
print np.argmax(LSTM_MDN_AUC)
print np.argmax(BLSTM_AUC)
print np.argmax(BLSTM_MDN_AUC)

plt.figure()
plt.plot(CNN_AUC, 'r', linewidth=2, label='CNN')
plt.plot(BLSTM_AUC, 'b', linewidth=2, label='BLSTM')
plt.plot(LSTM_MDN_AUC, 'g', linewidth=2, label='LSTM_MDN')
plt.plot(BLSTM_MDN_AUC, 'y', linewidth=2, label='BLSTM_MDN')
# plt.title("AUC for each model at 5 feet to basket",fontsize=15)
plt.ylim((0.5,1.0))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("epoch", fontsize=15)
plt.ylabel("AUC", fontsize=15)
plt.legend()
plt.grid()
plt.show()

