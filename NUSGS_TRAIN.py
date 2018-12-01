from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, GaussianNoise, Dropout, BatchNormalization
from keras import initializers
from keras.optimizers import SGD, Nadam, RMSprop
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing #
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
import keras.backend as kb
import keras.backend as K
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

#from netCDF4 import Dataset

def rae(y_true, y_pred):
  return tf.reduce_sum(tf.abs(y_pred-y_true)) / tf.reduce_sum(tf.abs(y_true))

def mae(y_true, y_pred):
  return tf.reduce_mean(tf.abs(y_pred-y_true))

def mape(y_true, y_pred):
  diff = tf.abs((y_true - y_pred) / (tf.abs(y_true)))
  return 100. * tf.reduce_mean(diff)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# reading the input data

SGSDATA = np.genfromtxt('NUSGS_TRAIN.dat')
print SGSDATA.shape

X_train0 = SGSDATA[:,0:9]
Y_train0 = -SGSDATA[:,9:10]

X_train, X_test, Y_train, Y_test = train_test_split(X_train0,Y_train0, test_size=0.01, random_state=42)
print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

XMEAN=np.max(X_train,axis=0) #np.mean(X_train,axis=0)
XSTDD=np.min(X_train,axis=0)

print XMEAN, XSTDD
np.savetxt('xmax.dat',XMEAN)
np.savetxt('xmin.dat',XSTDD)

for j in range(9):
 X_train[:,j]= (X_train[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j]) #(X_train[:,j]-XMEAN[j])/XSTDD[j]
 X_test[:,j]= (X_test[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j])  #(X_test[:,j]-XMEAN[j])/XSTDD[j]

#print XMEAN,XSTDD

YMEAN=np.max(Y_train,axis=0) #np.mean(Y_train,axis=0)
YSTDD=np.min(Y_train,axis=0) #np.std(Y_train,axis=0)

np.savetxt('ymax.dat',YMEAN)
np.savetxt('ymin.dat',YSTDD)

Y_train[:]=(Y_train[:]-YSTDD)/(YMEAN-YSTDD) #(Y_train[:]-YMEAN)/YSTDD

print YMEAN,YSTDD

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Dense(32, input_dim=9,       kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(32,                    kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(32,                    kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(16,                   kernel_initializer='uniform', activation = 'sigmoid'))
#model.add(Dense(128,                   kernel_initializer='uniform', activation = 'sigmoid'))
#model.add(Dense(32,                  kernel_initializer='uniform', activation = 'tanh'))
model.add(Dense(1,                     kernel_initializer='uniform'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='rmsprop')
history = model.fit(X_train,Y_train, epochs=100, batch_size=256)

score  = model.predict(X_test)

score[:]=score[:]*(YMEAN-YSTDD)+YSTDD #score[:]*YSTDD+YMEAN

print("Larger: %.2f (%.2f) MSE" % (score.mean(), score.std()))

W_Input_Hidden0 = model.layers[0].get_weights()[0]; print W_Input_Hidden0.shape
biases0  = model.layers[0].get_weights()[1]; print biases0.shape
W_Input_Hidden1 = model.layers[1].get_weights()[0]; print W_Input_Hidden1.shape
biases1  = model.layers[1].get_weights()[1]; print biases1.shape
W_Input_Hidden2 = model.layers[2].get_weights()[0]; print W_Input_Hidden2.shape
biases2  = model.layers[2].get_weights()[1]; print biases2.shape
W_Input_Hidden3 = model.layers[3].get_weights()[0]; print W_Input_Hidden3.shape
biases3  = model.layers[3].get_weights()[1]; print biases3.shape

np.save('SWHidden003.npy',W_Input_Hidden0)
np.save('SWbiases003.npy',biases0)
np.save('SWHidden013.npy',W_Input_Hidden1)
np.save('SWbiases013.npy',biases1)
np.save('SWHidden023.npy',W_Input_Hidden2)
np.save('SWbiases023.npy',biases2)
np.save('SWHidden033.npy',W_Input_Hidden3)
np.save('SWbiases033.npy',biases3)
np.save('SWX_test3.npy', X_test)
np.save('SWY_test3.npy', Y_test)
np.save('SWScore3.npy', score)

# list all data in history
#print(history.history.keys())
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()




