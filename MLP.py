import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import struct as st 
import os 
import urllib
from urllib.request import urlretrieve
import gzip
import lasagne
import theano
import theano.tensor as T

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading ",filename) 
    urllib.request.urlretrieve(source+filename,filename)

def load_mnist_images(filename):
    if not os.path.exists(filename): 
        download(filename)
        print("There is no such an item.")
    with gzip.open(filename,'rb') as f:             
        data=np.frombuffer(f.read(), np.uint8, offset=16)
        print("Filename: ", filename, "Shape: ", data.shape)

    data = data.reshape(-1,1,28,28)
    
    print("Image Shape: ", data.shape)
    return data/np.float32(256)
def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=8)
    
    print ("Label shape :" ,data.shape)
    return data

def load_dataset():
        
    X_train= load_mnist_images('train-images-idx3-ubyte.gz')
    y_train= load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test= load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test= load_mnist_labels('t10k-labels-idx1-ubyte.gz')
        
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset() 

def build_NN(input_var=None): 
    l_in =lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    l_in_drop =lasagne.layers.DropoutLayer(l_in,p=0.2)
    l_hidl= lasagne.layers.DenseLayer(l_in,num_units=100 ,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    l_hidl_drop=lasagne.layers.DropoutLayer(l_hidl,p=0.2)
    l_hid2= lasagne.layers.DenseLayer(l_hidl,num_units=50 ,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    l_hid2_drop=lasagne.layers.DropoutLayer(l_hid2,p=0.2)
    l_out = lasagne.layers.DenseLayer(l_hid2,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
   # l_out = lasagne.layers.DenseLayer(l_hid2,num_units=10,nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_out

input_var = T.tensor4('input')
target_var=T.ivector('target')

network=build_NN(input_var) 

prediction =lasagne.layers.get_output(network)
loss=lasagne.objectives.categorical_crossentropy(prediction,target_var)
loss = loss.mean() 

params = lasagne.layers.get_all_params(network,trainable=True)
updates=lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01,momentum=0.9)

train_fn=theano.function([input_var,target_var],loss,updates=updates)

num_training_steps=61
train_err=0
train_batches=0

for step in range(num_training_steps):
    train_err +=train_fn(X_train,y_train)
    train_batches += 1
    if(step%5==0):   
        print("Training Loss for " +str(step+1)+"th epoch : " ,train_err/train_batches)

test_prediction = lasagne.layers.get_output(network)
val_fn=theano.function([input_var],test_prediction)


print(val_fn([X_test[2]]))
plt.imshow(X_test[2][0])
plt.show()
print("Actual number: ",y_test[2])

test_prediction=lasagne.layers.get_output(network,deterministic=True) 
test_acc=T.mean(T.eq(T.argmax(test_prediction,axis=1),target_var),dtype=theano.config.floatX)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

acc_fn=theano.function([input_var,target_var],test_acc)
print("Accuracy: ",acc_fn(X_test,y_test))
