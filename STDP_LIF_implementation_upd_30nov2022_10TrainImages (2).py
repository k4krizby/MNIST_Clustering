#!/usr/bin/env python
# coding: utf-8

# ## STDP Implementation - 29 NOV 2022

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
# import cv2

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


# In[120]:


get_ipython().system('pip3 install python-mnist opencv-python')
# from mnist.loader import MNIST
# import os
# loader = MNIST(os.path.join(os.path.dirname(os.getcwd())))
import cv2

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

# loader returns a list of 768-element lists of pixel values in [0,255]
# and a corresponding array of single-byte labels in [0-9]

n_train = len(train_labels)
n_test = len(test_labels)

# TrainIm, TrainL = loader.load_training()
TrainIm = np.array(train_images) # convert to ndarray
TrainL = np.array(train_labels)
TrainIm = TrainIm / TrainIm.max() # scale to [0, 1] interval

# TestIm, TestL = loader.load_testing()
TestIm = np.array(test_images) # convert to ndarray
TestL = np.array(test_labels)
TestIm = TestIm / TestIm.max() # scale to [0, 1] interval

# Randomly select train and test samples
trainInd = np.random.choice(len(TrainIm), n_train, replace=False)
TrainIm = TrainIm[trainInd]
TrainLabels = TrainL[trainInd]

testInd = np.random.choice(len(TestIm), n_test, replace=False)
TestIm = TestIm[testInd]
TestLabels = TestL[testInd]


# In[121]:


train_images.shape


# In[5]:


# # get train and test MNIST data from keras
# (train_images, train_labels), (
#     test_images,
#     test_labels,
    
# ) = tf.keras.datasets.mnist.load_data()

# print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)


# In[122]:


## reducing the number of images to be processed by selecting a few from each label class
# _1,train_images, _2, train_labels = train_test_split(train_images, train_labels, test_size=0.01334, random_state=42)
# _3,test_images, _4, test_labels = train_test_split(test_images, test_labels, test_size=0.02, random_state=42)

_1,train_images, _2, train_labels = train_test_split(train_images, train_labels, test_size=0.002, random_state=42)
_3,test_images, _4, test_labels = train_test_split(test_images, test_labels, test_size=0.003, random_state=42)

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)


# In[113]:


# to check the frequency of the labels in each set 
pd.Series(train_labels).value_counts(), pd.Series(test_labels).value_counts()


# In[123]:


# get the length of the train and test sets
n_train = len(train_labels)
n_test = len(test_labels)

print('n_train: ',n_train,', n_test: ',n_test)

# TrainIm, TrainL = loader.load_training()
TrainIm = np.array(train_images) # convert to ndarray
TrainL = np.array(train_labels)
TrainIm = TrainIm / TrainIm.max() # scale to [0, 1] interval

# TestIm, TestL = loader.load_testing()
TestIm = np.array(test_images) # convert to ndarray
TestL = np.array(test_labels)
TestIm = TestIm / TestIm.max() # scale to [0, 1] interval

# Randomly select/shuffle train and test samples
trainInd = np.random.choice(len(TrainIm), n_train, replace=False)
TrainImg = TrainIm[trainInd]
TrainLabels = TrainL[trainInd]

testInd = np.random.choice(len(TestIm), n_test, replace=False)
TestImg = TestIm[testInd]
TestLabels = TestL[testInd]


# In[124]:


TestImg[1]


# In[105]:


# to generate a Poisson Train from Images after flattening (based on matlab code)

def make_spike_trains(freqs, n_steps):
    ''' Create an array of Poisson spike trains
        Parameters:
            freqs: Array of mean spiking frequencies.
            n_steps: Number of time steps
    '''
    r = np.random.rand(len(freqs), n_steps)
    spike_trains = np.where(r <= np.reshape(freqs, (len(freqs),1)), 1, 0)
    return spike_trains

def MNIST_to_Spikes(maxF, img, t_sim, dt):
    ''' Generate spike train array from MNIST image.
        Parameters:
            maxF: max frequency, corresponding to 1.0 pixel value
            img: MNIST image (784,)
            t_sim: duration of sample presentation (seconds)
            dt: simulation time step (seconds)
    '''
    n_steps = int(np.ceil(t_sim / dt)) #  sample presentation duration in sim steps
    freqs = img.flatten() * maxF * dt # scale [0,1] pixel values to [0,maxF] and flatten
    return make_spike_trains(freqs, n_steps)

def MNIST_image_set(trainImg, testImg, maxF, t_sim, dt):
    ''' Generate spike train array from MNIST image.
        Parameters:
            trainImg: the list of train images
            testImg: the list of test images
            maxF: max frequency, corresponding to 1.0 pixel value
            im: MNIST image (784,)
            t_sim: duration of sample presentation (seconds)
            dt: simulation time step (seconds)
    '''
    trainPoissonTrain = []
    testPoissonTrain = []
    for im in trainImg:
        trainPoissonTrain.append(MNIST_to_Spikes(maxF, im, t_sim, dt))
    for im in testImg:
        testPoissonTrain.append(MNIST_to_Spikes(maxF, im, t_sim, dt))
    return trainPoissonTrain, testPoissonTrain


# In[125]:


# final variable list

# variables taken from 
# https://github.com/Nu-AI/Research_Anurag_Daram/blob/master/BIC_2021/Assignment5/Assignment5_eRBP.py

# Learning rule parameters
Imin = -0.05 # Minimum current for learning
Imax = 0.05 # Maximum current for learning

# Neuron parameters
t_syn = 25 # Synaptic time-constant at output layer

t_mO = 25 # Neuron time constant in output layer



RO = t_mO/10 # Membrane resistance in output layer

VthO = 0.005 # Output neuron threshold
V_m = -0.065 # Membrane voltage
V_rest = 0 # Resting membrane potential
t_refr = 4 # Duration of refractory period

# Simulation parameters
tSim = 0.35 # Duration of simulation (seconds)
maxF = 150 # maximum frequency of the input spikes
maxFL = 500 # maximum frequency of the target label spikes
dt = 0.001 # Data is sampled in ms
nBins = int(np.ceil(tSim/dt)) #total no. of time steps

# Network architecture parameters
dim = 28 # dim by dim is the dimension of the input images
n_in = dim*dim # no. of input neurons
n_out = 10 # no. of output neurons # 10,50,100 need to be tested

w_scale = 1e-3 # Weight scale at output layer
w_out = np.random.normal(0, w_scale, (n_out, n_in)) # input->output weights

# STDP parameters for weight update
A_p = 0.01      # STDP change in weight parameter A_plus
A_n = -0.01     # STDP change in weight parameter A_minus
Xtar = 0.5      #for synaptic trace
tau = 1000       #Time contsant
k= 0.9          #k is a decay rat e, 0 < k < 1.
A= 0.01         # Magnitude of A is between 0.001 and 0.1

# intrinsic plasticity
C = 0.02 # 20mV

# MNIST Parameter
minE = 1 # minimum #of Epochs
maxE = 5  # Max #no. of epochs
train_acc = np.zeros((maxE))
test_acc = np.zeros_like(train_acc)

# synaptic scsaling params
Rm = 1e8
Cm = 1e-7
Tm = Rm * Cm
V_m = -0.065
V_rest = -0.055
V_th = -0.055
A_p = 0.01
A_n = -0.01
dt = 0.001
T = np.arange(0, 0.35, 0.001)
tau = 100
bins = 350
A = 0.01
Xtar = 0.5
del_t = np.arange(-5, 5, 0.001)
V_vec = np.zeros((n_out,len(T)))
S_vec = np.zeros_like(V_vec)
# Xpre = np.zeros((n_out, len(T)))
# Xpre[:, 0] = 1
Iinj = np.zeros_like(V_vec)
spike_train = np.zeros_like(Xpre)
# weights = np.repeat(0.04, 784)
dw = np.zeros((n_in,len(T)))
weight_plot = np.zeros_like(dw)
V_vec[0] = V_m
post_spike = 0
pre_spike = 0
mu = 0.1
w_max = 0.04



post_status = 0
sum_weight = 0
pre_index = np.zeros(len(T)) # vector size 350 
post_index = np.zeros_like(pre_index) # vector size 350 


# accuracy params
train_acc = np.zeros((maxE))
test_acc = np.zeros_like(train_acc)


# In[107]:


train_acc.shape, V_vec.shape, spike_train.shape, weight_plot.shape, weights.shape, w_out.shape, Iinj.shape, Xpre.shape, dw.shape


# In[61]:


for im in TrainData:
    print(im.shape)
    break


# In[126]:


# call the poisson train for the test and train images
TrainData, TestData = MNIST_image_set(TrainImg, TestImg, maxF, t_sim, dt)
print('TrainData shape',len(TrainData), ', each element: ',TrainData[0].shape)
print('TestData shape',len(TestData), ', each element: ',TestData[0].shape)


# In[109]:


for e in range(maxE): # for each epoch
    correct_predictions = 0
    test_predictions = 0
    
    for u in range(n_train): # for each training pattern
        # Generate poisson data and labels
        # spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u], tSim, dt_conv)
        spikeMat =  TrainData[u] # get each Train image Poisson Train, size: 784 X 350
        fr_label = np.zeros(n_out) # 10x1 size
        fr_label[TrainLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt, nBins) # target spikes based on spikes 10x350 size spike train


        # Initialize firing time variables
        ts_O = np.full((n_out,len(T)), -t_refr) # for refractive period , size 10 X 350

        train_counter = np.zeros((n_out,1))

            
        for i in range(len(T) - 1):
            # Forward pass
            
            # synaptic scaling
#             if i > 1:
#                 w_out /= np.sum(w_out) * 0.1

            # LIF neuron Implementation
    
            # Iinj[:,i]= Iinj[:,i] + (dt/t_syn)* (w_out.dot(spikeMat[:, i]) -Iinj[:,i])
            Iinj[:,i] = w_out.dot(spikeMat[:, i]) * 1.75e-7

            V_vec[:,(i + 1)] = V_vec[:,i] + (dt / Tm) * (V_rest - V_vec[:,i]) + Rm * Iinj[:,i]
            
            
            if (V_vec[:,(i +1)] > VthO).any():
                V_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = V_rest
                S_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = 1
                post_spike = i*dt
            if (V_vec[:,i + 1] <= V_rest).any():
                V_vec[(V_vec[:,i + 1] <= V_rest),i + 1] = V_rest

            # V_vec[V_vec < -VthO/10] = -VthO/10 # Limit negative potential

            # If neuron in refractory period, prevent changes to membrane potential
            refr1 = (i*dt - ts_O[:,i] <= t_refr)
            V_vec[refr1,i+1] = 0
            
            
            # to implement accruacy measure
            fired = np.nonzero(S_vec[:,(i +1)] > 0) # output neurons that spiked
            ts_O[fired,i+1] = dt*i # Update their most recent spike times

            ST_O = np.zeros((n_out,1)) # Output layer spiking activity
            ST_O[fired] = 1 # Set neurons that spiked to 1

            train_counter = train_counter + ST_O
            

            # STDP Implementation for Weight update:
            # Updating the weights
            for j in range(n_in):
                if spikeMat[j][i] == 1:
                    pre_spike = i*dt
                    post_status = 1
                pre_index[i + 1] = pre_spike
                post_index[i + 1] = post_spike
                if post_spike - pre_spike < 0 and post_status == 1:
                    dw[j][i + 1] = A_p * np.exp((post_spike - pre_spike) / tau)
                    w_out[:,j] += dw[j][i + 1]
                    post_status = 0
                elif post_spike - pre_spike >= 0 and post_status == 1:
                    dw[j][i + 1] = A_n * np.exp((pre_spike - post_spike) / tau)
                    w_out[:,j] += dw[j][i + 1]
                    post_status = 0
                else:
                    dw[j][i + 1] = 0




    # Check train and test accuracy here.
    # If the output neuron with highest firing rate matches the target
    # neuron, and that rate is > 0, then the sample was classified correctly
        tn = TrainLabels[u]
        wn = np.argmax(train_counter)
        # print (train_counter, wn, tn)
        if tn==wn:
            correct_predictions += 1

    print (f'Accuracy in epoch {e} is {(correct_predictions/n_train)} ,correct train predictions: {correct_predictions}')
    train_acc[e] = (correct_predictions/n_train)*100
                           
    # print Training weights
    print(w_out[6,:].reshape((28,28)))


                
                           
    ## Perform Testing
    for u in range(n_test): # for each training pattern
        # Generate poisson data and labels
        # spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u], tSim, dt_conv)
        spikeMat =  TestData[u] # get each Train image Poisson Train, size: 784 X 350
        fr_label = np.zeros(n_out) # 10x1 size
        fr_label[TestLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt, nBins) # target spikes based on spikes 10x350 size spike train


        # Initialize firing time variables
#         ts_O = np.full((n_out,len(T)), -t_refr) # for refractive period , size 10 X 350

        test_counter = np.zeros((n_out,1))

            
        for i in range(len(T) - 1):


            # LIF neuron Implementation
    
            # Iinj[:,i]= Iinj[:,i] + (dt/t_syn)* (w_out.dot(spikeMat[:, i]) -Iinj[:,i])
            Iinj[:,i] = w_out.dot(spikeMat[:, i]) * 1.75e-7
            V_vec[:,(i + 1)] = V_vec[:,i] + (dt / Tm) * (V_rest - V_vec[:,i]) + Rm * Iinj[:,i]
            
            
            if (V_vec[:,(i +1)] > VthO).any():
                V_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = V_rest
                S_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = 1
                post_spike = i
            if (V_vec[:,i + 1] <= V_rest).any():
                V_vec[(V_vec[:,i + 1] <= V_rest),i + 1] = V_rest

            
            
            # to implement accruacy measure
            fired = np.nonzero(S_vec[:,(i +1)] > 0) # output neurons that spiked

            ST_O = np.zeros((n_out,1)) # Output layer spiking activity
            ST_O[fired] = 1 # Set neurons that spiked to 1

            test_counter = test_counter + ST_O
      


        tn = TestLabels[u]
        wn = np.argmax(test_counter)
        if tn==wn:
            test_predictions += 1

    print (f' Test Accuracy in epoch {e} is {(test_predictions/n_test)} ,correct test predictions: {test_predictions}')
    test_acc[e] = 100*(test_predictions/n_test)

# Generating plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.plot(train_acc, color='blue', label="Train Accuracy")
plt.plot(test_acc, color='red', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Model Performance")
plt.title("Performance on MNIST Unsupervised Classification")
plt.legend()
plt.show()


## Saving the weights of the network
# np.savetxt('input_weights.txt', w_in)
# np.savetxt('hidden_weights.txt', w_out)



# In[127]:


# for 800 Train and 200 test Images
tau = 0.01 # 10 milliseconds

for e in range(maxE): # for each epoch
    correct_predictions = 0
    test_predictions = 0
    
    for u in range(n_train): # for each training pattern
        # Generate poisson data and labels
        # spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u], tSim, dt_conv)
        spikeMat =  TrainData[u] # get each Train image Poisson Train, size: 784 X 350
        fr_label = np.zeros(n_out) # 10x1 size
        fr_label[TrainLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt, nBins) # target spikes based on spikes 10x350 size spike train


        # Initialize firing time variables
        ts_O = np.full((n_out,len(T)), -t_refr) # for refractive period , size 10 X 350

        train_counter = np.zeros((n_out,1))

            
        for i in range(len(T) - 1):
            # Forward pass
            
            # synaptic scaling
#             if i > 1:
#                 w_out /= np.sum(w_out) * 0.1

            # LIF neuron Implementation
    
            # Iinj[:,i]= Iinj[:,i] + (dt/t_syn)* (w_out.dot(spikeMat[:, i]) -Iinj[:,i])
            Iinj[:,i] = w_out.dot(spikeMat[:, i]) * 1.75e-7

            V_vec[:,(i + 1)] = V_vec[:,i] + (dt / Tm) * (V_rest - V_vec[:,i]) + Rm * Iinj[:,i]
            
            
            if (V_vec[:,(i +1)] > VthO).any():
                V_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = V_rest
                S_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = 1
                post_spike = i*dt
            if (V_vec[:,i + 1] <= V_rest).any():
                V_vec[(V_vec[:,i + 1] <= V_rest),i + 1] = V_rest

            # V_vec[V_vec < -VthO/10] = -VthO/10 # Limit negative potential

            # If neuron in refractory period, prevent changes to membrane potential
            refr1 = (i*dt - ts_O[:,i] <= t_refr)
            V_vec[refr1,i+1] = 0
            
            
            # to implement accruacy measure
            fired = np.nonzero(S_vec[:,(i +1)] > 0) # output neurons that spiked
            ts_O[fired,i+1] = dt*i # Update their most recent spike times

            ST_O = np.zeros((n_out,1)) # Output layer spiking activity
            ST_O[fired] = 1 # Set neurons that spiked to 1

            train_counter = train_counter + ST_O
            

            # STDP Implementation for Weight update:
            # Updating the weights
            for j in range(n_in):
                if spikeMat[j][i] == 1:
                    pre_spike = i*dt
                    post_status = 1
                pre_index[i + 1] = pre_spike
                post_index[i + 1] = post_spike
                if post_spike - pre_spike < 0 and post_status == 1:
                    dw[j][i + 1] = A_p * np.exp((post_spike - pre_spike) / tau)
                    w_out[:,j] += dw[j][i + 1]
                    post_status = 0
                elif post_spike - pre_spike >= 0 and post_status == 1:
                    dw[j][i + 1] = A_n * np.exp((pre_spike - post_spike) / tau)
                    w_out[:,j] += dw[j][i + 1]
                    post_status = 0
                else:
                    dw[j][i + 1] = 0




    # Check train and test accuracy here.
    # If the output neuron with highest firing rate matches the target
    # neuron, and that rate is > 0, then the sample was classified correctly
        tn = TrainLabels[u]
        wn = np.argmax(train_counter)
        # print (train_counter, wn, tn)
        if tn==wn:
            correct_predictions += 1

    print (f'Accuracy in epoch {e} is {(correct_predictions/n_train)} ,correct train predictions: {correct_predictions}')
    train_acc[e] = (correct_predictions/n_train)*100
                           
    # print Training weights
    print(w_out[6,:].reshape((28,28)))


                
                           
    ## Perform Testing
    for u in range(n_test): # for each training pattern
        # Generate poisson data and labels
        # spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u], tSim, dt_conv)
        spikeMat =  TestData[u] # get each Train image Poisson Train, size: 784 X 350
        fr_label = np.zeros(n_out) # 10x1 size
        fr_label[TestLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt, nBins) # target spikes based on spikes 10x350 size spike train


        # Initialize firing time variables
#         ts_O = np.full((n_out,len(T)), -t_refr) # for refractive period , size 10 X 350

        test_counter = np.zeros((n_out,1))

            
        for i in range(len(T) - 1):


            # LIF neuron Implementation
    
            # Iinj[:,i]= Iinj[:,i] + (dt/t_syn)* (w_out.dot(spikeMat[:, i]) -Iinj[:,i])
            Iinj[:,i] = w_out.dot(spikeMat[:, i]) * 1.75e-7
            V_vec[:,(i + 1)] = V_vec[:,i] + (dt / Tm) * (V_rest - V_vec[:,i]) + Rm * Iinj[:,i]
            
            
            if (V_vec[:,(i +1)] > VthO).any():
                V_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = V_rest
                S_vec[(V_vec[:,(i +1)] > VthO),(i + 1)] = 1
                post_spike = i
            if (V_vec[:,i + 1] <= V_rest).any():
                V_vec[(V_vec[:,i + 1] <= V_rest),i + 1] = V_rest

            
            
            # to implement accruacy measure
            fired = np.nonzero(S_vec[:,(i +1)] > 0) # output neurons that spiked

            ST_O = np.zeros((n_out,1)) # Output layer spiking activity
            ST_O[fired] = 1 # Set neurons that spiked to 1

            test_counter = test_counter + ST_O
      


        tn = TestLabels[u]
        wn = np.argmax(test_counter)
        if tn==wn:
            test_predictions += 1

    print (f' Test Accuracy in epoch {e} is {(test_predictions/n_test)} ,correct test predictions: {test_predictions}')
    test_acc[e] = 100*(test_predictions/n_test)

# Generating plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.plot(train_acc, color='blue', label="Train Accuracy")
plt.plot(test_acc, color='red', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Model Performance")
plt.title("Performance on MNIST Unsupervised Classification")
plt.legend()
plt.show()


## Saving the weights of the network
# np.savetxt('input_weights.txt', w_in)
# np.savetxt('hidden_weights.txt', w_out)



# In[118]:


pd.Series(TrainLabels).value_counts()


# In[64]:


w_out.shape, dw.shape


# In[77]:


i =0
V_vec[:,(i +1)] > VthO


# In[90]:


for epoch in list(range(minEpoch,maxEpoch+1)):
    if epoch>1:
        dW=1e-5
    print(epoch,dW)
    x =0
    for trainSamp in range(n_train):
        print("training sample number: ", trainSamp)
        uin = TrainData[trainSamp]
        SpikeT = np.zeros((numOutputNodes2,uin.shape[1]))
        Xpre = np.ones((784,1))*5
        vH = np.zeros((numOutputNodes2,1))
        rH = np.zeros((numOutputNodes2,1))
        print(uin.shape,uin[:,1].shape, SpikeT.shape, Xpre.shape, vH.shape, rH.shape)
        for time in range(uin.shape[1]):
            Xpre = Xpre + np.resize(uin[:,time],(784,1)) - (Xpre/Tpre)
            I = np.matmul(W2S1,np.resize(uin[:,time],(784,1)))*IS
            VthOutT = VthOut + th
#             [xH, vH, rH] = LIFGA(I,vH,rH, R, VthOutT, In, 1)
            print('    ',Xpre.shape,I.shape,VthOutT.shape)
            break
        break


# In[56]:


list(range(1,5))


# ## Appendix - Old Code

# In[80]:


# def gen_Poisson_spikes(img_array, fr, bins, spike_train):
#     img_array = img_array.flatten()
#     dt = 0.001
#     for pixel in range(img_array.size):
#         if img_array[pixel] != 0:
#             fr2 = fr * img_array[pixel]
#             poisson_output = np.random.rand(1, bins) < fr2 * dt
#             spike_train[pixel] = poisson_output.astype(int)
#     return spike_train


# In[14]:


# #LIF neuron parameters

# Rm = 1e8
# Cm = 1e-7
# Tm = Rm * Cm
# V_m = -0.065
# V_rest = -0.055
# V_th = -0.055
# A_p = 0.01
# A_n = -0.01
# dt = 0.001
# T = np.arange(0, 5, 0.001)
# tau = 1000
# bins = 350
# A = 0.01
# Xtar = 0.5
# del_t = np.arange(-5, 5, 0.001)
# V_vec = np.zeros((len(T)))
# S_vec = np.zeros_like(V_vec)
# Xpre = np.zeros((784, len(T)))
# Xpre[:, 0] = 1
# Iinj = np.zeros_like(V_vec)
# spike_train = np.zeros_like(Xpre)
# weights = np.repeat(0.04, 784)
# dw = np.zeros_like(Xpre)
# weight_plot = np.zeros_like(dw)
# V_vec[0] = V_m
# post_spike = 0
# pre_spike = 0
# mu = 0.1
# w_max = 0.04
# maxF=150



# In[ ]:




