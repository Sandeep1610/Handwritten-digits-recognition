#!/usr/bin/env python
# coding: utf-8

# # HANDWRITTEN  DIGIT  RECOGNITION

# In[1]:


import keras              # The Keras library already contains some datasets and MNIST is one of them. So we can easily import the dataset             
import numpy as np
import pandas as pd


# In[2]:


from keras.datasets import mnist                                  # Modified National Institute of Standards and Technology database
(x_train, y_train), (x_test, y_test) = mnist.load_data()          # The mnist.load_data() method returns us the training data, its labels and also the testing data and its labels.


# In[3]:


np.shape(x_train)     # we having total 60000 traing examples to train our model
                      # each example having pixel matrix of shape 28 x 28 as an input and its corresponding label as output


# In[4]:


np.shape(x_test)    # we having total 10000 data points to evalute our model


# In[5]:


x_train[0]         # this is our first training example input ( pixel matrix 28 x 28 )


# In[6]:


x_train.max()  # every element in pixel matrix is in range of 0 to 255
                # Typically zero is taken to be black, and 255 is taken to be white.


# In[7]:


import matplotlib.pyplot as plt                  


# In[8]:


plt.imshow(x_train[1])              # The imshow() function in pyplot module of matplotlib library is used to display data as an image of its pixel matrix; i.e. on a 2D regular raster.                 


# In[9]:


# reshape() function is used to give a new shape to an array without changing its data.
x_train_flatten=x_train.reshape(len(x_train),28*28)      # reshaing the input pixel matrix 28 x 28 into one dimensional vector of length 784 
x_test_flatten=x_test.reshape(len(x_test),28*28)         # we can call it as Unrolling 2D matrix to 1D vector (for each input data point)


# In[10]:


np.shape(x_train_flatten)           # initially we having 60000 matrix's as input data for training but we reshaped into 60000 1D vectors of length 784


# In[11]:


np.shape(x_test_flatten)  # initially we having 10000 matrix's as input data to evalute but we reshaped into 10000 1D vectors of length 784


# In[12]:


x_train_flatten=x_train_flatten/255     # Normalizing by dividing every element in matrix with 255(The pixel values vary from 0 to 255.). Normalization typically means rescales the values into a range of [0,1].
x_test_flatten=x_test_flatten/255          


# # logistic regression Model

# In[13]:


from sklearn.linear_model import LogisticRegression      # Import Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(x_train_flatten,y_train)                     # train the model using the training sets


# In[14]:


y_pred_log_reg = log_reg.predict(x_test_flatten)  # Predict the response for test dataset
y_pred_log_reg


# In[15]:


from sklearn import metrics                              
cnf_matrix = metrics.confusion_matrix(y_test,y_pred_log_reg)     #  a confusion matrix C is such that C[i][j] is equal to the number of observations known to be in group i and predicted to be in group j.
cnf_matrix


# In[16]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_log_reg))           #


# # MLPClassifier Model

# In[17]:


# Multi-layer Perceptron classifier is different from logistic regression, in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers.
from sklearn.neural_network import MLPClassifier   
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100), random_state=1)   # we have one hidden node with 100 hidden nodes in it
clf.fit(x_train_flatten,y_train)  # Fitting the Model


# In[18]:


clf.n_layers_       # number of layers we having are 3 (input,hidden and output layers)


# In[19]:


y_pred=clf.predict(x_test_flatten)   # Predict the response for test dataset


# In[20]:


from sklearn import metrics   
cnf_matrix = metrics.confusion_matrix(y_test,y_pred_log_reg)     # a confusion matrix C is such that C[i][j] is equal to the number of observations known to be in group i and predicted to be in group j.

cnf_matrix


# In[21]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # SVM Model

# In[22]:


# Import svm model
from sklearn import svm

# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
clf.fit(x_train_flatten,y_train)

# Predict the response for test dataset
y_pred_svm = clf.predict(x_test_flatten)


# In[23]:


from sklearn import metrics 
cnf_matrix = metrics.confusion_matrix(y_test,y_pred_svm)    # a confusion matrix C is such that C[i][j] is equal to the number of observations known to be in group i and predicted to be in group j.

cnf_matrix


# In[24]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))

