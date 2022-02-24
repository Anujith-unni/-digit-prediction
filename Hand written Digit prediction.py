#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


pip install opencv-python


# In[3]:


import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


mnist=k.datasets.mnist
(Xtrain,ytrain),(Xtest,ytest)=mnist.load_data()


# In[5]:


Xtrain[10]


# In[6]:


ytrain[10]


# In[7]:


plt.imshow(Xtrain[10],cmap='gray')
plt.show()


# In[8]:


## Build the model
model = k.models.Sequential()
### Add the layers
model.add(k.layers.Flatten())  ## input layer
model.add(k.layers.Dense(784,activation="relu"))   ## hidden layer
model.add(k.layers.Dense(10,activation='softmax'))    ## output layer
### Compile the model
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam"
              ,metrics=["accuracy"])


# In[9]:


### scale the data/Normalize
Xtrain_scaled=Xtrain/255
Xtest_scaled=Xtest/255


# In[10]:


### train the model
model.fit(Xtrain_scaled,ytrain,epochs=15)


# In[11]:


## prediction
ypred=model.predict(Xtest_scaled)


# In[12]:


ytest[100]


# In[13]:


ypred[100]


# In[14]:


ypred[100].argmax()


# In[15]:


model.evaluate(Xtest_scaled,ytest)


# In[16]:


import cv2 
img = cv2.imread("2.png",0)
img.shape
## preprocessing of image
img = cv2.resize(img,(28,28,))
img = cv2.bitwise_not(img)
img = img/255
plt.imshow(img,cmap='gray')
plt.show()


# In[17]:


model.predict(np.array([img]))


# In[18]:


model.predict(np.array([img])).argmax()


# In[ ]:




