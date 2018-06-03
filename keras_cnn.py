
# coding: utf-8

# In[1]:


import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[2]:


import tensorflow as tf


# In[ ]:


tf.nn.


# In[60]:


PATH = os.getcwd()
# Define data path
data_path = 'D:\K_fold'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=3
num_epoch=20


# In[61]:


num_classes = 4


# In[62]:


img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)


# In[63]:


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)


# In[78]:


if num_channel==3:
    if K.image_dim_ordering()=='th':
        img_data= np.expand_dims(img_data, axis=1) 
        print (img_data.shape)
    else:
        img_data= np.expand_dims(img_data, axis=4) 
        print (img_data.shape)

else:
    if K.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,0,1)
        print (img_data.shape)


# In[79]:


USE_SKLEARN_PREPROCESSING=False


# In[80]:


if USE_SKLEARN_PREPROCESSING:
# using sklearn for preprocessing
    from sklearn import preprocessing

    def image_to_feature_vector(image, size=(128, 128)):
# resize the image to a fixed size, then flatten the image into
# a list of raw pixel intensities
        return cv2.resize(image, size).flatten()

    img_data_list=[]
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_flatten=image_to_feature_vector(input_img,(128,128))
            img_data_list.append(input_img_flatten)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    print (img_data.shape)
    img_data_scaled = preprocessing.scale(img_data)
    print (img_data_scaled.shape)
    
    print (np.mean(img_data_scaled))
    print (np.std(img_data_scaled))
    
    print (img_data_scaled.mean(axis=0))
    print (img_data_scaled.std(axis=0))
    
    if K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)

    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)


    if K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)

    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)


# In[81]:


if USE_SKLEARN_PREPROCESSING:
    img_data=img_data_scaled


# In[82]:


num_classes = 4

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3


# In[83]:


names = ['accordion','anchor','brain','chair']


# In[84]:


# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)


# In[85]:


#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[86]:


img_data[1].shape


# In[87]:


# Defining the model
input_shape=img_data[0].shape

model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])


# In[88]:


# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


# In[89]:


# Training
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import tensorflow as tf
import numpy as np
from PIL import Image


# In[ ]:


import tensorflow as tf


# In[3]:


#int weights

def int_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# In[4]:


#int bias

def int_bias(shape):
    init_bias_val = tf.constant(0.1,shape)
    return tf.Variable(init_bias_val)


# In[5]:


#con2d

def conv2d(x,W):
    #x = [batch, h , w , ch]
    #W = [filter h, filter W, ch in , ch out]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# In[6]:


#pooling

def max_pooling(x):
    #x = [batch, h , w , ch]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    


# In[8]:


#convolution layer

def convolution_layer(input_x,shape):
    W = int_weights(shape)
    b = int_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# In[9]:


#fully connected layer

def fully_connected_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = int_weights([input_size,size])
    b = int_bias([size])
    return tf.matmul(input_layer, W) + b
    


# In[ ]:




