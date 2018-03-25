
# coding: utf-8

# In[20]:


'''
根据以下博客编写，第一次上手keras，MNIST就算是hello world啦~
http://nooverfit.com/wp/keras-手把手入门1-手写数字识别-深度学习实战/#more-2354
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import numpy as np


# In[21]:


batch_size=128
num_classes=10
epochs=12

img_rows,img_cols=28,28
#(x_train,y_train),(x_test,y_test)=mnist.load_data()
path='mnist.npz'
f = np.load(path)  
x_train, y_train = f['x_train'], f['y_train']  
x_test, y_test = f['x_test'], f['y_test'] 
print(x_train)

if K.image_data_format()=='channels_first':
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)


# In[22]:


#把数据变成float32更精确
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train sampeles')
print(x_test.shape[0],'test samples')

import keras
#把类别0-9变成二进制，方便训练
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)


# In[25]:


#构造训练CNN
model=Sequential()
model.add(Conv2D(32,
                 activation='relu',
                 input_shape=input_shape,
                 nb_row=3,
                 nb_col=3))
model.add(Conv2D(64,activation='relu',nb_row=3,nb_col=3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

