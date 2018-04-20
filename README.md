


# In[2]:


#import libraries
import librosa
get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa.display
import glob 
import IPython.display # display audio data as sound
import keras


# In[3]:


get_ipython().system('pip install data_utils')


# In[4]:


#Load the audio data file
data, sampling_rate = librosa.load('E:/gaurav_back up/Spring 2018/voice_recognition/train/Train/2022.wav')


# In[5]:


# What is the numerical output of a audio file?
data,sampling_rate


# # STEP1: Visualization

# In[6]:


#By hearing
import IPython.display as ipd
ipd.Audio('E:/gaurav_back up/Spring 2018/voice_recognition/train/Train/2022.wav')


# In[7]:


#By graphs
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)


# STEP 2: Insights about data.

# In[8]:


# Setting directories becuase we have 5435 .wav,individual files to read. 
root_dir = os.path.abspath('.')
data_dir_train = 'E:/gaurav_back up/Spring 2018/voice_recognition/train/Train/'
train = pd.read_csv('E:/gaurav_back up/Spring 2018/voice_recognition/train/train.csv')
data_dir_test = 'E:/gaurav_back up/Spring 2018/voice_recognition/test/Test/'
test = pd.read_csv('E:/gaurav_back up/Spring 2018/voice_recognition/test/test.csv')


# In[34]:


#Magic of randamization
i = random.choice(train.index)
print(i)
audio_name = train.ID[i]
path = os.path.join(data_dir_train, 'Train', str(audio_name) + '.wav')

print('Class: ', train.Class[i])
x, sr = librosa.load(data_dir_train + str(train.ID[i]) + '.wav')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr)


# Step 3: Let's have out first model. Majority wins

# In[10]:


(train.Class.value_counts())


# In[11]:


train.Class.value_counts()[:].plot(kind='barh')


# In[12]:


#Model 1: Concept of majority
# a daily life model. it is averaging or just labeling every one based on  mazority
test['Class'] ='jackhammer'
test.to_csv('sub01.csv',index=False)


# STEP 4: Feature extraction

# In[13]:


def parser(row):
   # function to load files and extract features
   data_dir = 'E:/gaurav_back up/Spring 2018/voice_recognition/train/'
   file_name = os.path.join(os.path.abspath(data_dir),'Train',str(row.ID) + '.wav')
   print(file_name) 
   # handle exception to check if there isn't a file which is corrupted
   try:
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      #print(X,sample_rate)
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
print(temp)


# STEP 5: Convert the data to pass it in Deep learning model

# In[14]:


from sklearn.preprocessing import LabelEncoder
import keras
import numpy

#For training set
X1 = np.array(temp.feature.tolist())
X =list(filter(None.__ne__, X1))

#For training set
y1 = np.array(temp.label.tolist())
y = [x for x in y1 if x != None]

lb = LabelEncoder()

#Label encoding for training set
p=lb.fit_transform(y)
print(p)
y=keras.utils.to_categorical(p)
print(y)


# In[15]:


np.array(X).shape


# In[16]:


len(X1)


# In[17]:


len(X)


# Step 6:Run a deep learning model and study results

# In[18]:


import numpy as np
from keras.models import Sequential #We can add individual layers to our neural network one layer at a time.
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam,RMSprop
#from keras.utils import np_utils
from sklearn import metrics 

num_labels = y.shape[1]
print(num_labels)
filter_size = 2

model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


# In[19]:


#(b)
# optimizer function that are faster than gradient descent
#1 Momentum Optimization
#2 Nesterov Accelerated Gradient
#3 RMSProp()
#4 Adam
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')


# 
# 
# http://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=6,2&seed=0.86127&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

# In[20]:


model.summary()


# Step 7 : Train the model

# In[21]:


model.fit(np.array(X),np.array(y),batch_size=32,epochs=5)


# In[22]:


# cross_validation
model.fit(np.array(X),np.array(y),batch_size=32,epochs=10,validation_split=0.3)
# we see there are better results on the validation set


# Step 8 : Training to Testing

# In[23]:


def parser2(row):
   # function to load files and extract features
   data_dir = 'E:/gaurav_back up/Spring 2018/voice_recognition/test/'
   file_name = os.path.join(os.path.abspath(data_dir),'Test',str(row.ID) + '.wav')
   print(file_name) 
   # handle exception to check if there isn't a file which is corrupted
   try:
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      #print(X,sample_rate)
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      
      print("Error encountered while parsing file: ", file_name)
      return None
 
   feature = mfccs
   #label = row.Class
 
   return [feature]

temp1 = test.apply(parser2, axis=1)

# #temp1.columns = ['ID','feature']
# print(temp1)


# In[24]:


print(temp1)
temp1.columns = ["feature"]
print(temp1)


# In[26]:


#Processing the test file. Removing corrupt .wav file
from sklearn.preprocessing import LabelEncoder
import np_utils
import keras
import numpy

#For testing set
X1 = np.array(temp1.tolist())
test_X =list(filter(None.__ne__, X1))

#For testing set
p1=np.array(test)


# In[44]:


np.array(temp2).shape


# In[45]:


test_X = np.reshape(temp2, (2131,40),1)


# In[47]:


#test_X
#model1
pred = model.predict_classes((np.array(test_X)))
pred = lb.inverse_transform(pred)
test['Class'] = pred
#test.to_csv(‘sub02.csv’, index=False)


# In[30]:


pred


# In[39]:


temp2 = [t for t in temp1 if t is not None]


# In[42]:


temp2


# key take aways:
#  Nomalization is important as neural network usually work better than. It has mean of 0 and unit variance.
# #scikit_learn's StandardScaler can do this,but do not forget to use it.
