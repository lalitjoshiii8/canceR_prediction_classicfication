#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('cancer_classification.csv')


# In[4]:


df.info()


# In[59]:


df.describe().transpose()


# ## EDA

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


sns.countplot(x='benign_0__mal_1',data=df)


# In[7]:


sns.heatmap(df.corr())


# In[8]:


df.corr()['benign_0__mal_1'].sort_values()


# In[9]:


df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')


# In[10]:


df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# ## Train Test Split

# In[11]:


X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)


# 
# ## Scaling Data

# In[14]:


from sklearn.preprocessing import MinMaxScaler


# In[15]:


scaler = MinMaxScaler()


# In[16]:


scaler.fit(X_train)


# In[17]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[18]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# In[19]:


X_train.shape


# In[20]:


model = Sequential()



model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[21]:


# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )


# In[22]:


# model.history.history


# In[114]:


model_loss = pd.DataFrame(model.history.history)


# In[115]:


# model_loss


# In[116]:


model_loss.plot()


# ## Example Two: Early Stopping
# 
# We obviously trained too much! Let's use early stopping to track the val_loss and stop training once it begins increasing too much!

# In[117]:


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[36]:


from tensorflow.keras.callbacks import EarlyStopping


# In[37]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[39]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[40]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# ## Example Three: Adding in DropOut Layers

# In[41]:


from tensorflow.keras.layers import Dropout


# In[42]:


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[43]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


# In[44]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# # Model Evaluation

# In[52]:


predictions = model.predict(X_test)
classes=np.argmax(predictions,axis=1)


# In[56]:


from sklearn.metrics import classification_report , confusion_matrix


# In[ ]:


print(classification_report (y_test , predictions))


# In[ ]:


print(confusion_matrix(y_Test , predictions))

