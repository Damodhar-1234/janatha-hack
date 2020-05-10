#!/usr/bin/env python
# coding: utf-8

# In[262]:


import pandas as pd
import numpy as np


# In[264]:


#loading train file
df = pd.read_csv("C:\\Users\\Hemanth Vuribindi\\Desktop\\train_8wry4cB.csv")


# In[150]:


df


# In[265]:


df.gender


# In[266]:


df.isnull().sum()/len(df)*100


# In[267]:


#upsampling minor samples
from sklearn.utils import resample
df_majority = df[df.gender==1]
df_minority = df[df.gender==0]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True,n_samples=8192,random_state=123) 
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.gender.value_counts()
# to match majority class
# reproducible results
# sample with replacement


# In[268]:


df


# In[269]:


df['gender']


# In[270]:


df


# In[271]:


X = df.iloc[:,0:4]


# In[272]:


X


# In[273]:


Y = df.iloc[:,4]


# In[274]:


Y


# In[275]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[162]:


X,Y = make_classification(n_samples=10500, n_features=4,
                           n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)


# In[163]:


pred_y = clf.predict(X)


# In[164]:


pred_y


# In[165]:


print( accuracy_score(pred_y, Y) )


# In[276]:


df_1 = pd.read_csv("C:\\Users\\Hemanth Vuribindi\\Desktop\\test_Yix80N0.csv")


# In[277]:


df_1


# In[278]:


df


# In[308]:


#random forest classifier
clf


# In[309]:


#loading classifier on test data
df_1,Y = make_classification(n_samples=4499, n_features=4,
                           n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(df_1,Y)


# In[310]:


pred_y_1 = clf.predict(df_1)


# In[320]:


print( accuracy_score(pred_y_1, Y) )


# In[321]:


pred_y_1


# In[322]:


len(pred_y_1)


# In[316]:


pred_y_1


# In[317]:


df_final = pd.DataFrame(pred_y_1)


# In[318]:


df_final


# In[319]:


df_final.to_csv("C:\\Users\\Hemanth Vuribindi\\Desktop\\Book2.csv", encoding='utf-8', index=False)


# In[327]:


#loading test file
df_sol = pd.read_csv("C:\\Users\\Hemanth Vuribindi\\Desktop\\test_Yix80N0.csv")


# In[328]:


df_sol


# In[329]:


df_sol


# In[ ]:





# In[ ]:




