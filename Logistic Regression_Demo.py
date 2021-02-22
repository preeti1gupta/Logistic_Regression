#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data=pd.read_csv("E://1. Logistic Regression in Python Demo Part 1 (1)//1. Logistic Regression in Python Demo Part 1//dm.csv",na_values=[""," ","NA","N/A"])


# In[4]:


data.head()


# In[5]:


## Assume people who spend more than the average are good customers
data['target']=data['AmountSpent'].map(lambda x: 1 if x>data['AmountSpent'].mean() else 0)


# In[6]:


data=data.drop("AmountSpent",axis=1)


# In[7]:


data.head()


# In[8]:


data['History'].value_counts()


# In[9]:


data['History'].isnull().sum()


# In[10]:


## Minimal Data Prep
data['History']=data['History'].fillna("NewCust")


# In[12]:


data.head(10)


# In[13]:


## Split the data into test and train
data_train=data.sample(frac=0.70,random_state=200)
data_test=data.drop(data_train.index)


# In[14]:


## Build Model
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[15]:


model1=smf.glm("target~C(Age)+C(Gender)+C(OwnHome)+C(Married)+C(Location)+Salary+Children+C(History)+Catalogs",data=data_train,
              family=sm.families.Binomial()).fit()


# In[16]:


print(model1.summary())


# In[17]:


## Variables to exclude
#Age
#Gender
#Ownhome
#Married
## Variables for dummy creation
#Hist_Low
#Hist_Medium

data_train['Hist_Low']=data_train['History'].map(lambda x: 1 if x=="Low" else 0)
data_test['Hist_Low']=data_test['History'].map(lambda x: 1 if x=="Low" else 0)
data_train['Hist_Med']=data_train['History'].map(lambda x: 1 if x=="Medium" else 0)
data_test['Hist_Med']=data_test['History'].map(lambda x: 1 if x=="Medium" else 0)


# In[18]:


model2=smf.glm("target~Children+Catalogs+Salary+Hist_Med",data=data_train,
              family=sm.families.Binomial()).fit()


# In[19]:


print(model2.summary())


# In[20]:


## Let's check confusion matrix and AUC
import sklearn.metrics as metrics


# In[21]:


y_true=data_test['target']
y_pred=model2.predict(data_test)


# In[22]:


y_pred.head()


# In[23]:


y_true=data_test['target']
y_pred=model2.predict(data_test).map(lambda x:1 if x>0.5 else 0)
metrics.confusion_matrix(y_true,y_pred)


# In[24]:


## ROC curve
y_score=model2.predict(data_test)
fpr,tpr,thresholds=metrics.roc_curve(y_true,y_score)
x,y=np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)


# In[25]:


plt.plot(fpr,tpr,"-")
plt.plot(x,y,'b--')


# In[26]:


## AUC
metrics.roc_auc_score(y_true,y_score)


# In[27]:


## Gains
data_test['prob']=model2.predict(data_test)


# In[28]:


data_test['prob'].head()


# In[29]:


data_test['prob_deciles']=pd.qcut(data_test['prob'],q=10)


# In[30]:


data_test.head()


# In[31]:


data_test.sort_values('prob',ascending=False).head()


# In[32]:


gains=data_test.groupby("prob_deciles",as_index=False)['target'].agg(['sum','count']).reset_index().sort_values("prob_deciles",
                 ascending=False)


# In[33]:


gains.columns=["Deciles","TotalEvents","NumberObs"]


# In[34]:


gains["PercEvents"]=gains['TotalEvents']/gains['TotalEvents'].sum()


# In[35]:


gains["CumulativeEvents"]=gains.PercEvents.cumsum()


# In[36]:


gains


# In[37]:


data_test.sort_values("prob",ascending=False)[['Cust_Id']].head(90)## These are the people to target

