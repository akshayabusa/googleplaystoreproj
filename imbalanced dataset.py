#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np

df=pd.read_csv('datasets/creditcard.csv')
df.head()


# In[27]:



df['Class'].value_counts()


# In[37]:


X=df.drop('Class',axis=1)
Y=df.Class


# In[29]:


df.info()


# In[30]:


df.isnull().values.any()


# In[31]:


countclasses=pd.value_counts(df["Class"],sort=True)
countclasses.plot(kind='bar',rot=0)


# In[32]:


FRAUD=df[df['Class']==1]
NORMAL=df[df['Class']==0]


# In[33]:


print(FRAUD.shape,NORMAL.shape)


# In[34]:


from imblearn.under_sampling import NearMiss


# In[38]:


nm=NearMiss()
X_res,Y_Res=nm.fit_sample(X,Y)


# In[39]:


X_res.shape,Y_Res.shape


# In[43]:


from collections import Counter
print ('the orginal dataset: {}'.format(Counter(Y)))
print('the resampled dataset:{}'.format(Counter(Y_Res)))


# In[45]:


from imblearn.combine import SMOTETomek


# In[46]:


sm=SMOTETomek(random_state=42)
x_res,y_res=sm.fit_sample(X,Y)


# In[47]:


x_res.shape,y_res.shape


# In[48]:


from collections import Counter
print ('the orginal dataset: {}'.format(Counter(Y)))
print('the resampled dataset:{}'.format(Counter(y_res)))


# In[ ]:





# In[ ]:




