#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


google_data = pd.read_csv('googleplaystore.csv')


# In[4]:


google_data.head()     


# In[5]:


google_data.shape


# In[6]:


google_data.describe() 


# In[7]:


google_data.boxplot()


# In[8]:


google_data.hist()


# In[9]:


google_data.info()


# In[10]:


google_data.isnull()


# In[11]:


google_data.isnull().sum()


# In[13]:


google_data[google_data.Rating>5]


# In[15]:


google_data.drop([10472],inplace=True)


# In[17]:


google_data[10471:10475]


# In[18]:


google_data.boxplot()


# In[19]:


google_data.hist()


# In[20]:


def impute_median(series):
    return series.fillna(series.median())


# In[21]:


google_data.Rating = google_data['Rating'].transform(impute_median)


# In[22]:


google_data.isnull().sum()


# In[23]:


print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())


# In[24]:


google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace=True)


# In[25]:


google_data.isnull().sum()


# In[26]:


google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_data['Price'] = google_data['Price'].apply(lambda x: float(x))
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce')


# In[27]:


google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))


# In[28]:


google_data.head(10)


# In[29]:


google_data.describe()


# In[30]:


grp = google_data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)


# In[31]:


plt.figure(figsize=(12,5))
plt.plot(x, "ro", color='g')
plt.xticks(rotation=90)
plt.show()


# In[32]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro', color='r')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()


# In[33]:


plt.figure(figsize=(16,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




