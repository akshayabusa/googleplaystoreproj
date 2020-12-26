#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd


# In[91]:


df=pd.read_csv('car data.csv')


# In[92]:


df.shape


# In[93]:


print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[94]:


df.isnull().sum()


# In[95]:


df.describe()


# In[96]:


df.head()


# In[97]:


final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

final_dataset.head()


# In[98]:


final_dataset['current_year']=2020


# In[99]:


final_dataset.head()


# In[100]:


final_dataset['no_year']=final_dataset['current_year']- final_dataset['Year']


# In[101]:


final_dataset.head()


# In[102]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[103]:


final_dataset.head()


# In[104]:



final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[105]:


final_dataset.head()


# In[106]:


final_dataset=final_dataset.drop(['current_year'],axis=1)


# In[107]:


final_dataset.head()


# In[108]:


final_dataset.corr()


# In[109]:


import seaborn as sns


# In[110]:


sns.pairplot(final_dataset)


# In[111]:


import matplotlib.pyplot as plt


# In[112]:


corrmat=df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[113]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[114]:


X.head()


# In[115]:


y.head()


# In[116]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[117]:



print(model.feature_importances_)


# In[118]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[119]:


from sklearn.ensemble import RandomForestRegressor


# In[120]:


model=RandomForestRegressor()


# In[121]:


import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[122]:


from sklearn.model_selection import RandomizedSearchCV


# In[123]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[124]:


random_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}
print(random_grid)


# In[125]:


rf=RandomForestRegressor()


# In[142]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[143]:



rf_random.fit(X_train,y_train)


# In[144]:


rf_random.best_params_


# In[145]:


rf_random.best_score_


# In[146]:



predictions=rf_random.predict(X_test)


# In[147]:



sns.distplot(y_test-predictions)


# In[148]:


plt.scatter(y_test,predictions)


# In[150]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[151]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:





# In[ ]:





# In[ ]:




