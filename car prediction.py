#!/usr/bin/env python
# coding: utf-8

# # Import the labraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# # Read the Dataset

# In[5]:


data = pd.read_csv('CAR DETAILS.csv')
data.head()


# # Shape

# In[6]:


data.shape


# # Data Preprocessing

# # 1) Handle the null values

# In[37]:


data.isnull().sum()


# # 2) Handle the dupliactes

# In[39]:


data.duplicated().sum()


# In[41]:


data.drop_duplicates(inplace=True)


# In[43]:


data.duplicated().sum()


# # 3) Check the data types

# In[44]:


data.dtypes


# In[7]:


data.info()


# In[8]:


data.describe().T


# # Data Cleaning And Visualization

# In[10]:


data = data.drop('name', axis=1)
data.head()


# In[11]:


# to know how old the car is we subtract current year with the year in which the car was bought
data['Years_old'] = 2020 - data.year     
data.head()


# In[12]:


data.drop('year', axis=1, inplace=True)
data.head()


# # Use One Hot Encoding

# In[13]:


data = pd.get_dummies(data,drop_first=True)


# In[14]:


data.head()
# here 'Selling_Price' is what we have to predict


# # EDA -Exploratory Data Analysis

# In[15]:


sns.pairplot(data);
# This shows the relationship for (n,2) combination of variable in a DataFrame 
# as a matrix of plots and the diagonal plots are the univariate plots.


# In[45]:


plt.figure(figsize= (15,5))
sns.histplot(x = 'selling_price', kde = True, data = data)


# In[46]:


sns.scatterplot(x= 'km_driven' , y= 'selling_price', data = data)


# In[16]:


plt.figure(figsize=(10,10))
sns.heatmap(
    data.corr(), 
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
);


# In[17]:


data.head()


# In[19]:


X = data.drop('selling_price', axis = 1)
y = data['selling_price']
print(X.shape)
print(y.shape)


# # Cheaking For Important Features!

# In[20]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[21]:


model.feature_importances_


# In[22]:


pd.Series(model.feature_importances_, index=X.columns).plot(kind='bar',alpha=0.75, rot=90);


# # Model Training

# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[24]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[25]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits = 5, test_size=0.2, random_state=0)


# In[26]:


cross_val_score(LinearRegression(), X,y,cv=cv)


# # Finding best model using RandomizedSearchCV

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# In[28]:


def perfect_model(X, y):
  model_algo = {
      
      'Linear_Regression':{
          'model': LinearRegression(),
          'params': {
              'normalize': [True, False]
            }
        },

        'Decision_Tree':{
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse', 'mae'],
                'splitter': ['best', 'random'],
                'max_depth': [x for x in range(5,35,5)],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        },

        'Random_forest':{
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [x for x in range(20,150,20)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [x for x in range(5,35,5)],
                'min_samples_split': [2, 5, 10, 15, 100],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        }
    }
  
  score = []
  cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
  for algo_name, config in model_algo.items():
      rs =  RandomizedSearchCV(config['model'], config['params'], cv=cv, return_train_score=False, n_iter=5)
      rs.fit(X_train,y_train)
      score.append({
          'model': algo_name,
          'best_score': rs.best_score_,
          'best_params': rs.best_params_
      })

  result = pd.DataFrame(score,columns=['model','best_score','best_params'])
  print(result.best_params.tolist())
  return result


# In[29]:


perfect_model(X, y)


# In[30]:


final_dec_model = DecisionTreeRegressor(splitter='best', min_samples_leaf= 2, max_depth=15, criterion='mae')
final_dec_model.fit(X_train,y_train)
final_dec_model.score(X_test,y_test)


# In[31]:


final_rf_model = RandomForestRegressor(n_estimators=120, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=20)
final_rf_model.fit(X_train,y_train)
final_rf_model.score(X_test,y_test)


# In[32]:


cross_val_score(DecisionTreeRegressor(splitter='best', min_samples_leaf= 2, max_depth=15, criterion='mae'), X,y,cv=cv)


# In[33]:


cross_val_score(RandomForestRegressor(n_estimators=120, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=20), X,y,cv=cv)


# # Based on above results we can say that Random Forest Regressor gives the best score

# In[34]:


predictions=final_rf_model.predict(X_test)
plt.scatter(y_test,predictions)


# # Inference
# 1) RF Reg is the best model in terms of Score.

# # Exporting the tested model to a pickle file

# In[35]:


import pickle
with open('RF_price_predicting_model.pkl', 'wb') as file:
  # dump information to that file
  pickle.dump(final_rf_model, file)


# In[ ]:




