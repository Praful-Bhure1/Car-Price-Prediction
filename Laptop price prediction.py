#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  # data preprocessing
import numpy as np   # mathematical computation
import matplotlib.pyplot as plt # visualization
import seaborn as sns   # visualization


# In[6]:


# df is a vraiable which stores dataframe
df = pd.read_csv('laptop_price_data.csv')
print(type(df))  # dataframe
df.head()  # top 5 rows


# In[7]:


df.shape
# rows = 1302,columns= 13


# In[8]:


# isnulll()  returns True if a cell contains cnull values
df.isnull().sum()
# Returns sum of null values for each column


# In[9]:


df['Unnamed: 0'].nunique()


# In[10]:


df.drop('Unnamed: 0',axis=1,inplace=True)
df.columns


# In[11]:


# dupliacted() returns True, if the rows is duplicated (repeated)
df.duplicated().sum()
# Returns the sum of all the entire duplicated rows


# In[12]:


# inpalce=True ensures that changes are reflected in the actual dataframe.
df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[13]:


df.shape
# rows=1272,cols=12


# In[14]:


df.dtypes


# In[15]:


cat_cols = df.dtypes[df.dtypes=='object'].index
num_cols = df.dtypes[df.dtypes!='object'].index
print(cat_cols)
print(num_cols)


# In[16]:


print(df.columns)
df.head()


# In[17]:


df['Company'].value_counts()


# In[18]:


sns.countplot(y = df['Company'], order=df['Company'].value_counts()[:7].sort_values(ascending=False).index)
plt.title('Top 7 Most bought laptop brands')
plt.show()


# In[19]:


df['TypeName'].value_counts()


# In[20]:


sns.countplot(y=df['TypeName'],order=df['TypeName'].value_counts().sort_values(ascending=False).index)
plt.title('Count of different laptop types')
plt.show()


# In[21]:


df.columns


# In[22]:


sns.boxplot(y=df['Cpu brand'],x=df['Price'])
plt.show()


# In[23]:


sns.boxplot(y=df['Gpu brand'],x=df['Price'])
plt.show()


# In[24]:


sns.displot(df['Price'])
plt.show()


# In[25]:


corr = df.corr()
sns.heatmap(corr,annot=True,cmap='RdBu')
plt.show()


# In[26]:


print(num_cols)


# In[27]:


for i in num_cols:
    sns.boxplot(x=df[i])
    plt.title(f'Boxplot for {i}')
    plt.show()


# In[29]:


print(df[df['Weight']>3.5].shape)
print(df.shape)
print(46/127)


# In[30]:


# Outlier Clipping or Outlier capping
df['Weight'] = np.where(df['Weight']>3.5,3.5,df['Weight'])
sns.boxplot(x=df['Weight'])
plt.title('Boxplot for Weight')
plt.show()


# In[31]:


df['Ram'].value_counts()


# In[32]:


# df[df['Price']>150000].head(30)


# In[33]:


# df['SSD'].value_counts()


# In[34]:


x = df.drop('Price',axis=1)
y = df['Price']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[37]:


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


# In[38]:


def eval_model(ytest,ypred):
    mae= mean_absolute_error(ytest,ypred)
    mse = mean_squared_error(ytest,ypred)
    rmse= np.sqrt(mse)
    r2s = r2_score(ytest,ypred)
    print("MAE",mae)
    print("MSE",mse)
    print("RMSE",rmse)
    print('R2_Score',r2s)


# In[39]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[40]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[41]:


print(df.columns)
print(cat_cols)  # [0,1,6,9,10]
print(x_train.columns)


# In[42]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = LinearRegression()

pipe_lr = Pipeline([('step1',step1),('step2',step2)])

pipe_lr.fit(x_train,y_train)
ypred_lr = pipe_lr.predict(x_test)
eval_model(y_test,ypred_lr)


# In[43]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = Ridge(alpha=4)

pipe_rid = Pipeline([('step1',step1),('step2',step2)])

pipe_rid.fit(x_train,y_train)
ypred_rid = pipe_rid.predict(x_test)
eval_model(y_test,ypred_rid)


# In[44]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = Lasso(alpha=7)

pipe_las = Pipeline([('step1',step1),('step2',step2)])

pipe_las.fit(x_train,y_train)
ypred_las = pipe_las.predict(x_test)
eval_model(y_test,ypred_las)


# In[45]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = KNeighborsRegressor(n_neighbors=7)

pipe_knn = Pipeline([('step1',step1),('step2',step2)])

pipe_knn.fit(x_train,y_train)
ypred_knn = pipe_knn.predict(x_test)
eval_model(y_test,ypred_knn)


# In[46]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = DecisionTreeRegressor(criterion='squared_error',max_depth=10,min_samples_split=15)

pipe_dt = Pipeline([('step1',step1),('step2',step2)])

pipe_dt.fit(x_train,y_train)
ypred_dt = pipe_dt.predict(x_test)
eval_model(y_test,ypred_dt)


# In[47]:


step1 = ColumnTransformer(transformers=
                          [('encoder',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                         remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=100,criterion='squared_error',
                              max_depth=10,min_samples_split=15,random_state=5)

pipe_rf = Pipeline([('step1',step1),('step2',step2)])

pipe_rf.fit(x_train,y_train)
ypred_rf = pipe_rf.predict(x_test)
eval_model(y_test,ypred_rf)


# In[48]:


import pickle


# In[49]:


pickle.dump(pipe_rf,open('rf.pkl','wb'))  # wb= write binary
pickle.dump(df,open('df.pkl','wb'))  # wb= write binary


# In[50]:


df.columns


# In[51]:


df['Ram'].unique()


# In[52]:


df['Touchscreen'].value_counts()


# In[53]:


df['Ips'].value_counts()


# In[54]:


print(df['HDD'].unique())
print(df['SSD'].unique())


# In[ ]:




