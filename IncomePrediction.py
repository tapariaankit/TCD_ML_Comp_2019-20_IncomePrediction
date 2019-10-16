#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries and training data

# In[247]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
df.head()


# ## Analysing the training dataset

# In[248]:


df.info()


# In[249]:


df.describe()


# ## EDA

# In[250]:


sns.pairplot(df)


# ## Function to remove outliers

# In[251]:


def detect_outlier(data_1):
    outliers=[]
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# ## Function to clean the data, fill missing values, scale the features and choose the features for building the model

# In[252]:


def featureEngineering(path):
    from sklearn.preprocessing import StandardScaler
    
    df=pd.read_csv(path)
    
    scaler = StandardScaler()
    df['Size of City'] = df.groupby("Country")['Size of City'].apply(lambda x: x.fillna(x.mean()))
    df['Size of City'] = (df['Size of City'] - df['Size of City'].min())/ (df['Size of City'].max() - df['Size of City'].min())
   
    df['Profession'] = df['Profession'].str.lower(); 
    df['University Degree'] = df['University Degree'].str.lower(); 
    df['Country'] = df['Country'].str.lower(); 
    
    df['Year of Record'] = df.groupby(["Country"])['Year of Record'].apply(lambda x: x.fillna(x.mode()[0]))
    df['Year of Record'] = (df['Year of Record'] - df['Year of Record'].min())/ (df['Year of Record'].max() - df['Year of Record'].min())
    df['Profession'].fillna('9999',inplace=True)
    df['University Degree'].fillna(df['University Degree'].mode(),inplace=True)
    
    if 'Income in EUR' in df.columns:
        df = df[df['Income in EUR']>0]
        df['Income in EUR'] = np.log10(df['Income in EUR'])
        df = df[~df['Income in EUR'].isin(detect_outlier(df['Income in EUR']))] #Removing outliers from Income
        df = df[~df['Body Height [cm]'].isin(detect_outlier(df['Body Height [cm]']))] #Removing outliers from Body Height
        
        df2= df[['Age','Income in EUR','Size of City','Body Height [cm]','Country','Year of Record','Profession','University Degree']]
    else:
        df2= df[['Age','Size of City','Body Height [cm]','Country','Year of Record','Profession','University Degree']]
    df2['Age'].fillna(df2['Age'].mean(), inplace = True)
    df2 = pd.get_dummies(data=df2, columns=['Country','Profession','University Degree'])
    df2['Age'].fillna(df2['Age'].mean(), inplace = True)
    
    return df2


# In[253]:


df2 = featureEngineering('tcd ml 2019-20 income prediction training (with labels).csv')
X = df2.loc[:, df2.columns != 'Income in EUR']
y = df2['Income in EUR'].values.reshape(-1,1)
df3 = featureEngineering('tcd ml 2019-20 income prediction test (without labels).csv')

# Equalise columns for training and to be predicted datasets
for x in list(X.columns.difference(df3.columns)):
    df3[x]= 0
for x in list(df3.columns.difference(X.columns)):
    X[x]= 0
df3= df3[X.columns]


# ## Visualising the 'Income in EUR' and 'Size of City' after scaling them

# In[208]:


sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    (df2['Income in EUR']), norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Income in EUR', ylabel='Count');


# In[209]:


sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    (df2['Size of City']), norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Size of City', ylabel='Count');


# ## Create, train a Linear Regression Model, Predict and Calculate RSME

# In[258]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(10**(y_test), 10**(predictions))))


# ## Visualizing the actual and predicted income

# In[259]:



pred_df = pd.DataFrame({'Actual': 10**y_test.flatten(), 'Predicted': 10**predictions.flatten()})
pred_df
pred_df1 = pred_df.head(25)
pred_df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ## Calculate RSME w.r.t baseline (my 1st submission) and store predictions to CSV 

# In[255]:


predictions = lm.predict(df3)
ee = pd.read_csv('faulty result.csv')
pred_income = pd.DataFrame(data=10**(predictions),
            columns=['Income'])
dd = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
dd = dd[['Instance']]
result = pd.concat([dd, pred_income], axis=1)
print(result.head())

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(ee["Income"], 10**(predictions))))

result.to_csv('tcd ml 2019-20 income prediction submission file.csv', encoding='utf-8', index=False)

