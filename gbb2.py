#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 다른 노트북 작성할 때도 이 셀만 떼서 사용 가능하다.
import matplotlib.pyplot as plt 
import platform                

# 웬만하면 해주는 것이 좋다.
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus']= False

if platform.system() == 'Darwin': # 맥os 사용자의 경우에
    plt.style.use('seaborn-darkgrid') 
    rc('font', family = 'AppleGothic')
    
elif platform.system() == 'Windows':# 윈도우 사용자의 경우에
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    plt.style.use('seaborn-darkgrid') # https://python-graph-gallery.com/199-matplotlib-style-sheets/
    rc('font', family=font_name)


# In[182]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
import json
import requests, bs4
from bs4 import BeautifulSoup
from dateutil.parser import parse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping, ModelCheckpoint
import urllib
import xmltodict
import xml.etree.ElementTree as ET
import scipy.stats
import folium
import scipy.stats as stats
from folium.plugins import MarkerCluster
from folium import plugins
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge,Lasso
import statsmodels.api as sm
os.chdir('..')


# In[115]:


df = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/전국자전거보관소표준데이터.csv',encoding='cp949')


# In[119]:


df


# In[117]:


df.columns = ['name','add','a','lat','lot','have_bike','b','c','d','e','f','g','h','i','data_sel_day','j','city']


# In[135]:


df2 = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/전국자전거대여소표준데이터.csv',encoding='cp949')


# In[148]:


df2


# In[156]:


df2.columns = ['name','add','lat','lot','have_bike','data_sel_day','city']


# In[147]:


df2 = df2.drop(['a','b','c','d','e','f','g','i','j','k','l','m','n','o'], axis = 1)


# In[169]:


fdf.isnull().sum()


# In[153]:


df2.dropna(inplace = True)


# In[157]:


fdf = pd.concat([df,df2], axis=0)


# In[170]:


fdf


# In[166]:


fdf.reset_index(drop=True,inplace = True)


# In[188]:


aa = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/a.csv',encoding='cp949')


# In[197]:


aa


# In[284]:


df.isnull().sum()


# In[191]:


aa.dropna(inplace = True)


# In[192]:


aa.reset_index(drop=True,inplace = True)


# In[194]:


aa = aa.drop(['Unnamed: 0','Unnamed: 0.1'], axis = 1) 


# In[196]:


fdf


# In[ ]:





# In[201]:


fdf.to_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/total_data.csv', encoding = 'cp949')


# In[198]:


map = folium.Map(location =  [35.9078,127.7669], zoom_start =12)


# In[199]:


marker_cluster = MarkerCluster().add_to(map)

for i in fdf.index:
    folium.Marker(location = [fdf.loc[i,"lat"],fdf.loc[i,"lot"]],
                  zoom_start=10,
                  popup=str(fdf.loc[i,"name"])).add_to(marker_cluster)


# In[200]:


map


# In[202]:


map.save('C:/Users/aiicon-KJG/Desktop/진기/지비비/all_map_total.html')


# In[68]:


map = folium.Map(location =  [35.9078,127.7669], zoom_start =12)


# In[69]:


marker_cluster = MarkerCluster().add_to(map)

for i in df.index:
    folium.Marker(location = [df.loc[i,"lat"],df.loc[i,"lot"]],
                  zoom_start=12,
                  popup=str(df.loc[i,"name"])).add_to(marker_cluster)


# In[30]:


map


# In[70]:


map.save('C:/Users/aiicon-KJG/Desktop/진기/지비비/all_map2.html')


# In[71]:


df = df[['city','have_bike']].groupby(by=['city']).sum()


# In[72]:


df = df.reset_index().rename(columns={"index": "id"})


# In[73]:


df


# In[208]:


df = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/지역별 자전거보유대수_total.csv',encoding='cp949')


# In[12]:


df = df.fillna(0)


# In[107]:


df['have_bike2'] = df['have_bike2'].apply(lambda x: int(x))


# In[209]:


df


# In[114]:


plt.figure(figsize=(30,7))
plt.xticks(rotation = 90)
plt.title('지역별 보유대수')
sns.barplot(data=df, x="city", y="total")


# In[10]:


df = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/total_data.csv',encoding='cp949')


# In[13]:


df


# In[40]:


qq = df.set_index(df['city'])


# In[42]:


qq = qq.drop(['city','total'], axis = 1) 


# In[66]:


qq


# In[56]:


ww = qq.drop(['person'], axis = 1) 


# In[73]:


ww.plot(kind='bar',figsize=(30,5))
plt.show()


# In[297]:


df = df.drop(['city'], axis = 1) 


# In[294]:


df = pd.read_csv('C:/Users/aiicon-KJG/Desktop/진기/지비비/전국자전거대여소표준데이터/fff.csv',encoding='cp949')


# In[295]:


df.dropna(inplace = True)


# In[6]:


df = df.set_index(df['city'])


# In[298]:


df


# In[206]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot = True)


# In[217]:


x_data = df[['have_bike','have_bike2','2018_sp','2019_sp','2020_sp','2018_rain_sp','2019_rain_sp','2020_rain_sp']]
target = df[['play_bike']]


# In[221]:


x_data2 = df[['have_bike','have_bike2','2018_su','2019_su','2020_su','2018_rain_su','2019_rain_su','2020_rain_su']]
target = df[['play_bike']]


# In[225]:


x_data3 = df[['have_bike','have_bike2','2018_fa','2019_fa','2020_fa','2018_rain_fa','2019_rain_fa','2020_rain_fa']]
target = df[['play_bike']]


# In[245]:


x_data4 = df[['have_bike','have_bike2','2018_wi','2019_wi','2020_wi','2018_rain_wi','2019_rain_wi','2020_rain_wi']]
target = df_target[['play_bike']]


# In[246]:


x_data1 = sm.add_constant(x_data4, has_constant='add')


# In[247]:


multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()


# In[248]:


fitted_multi_model.summary()


# In[299]:


train_val = df.values


# In[300]:


train_val = pd.DataFrame(train_val)


# In[310]:


test_df = df[['have_bike','have_bike2','2018_rain_sp','2018_rain_su','2018_rain_fa','2018_rain_wi','2019_rain_sp','2019_rain_su','2019_rain_fa','2019_rain_wi','2020_rain_sp','2020_rain_su','2020_rain_fa','2020_rain_wi','play_bike']]


# In[311]:


test_df


# In[312]:


scaler = MinMaxScaler()
scaler.fit(test_df)


# In[313]:


train_scaler = scaler.transform(test_df)


# In[314]:


new_df = pd.DataFrame(train_scaler)
new_df.columns = test_df.columns


# In[ ]:





# In[315]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(new_df.values, i) 
                     for i in range(new_df.shape[1])]
vif["features"] = new_df.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[317]:


vif = pd.DataFrame()
x_data= new_df.drop('2019_rain_sp',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data.values, i) for i in range(x_data.shape[1])]
vif["features"] = x_data.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[318]:


vif = pd.DataFrame()
x_data= x_data.drop('2019_rain_fa',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data.values, i) for i in range(x_data.shape[1])]
vif["features"] = x_data.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[319]:


vif = pd.DataFrame()
x_data= x_data.drop('2019_rain_su',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data.values, i) for i in range(x_data.shape[1])]
vif["features"] = x_data.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[320]:


vif = pd.DataFrame()
x_data= x_data.drop('2020_rain_su',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data.values, i) for i in range(x_data.shape[1])]
vif["features"] = x_data.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[321]:


vif = pd.DataFrame()
x_data= x_data.drop('2018_rain_fa',axis=1)
vif["VIF Factor"] = [variance_inflation_factor(
    x_data.values, i) for i in range(x_data.shape[1])]
vif["features"] = x_data.columns
vif.sort_values(by='VIF Factor', ascending=True)


# In[324]:


x_data


# In[358]:


x = x_data.drop(['play_bike'], axis = 1)
y = pd.DataFrame(x_data)


# In[361]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state = 0)


# In[363]:


ridge = Ridge()
ridge.fit(X_train, y_train)


# In[364]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[374]:


lasso.fit(X_train,y_train)


# In[365]:


print('{:.3f}'.format(ridge.score(X_train, y_train)))
print('{:.3f}'.format(lr.score(X_train, y_train)))
print('{:.3f}'.format(ridge.score(X_test, y_test)))
print('{:.3f}'.format(lr.score(X_test, y_test)))


# In[379]:


alpha_train_score = []
alpha_test_score = []
alpha_set = [0.1, 1 ,10]

for i in alpha_set:
    ridge = Ridge(alpha = i)
    ridge.fit(X_train, y_train)
    
    ridge_tr_score = round(ridge.score(X_train, y_train), 3)
    ridge_te_score = round(ridge.score(X_test, y_test), 3)
    
    alpha_train_score.append(ridge_tr_score)
    alpha_test_score.append(ridge_te_score)

print(alpha_train_score)
print(alpha_test_score)


# In[389]:


ridge_result = []
lasso_result = []
alpha = [0.001,0.01,0.1,1,10]

for a in alpha:   
    ridge = Ridge(alpha = a)
    lasso = Lasso(alpha = a)

    ridge.fit(X_train,y_train)
    lasso.fit(X_train,y_train)

    ridge_y_hat = ridge.predict(X_test)
    lasso_y_hat = lasso.predict(X_test)

    ridge_r2, lasso_r2 = r2_score(y_test,ridge_y_hat), r2_score(y_test,lasso_y_hat)
    ridge_result.append(ridge_r2)
    lasso_result.append(lasso_r2)


# In[384]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(5), ridge_result, '-x', c='blue', label="R2 of Ridge")
ax.plot(range(5), lasso_result, '-x', c='red', label="R2 of Lasso")
plt.xticks(range(5), alpha)
plt.xlabel('alpha')
plt.ylabel('R2')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[390]:


ridge_r2, lasso_r2 = r2_score(y_test,ridge_y_hat), r2_score(y_test,lasso_y_hat)
ridge_MSE, lasso_MSE = mean_squared_error(y_test,ridge_y_hat), mean_squared_error(y_test,lasso_y_hat)
ridge_MAE, lasso_MAE = mean_absolute_error(y_test,ridge_y_hat), mean_absolute_error(y_test,lasso_y_hat)


# In[392]:


print('R2 score - Ridge: %.2f, Lasso: %.2f' %(ridge_r2, lasso_r2))
print('MSE - Ridge: %.2f, Lasso: %.2f' %(ridge_MSE, lasso_MSE))
print('MAE - Ridge: %.2f, Lasso: %.2f' %(ridge_MAE, lasso_MAE))


# In[ ]:





# In[ ]:




