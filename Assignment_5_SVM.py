#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Reference 1 : Hints & information provided by Prof. Matthew D. Murphy
# Reference 2 : Raschka textbook on Python & Machine Learning 


# Importing necessary libraries

import sklearn
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


# In[2]:


# Reading the treasury yield curve dataset using pandas

tyc_df = pd.read_csv("hw5_treasury_yield_curve_data.csv")


# In[3]:


# dropping the date column from the dataset

tyc_df = tyc_df.drop(["Date"],axis =1)


# In[4]:


# Summary statistics of the dataset

tyc_df.describe()


# In[5]:


# Checking for missing values : if unique is 1 and if it corresponds to False, then no missing values

tyc_df.isnull().describe() 


# In[6]:


# Printing top 5 samples of the dataset

tyc_df.head()


# In[7]:


# Creating feature & target variables

X = tyc_df.drop(["Adj_Close"], axis = 1)
y = tyc_df["Adj_Close"]


# In[8]:


# Printing top 5 feature and target samples

print(X.head(),"\n\n")
print(y.head())


# In[9]:


# Checking the dimensions of feature and target matrices

print(X.shape,"\n\n",y.shape)


# In[11]:


# Checking the data type of all the variables

tyc_df.dtypes


# In[13]:


# Data Normalisation

sc_x  = StandardScaler()
sc_y  = StandardScaler()

X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# In[16]:


# Printing normalised samples

print(X[0],y[0])


# In[17]:


# train test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15,random_state = 42)


# In[19]:


print(X_train.shape,"\n",X_test.shape,"\n",y_test.shape,"\n",y_train.shape)


# In[21]:


# PCA using sklearn : all components

from sklearn.decomposition import PCA

pca         = PCA()

X_pca       = pca.fit_transform(X)

exp_var     = pca.explained_variance_ratio_

cum_exp_var = np.cumsum(exp_var)

print("explained variance: ", exp_var,"\n")
print("cummulative explained variance: ", cum_exp_var)


# In[23]:


len(exp_var)


# In[32]:


pca_ev = plt.bar(range(1,31),exp_var,
                 label = 'individual explained variance')
pca_ev = plt.step(range(1,31),cum_exp_var,
                  label = 'cumulative explained variance')

pca_ev = plt.ylabel("Explained variance ratio")
pca_ev = plt.xlabel("Principal component index")
pac_ev = plt.legend(loc='best')
plt.show()


# In[34]:


# PCA with 3 components

pca_3   = PCA(n_components = 3)

X_pca_3 = pca_3.fit_transform(X)

exp_var_3 = pca_3.explained_variance_ratio_

cum_exp_var_3 = np.cumsum(exp_var_3)

print("explained variance of 3 components: ", exp_var_3, "\n\n")

print("cumulative explained variance of 3 components: ", cum_exp_var_3)


# In[40]:


pca_ev_3 = plt.bar(range(1,4),exp_var_3, label = "individual explained variance of 3 PCs")
pca_ev_3 = plt.step(range(1,4),cum_exp_var_3, label = "cumulative explained variance")

pca_ev_3 = plt.xlabel("Principal Component Index (3 PCs)")
pca_ev_3 = plt.ylabel("Explained variance ratio (3 PCs)")

pca_ev_3 = plt.legend(loc = "best")
plt.show()


# In[41]:


X_pca_3.shape


# In[46]:


X_pca_3[1]


# In[73]:


# train test split of PCA data

X_train_pca, X_test_pca,y_train_pca,y_test_pca = train_test_split(X_pca_3,y,test_size = 0.15, random_state =42)


# In[74]:


print(X_train_pca.shape,"\n",X_test_pca.shape,"\n",y_train_pca.shape,"\n",y_test_pca.shape)


# In[49]:


# linear regression using all attributes

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)


# In[50]:


# Intercept of lr

lr.intercept_


# In[52]:


# Predicting the target variable

y_train_pred = lr.predict(X_train)
y_test_pred  = lr.predict(X_test)


# In[53]:


# printing samples of predictions on training data

y_train_pred[0:5]


# In[54]:


# printing samples of predictions on test data

y_test_pred[0:5]


# In[55]:


# printing coefficient array of the linear regression

lr.coef_


# In[56]:


# length of the coefficient array

len(lr.coef_)


# In[57]:


# Importing relevant score metrics

from sklearn.metrics import r2_score, mean_squared_error


# In[65]:


# calculating & printing accuracy R^2 and rmse

r2_tr_lr     = r2_score(y_train,y_train_pred)
r2_test_lr   = r2_score(y_test,y_test_pred)

mse_tr_lr    = mean_squared_error(y_train,y_train_pred)
mse_test_lr  = mean_squared_error(y_test,y_test_pred)

rmse_tr_lr   = np.sqrt(mse_tr_lr)
rmse_test_lr = np.sqrt(mse_test_lr)

print("Score metrics of linear regression model with all the attributes","\n\n")

print("R^2 of training set: ",r2_tr_lr,"\n")
print("R^2 of test set: ",r2_test_lr,"\n")

print("rmse of training set: ",rmse_tr_lr,"\n")
print("rmse of test set: ",rmse_test_lr,"\n")


# In[68]:


# SVM regression with all the attributes

from sklearn.svm import SVR

svr = SVR(gamma = "auto")

svr.fit(X_train,y_train)


# In[69]:


# predicting the target variable using svr

y_train_pred_svr = svr.predict(X_train)
y_test_pred_svr  = svr.predict(X_test)


# In[70]:


# printing samples of training predictions

y_train_pred_svr[0:5]


# In[71]:


# printing samples of test predictions

y_test_pred_svr[0:5]


# In[72]:


# Calculating and printing accuracy R^2 and rmse

r2_tr_svm     = r2_score(y_train,y_train_pred_svr)
r2_test_svm   = r2_score(y_test,y_test_pred_svr)

mse_tr_svm    = mean_squared_error(y_train,y_train_pred_svr)
mse_test_svm  = mean_squared_error(y_test,y_test_pred_svr)

rmse_tr_svm   = np.sqrt(mse_tr_svm)
rmse_test_svm = np.sqrt(mse_test_svm)

print("Score metrics of SVM Regressor with all the attributes","\n")

print("Accuracy R^2 Score of training set: ",r2_tr_svm,"\n")
print("Accuracy R^2 Score of test set: ",r2_test_svm,"\n")

print("RMSE of training set: ",rmse_tr_svm,"\n")
print("RMSE of test set: ",rmse_test_svm,"\n")


# In[75]:


# Linear Regression with 3 PCAs

lr_pca = LinearRegression()

lr_pca.fit(X_train_pca,y_train_pca)


# In[76]:


# predicting the target variable

y_train_pred_pca = lr_pca.predict(X_train_pca)
y_test_pred_pca  = lr_pca.predict(X_test_pca)


# In[77]:


# Printing the sample predictions

print(y_train_pred_pca[0:5],"\n",y_test_pred_pca[0:5])


# In[79]:


# printing the coefficient estimates of the model

lr_pca.coef_


# In[80]:


# printing the intercept of the model

lr_pca.intercept_


# In[81]:


# calculating & printing accuracy R^2 and rmse

r2_tr_lr_pca     = r2_score(y_train_pca,y_train_pred_pca)
r2_test_lr_pca   = r2_score(y_test_pca,y_test_pred_pca)

mse_tr_lr_pca    = mean_squared_error(y_train_pca,y_train_pred_pca)
mse_test_lr_pca  = mean_squared_error(y_test_pca,y_test_pred_pca)

rmse_tr_lr_pca   = np.sqrt(mse_tr_lr_pca)
rmse_test_lr_pca = np.sqrt(mse_test_lr_pca)

print("Score metrics of linear regression model with 3 PCAs","\n\n")

print("R^2 of training set: ",r2_tr_lr_pca,"\n")
print("R^2 of test set: ",r2_test_lr_pca,"\n")

print("rmse of training set: ",rmse_tr_lr_pca,"\n")
print("rmse of test set: ",rmse_test_lr_pca,"\n")


# In[83]:


# SVM Regressor with 3 PCAs

svr_pca = SVR(gamma = "auto")

svr_pca.fit(X_train_pca,y_train_pca)


# In[84]:


# predicting the target variable 

y_train_pred_svr_pca = svr_pca.predict(X_train_pca)
y_test_pred_svr_pca  = svr_pca.predict(X_test_pca)


# In[86]:


# printing the prediction samples

print(y_train_pred_svr_pca[0:5],"\n",y_test_pred_svr_pca[0:5])


# In[87]:


# Calculating & printing the Accuracy R^2 Score and RMSE

r2_tr_svr_pca     = r2_score(y_train_pca,y_train_pred_svr_pca)
r2_test_svr_pca   = r2_score(y_test_pca,y_test_pred_svr_pca)

mse_tr_svr_pca    = mean_squared_error(y_train_pca,y_train_pred_svr_pca)
mse_test_svr_pca  = mean_squared_error(y_test_pca,y_test_pred_svr_pca)

rmse_tr_svr_pca   = np.sqrt(mse_tr_svr_pca)
rmse_test_svr_pca = np.sqrt(mse_test_svr_pca)

print("Score metrics of SVR with 3 PCAs: ","\n")

print("R^2 Score of training set: ",r2_tr_svr_pca,"\n")
print("R^2 Score of test set: ",r2_test_svr_pca,"\n")

print("RMSE of training set: ",rmse_tr_svr_pca,"\n")
print("RMSE of test set: ",rmse_test_svr_pca,"\n")


# In[106]:


import time


# In[107]:


# time to fit and predict with linear regression, considering all the attributes

s_lr_t    = time.time()

lr_t      = LinearRegression()

lr_t.fit(X_train,y_train)

y_t_pred  = lr_t.predict(X_train)
y_ts_pred = lr_t.predict(X_test)

e_lr_t    = time.time()

print("lr time with all the attributes: ",(e_lr_t-s_lr_t))


# In[108]:


# time to fit and predict with linear regression, considering 3 PCAs

s_lr_t_pca    = time.time()

lr_t_pca      = LinearRegression()

lr_t_pca.fit(X_train_pca,y_train_pca)

y_t_pred_pca  = lr_t_pca.predict(X_train_pca)
y_ts_pred_pca = lr_t_pca.predict(X_test_pca)

e_lr_t_pca    = time.time()

print("lr time with PCA: ",(e_lr_t_pca-s_lr_t_pca))


# In[109]:


# time to fit and predict SVR with all the attributes

s_svr_t       = time.time()

svr_t         = SVR(gamma = "auto")

svr_t.fit(X_train,y_train)

y_t_pred_svr  = svr_t.predict(X_train)
y_ts_pred_svr = svr_t.predict(X_test)

e_svr_t       = time.time()

print("SVR time with all the atrributes: ",(e_svr_t-s_svr_t))


# In[110]:


# time to fit and predict SVR with 3 PCAs

s_svr_t_pca   = time.time()

svr_t_pca     = SVR(gamma = "auto")

svr_t_pca.fit(X_train_pca,y_train_pca)

y_t_pred_svr_pca = svr_t_pca.predict(X_train_pca)
y_ts_pred_svr_pca = svr_t_pca.predict(X_test_pca)

e_svr_t_pca = time.time()

print("SVR time with 3 PCAs: ",(e_svr_t_pca-s_svr_t_pca))


# In[ ]:


print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is: rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

