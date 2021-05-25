#!/usr/bin/env python
# coding: utf-8

# # Import Python Libraries and Modules

# In[ ]:





# In[62]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np


# In[63]:


# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[64]:


# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


# In[65]:


# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split


# In[66]:


# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[67]:


# Import scikit-Learn module classification report to later use for information about how the system try to classify / lable each record
from sklearn.metrics import classification_report


# ## Load the data - iris.csv
# 

# In[68]:


# Specify location of the dataset
filename = 'C:/Users/sohai/iris.csv'
# Load the data into a Pandas DataFrame
df = pd.read_csv(filename)


# ## Preprocess Dataset - Clean Data: Find & Mark Missing Values

# In[69]:


# mark zero values as missing or NaN
df[[ 'SepalLengthCm' , 'SepalWidthCm' , 'PetalLengthCm ' ,'PetalWidthCm' ]] = df[['SepalLengthCm' , 'SepalWidthCm' ,'PetalLengthCm' , 'PetalWidthCm' ]].replace(0,np.NaN)
# count the number of NaN values in each column
print (df.isnull().sum())


# ## Perform the exploratory data analysis (EDA) on the dataset

# In[70]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[71]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[72]:


#return the first five records / rows of the data set
print(df.head(5))


# In[73]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[74]:


#class distribution i.e. how many records are in each class
print(df.groupby('Species').size())


# ## Plotting Histogram

# In[75]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()


# ## Plotting Density Plots

# In[76]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# ## Box Plots

# In[77]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# ## Scatter Plot Matrix

# In[78]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ## Separate Dataset into Input & Output NumPy arrays

# In[79]:


# store dataframe values into a numpy array
array = df.values
# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 4 (5 - 1)
# these are the independent variables or predictors
X = array[:,1:5]
# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
Y = array[:,5]


# ## Split Input/Output Arrays into Training/Testing Datasets

# In[80]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33
#selection of records to include in each data sub-dataset must be done randomly
seed = 7
#split the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ## Build and Train the Model
# 

# In[81]:


#build the model
model = KNeighborsClassifier()
# train the model using the training sub-dataset
model.fit(X_train, Y_train)
#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# ## Score the accuracy of the model
# 

# In[82]:


#score the accuracy leve
result = model.score(X_test, Y_test)
#print out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# ## Classify/Predict Model
# 

# In[83]:


model.predict([[5.3, 3.0, 4.5, 1.5]])


# ## Evaluate the model using the 10-fold cross-validation technique.

# In[84]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10
# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7
# split the whole dataset into folds
# In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The crossvalidation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once
kfold = KFold(n_splits, random_state=seed)
# for logistic regression, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'

# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print the evaluationm results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# ##### The accuracy of the logistic regression model is 93%

# In[ ]:









# #  Cart Regression with Scikit-Learn

# ## Load Data - Import Python Libraries and Modules

# In[85]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np
# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Import scit-Learn module for the algorithm/model: Linear Regression
from sklearn. tree import DecisionTreeRegressor
# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split
# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ## Load dataset

# In[86]:


# Specify location of the dataset
filename = 'c:/Users/sohai/housing_boston.csv'
# Specify the fields with their names
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
'LSTAT', 'MEDV']
# Load the data into a Pandas DataFrame
df = pd.read_csv(filename, names=names)
# VIP NOTES:
# Extract a sub-dataset from the original one -- > dataframe: df2
df2 = df[['RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'MEDV']]


# ## Pre-process Dataset - Clean Data: Find & Mark Missing Values

# In[87]:


# mark zero values as missing or NaN
df[['RM', 'PTRATIO', 'MEDV']] = df[['RM', 'PTRATIO', 'MEDV']].replace(0, np.NaN)
# count the number of NaN values in each
print(df.isnull().sum())


# ## Perform the Exploratory Data Analysis (EDA) on the dataset

# In[88]:


# Get the dimensions or Shape of the dataset
# i.e. number of records/rows x number of variables/columns
print(df2.shape)


# In[89]:


# Get the data types of all variabLes/attributes of the data set
# The results show
print(df2.dtypes)


# In[90]:


# Get several records/rows at the top of the dataset
# Get the first five records
print(df2.head(5))


# In[91]:


# Get the summary statistics of the numeric variables/attributes of the dataset
print(df2.describe())


# ## Histogram

# In[92]:


# Plot histrogram for each numeric
df2.hist(figsize=(12, 8))
pyplot.show()


# ## Density Plots 

# In[93]:


# Density plots
# IMPORTANT NOTES: 5 numeric variables -> at Least 5 plots -> Layout (2, 3): 2 rows, each row with 3 plots
df2.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# ## Box Plots 

# In[94]:


df2.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# ## Scatter plot matrix

# In[95]:


# scatter plot matrix
scatter_matrix(df2, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ## Separate Dataset into Input & Output Arrays

# In[96]:


# Store dataframe values into a numpy array
array = df2.values
# separate array into input and output components by slicing
# For X (input)[:, 5] --> all the rows, columns from 0 - 4 (5 - 1)
X = array[:,0:5]
# For Y (output)[:, 5] --> all the rows, column index 5 (Last column)
Y = array[:,5]


# ## Split Input/Output Arrays into Training/Testing Datasets
# 

# In[97]:


# Split the dataset --> training sub-dataset: 67%; test sub-dataset:
test_size = 0.33
# Selection of records to include in which sub-dataset must be done randomly
# use this seed for randomization
seed = 7
# Split the dataset (both input & outout) into training/testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ## Build and Train the Model

# In[98]:


# Build the model
model = DecisionTreeRegressor()
# Train the model using the training sub-dataset
model.fit(X_train,Y_train)
# Non-Linear --> NO coefficients and the intercept
DecisionTreeRegressor (criterion='mse', max_depth=None, max_features=None,
max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1,
min_samples_split=2, min_weight_fraction_leaf=0.0 ,presort=False, random_state=None,
splitter='best')


# ## Calculate R Squared

# In[99]:


R_squared = model.score(X_test, Y_test)
print(R_squared)


# ## Prediction

# In[100]:


model.predict([[6.0, 55, 5, 2, 16]])


# ## Evaluate/Validate Algorithm/Model - Using K-Fold Cross-Validation
# 

# In[103]:


# Evaluate the algorithm
# Specify the K-size
num_folds = 10
# Fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7
# Split the whole data set into folds
kfold = KFold(n_splits=num_folds, random_state=seed)
# For Linear regression, we can use MSE (mean squared error) value
# to evaluate the model/algorithm
scoring = 'neg_mean_squared_error'
# Train the model and run K-foLd cross-validation to validate/evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# Print out the evaluation results
# Result: the average of all the results obtained from the k-foLd cross-validation
print(results.mean())


# In[ ]:





# In[ ]:




