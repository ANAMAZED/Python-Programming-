#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Unsupervised Learning - KMeans

# ## Import Python Libraries and Modules 

# In[4]:


# Import Python Libraries: NumPy and Pandas 
import pandas as pd 
import numpy as np

# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix 
from matplotlib import pyplot


# Import scikit-Learn module for the algorithm/modeL: K-Means
from sklearn.cluster import KMeans


# ## Load data-set (iris.csv)

# In[5]:


# Specify location of the dataset
filename = 'iris.csv'
 
# Load the data into a Pandas DataFrame 
df = pd.read_csv(filename)


# ## Pre-process Dataset (Clean Data)

# In[10]:


# mark zero values as missing or NaN
df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]].replace(0,np.NaN) 
 
# count the number of NaN values in each column 
print(df.isnull().sum()) 


# ## Perform the exploratory data analysis (EDA) on the dataset
#  

# In[11]:


# to get the dimension of the dataset, no. of rows x no. of columns 
print(df.shape)


# In[12]:


# to get data types of all variables 
print(df.dtypes)


# In[13]:


#display first 5 rows 
print(df.head(5))


# In[15]:


#display summary statistics
print(df.describe())


# In[16]:


#class distribution - how many records in each class 
print(df.groupby('Species').size())


# ## Histogram

# In[17]:


#plot histogram of each numeric variable / attribute in the data set 
df.hist(figsize=(12, 8))
pyplot.show()


# ## Density Plots

# In[18]:


# generate density plots of each numeric variable / attribute in the data set 
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1, figsize=(12, 16))
pyplot.show()


# ## Box Plots

# In[19]:


# generate box plots of each numeric variable / attribute in the data set 
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# ## Scatter Plots

# In[20]:


# generate scatter plot matrix of each numeric variable / attribute in the data set 
scatter_matrix(df, alpha=0.8, figsize=(15, 15)) 
pyplot.show()


# ## Separate Dataset into Input & Output NumPy arrays

# In[21]:


# store dataframe values into a numpy array 
array = df.values
 
# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 4 (5 - 1)
# these are the independent variables or predictors 
# we will only use this going forward
X = array[:,1:5]
 
# for Y(input) [:, 5] --> all the rows, column 5 
# this is the value we are trying to predict 
# we wont use this going forward 
Y = array[:,5]


# ## Build and Train the Model

# In[24]:


# Build the model 
# set cluster (K) to 3 to start 
model = KMeans(n_clusters=3)
 
# defaults 
KMeans(algorithm='auto', copy_x=True, init= 'k-means++', max_iter=300,
n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto' ,
random_state=None, tol=0.0001, verbose=0) 
 
 
# Use the model to cluster the inputdata 
model.fit (X) 
 
centroids = model.cluster_centers_
print(centroids)


# In[27]:


cluster_labels = model.labels_[::10] 
print (cluster_labels) 
# this will print 10 records 


# In[28]:


cluster_labels = model.labels_ 
print (cluster_labels) 
# this will print all records - shows clustering of values in an unlabeled set


# In[29]:


pyplot.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap= 'rainbow' ) 
 
pyplot.show ( )


# In[31]:


# plot the data points with centroids
# plot using first and second variables of the vector 
pyplot.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap= 'rainbow') 
 
lines = pyplot.plot(centroids[0,0],centroids[0,1], 'kx', color= 'black')
pyplot.setp (lines, ms=15.0)
pyplot.setp(lines, mew=2.0)
 
lines = pyplot.plot(centroids[1,0],centroids[1,1], 'kx', color= 'black')
pyplot.setp (lines, ms=15.0) 
pyplot.setp(lines, mew=2.0)
 
lines = pyplot.plot(centroids[2,0],centroids[2,1], 'kx', color= 'black') 
pyplot.setp (lines, ms=15.0) 
pyplot.setp(lines, mew=2.0)
 
pyplot.show ()


# ## Classify/Predict Model 

# In[32]:


model.predict([[5.3, 3.0, 4.5, 1.5]])


# In[ ]:





# In[ ]:




