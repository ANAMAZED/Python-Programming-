#!/usr/bin/env python
# coding: utf-8

# # Python Data Visualization:Matplotlib

# In[ ]:


# Matplotlib is a 2D plotting library in Python. It has two interfaces for users to develop plots with : Stateful and stateless 


# ## Stateful Visualization

# In[95]:


#Visualizaing data with matplotlib - stateful approach with python
import matplotlib.pyplot as plt

x=[-3,5,7]
y=[10,2,5]

plt.figure(figsize=(15,3))
plt.plot(x,y)
plt.xlim(0,10)
plt.ylim(-3,8)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line Plot')

plt.show()


# In[ ]:


# Description :
# In the above code we are plotting a figure using the stateful approach, we have deifined x axis and y axis .
# A line plot is created with limits for both axis defined 


# ## Stateless Visualization
# ### Visualization with class Axes: Stateless (OO Approach)

# In[22]:


import matplotlib.pyplot as plt

x=[-3,5,7]
y=[10,2,5]

fig, ax = plt.subplots(figsize=(15,3))
ax.plot(x,y)
ax.set_xlim(0,10)
ax.set_ylim(-3,8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')

plt.show()


# In[ ]:


# Description 
# In the above code we are visualizing a figure using stateless approach with class axes 
# ax refers to axes
# we have set limits for x and y both
# ax.plot will plot the data given to it 
#xlim, ylim are used to adjust the limits of both axes 


# ## Import and load Dataset 

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/sohai/Iris.csv')
df.head()


# In[ ]:


# Description: above mentioned code 
# Fist all desired libraries are imported 
# the dataset was loaded into the pandas dataframe 
#  we then read the csv file 
# df.head() This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.


# ## Data visualization with Pandas and Matplotlib

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/sohai/Iris.csv')
df.hist(figsize=(12,8))
plt.show()


# In[ ]:


# Description
# In the above mentioned code we are creating histograms of the data provided to us
# Histogram-The purpose of a histogram (Chambers) is to graphically summarize the distribution of a univariate data set.
# we have provided what file we want to use for dataset by providing teh location and read command 
# a representation of distribution of data is created in the form of a histogram 
# histogram for each numerical value is created 


# ## Density Plots 

# In[26]:


df.plot(kind='density', subplots=True, 
        layout=(2,3),sharex=False, legend=True, fontsize=1, 
        figsize=(12,6))


plt.show()


# In[ ]:


# Description
# Density plot for the data set is created
# It takes numerical values as input and plot it. we have created plots for the different species of plant by showing lengths and widths


# ## Boxplots (Box & Whisker Plots)

# In[15]:


df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(12,8))
plt.show()


# In[ ]:


# Description
# Now for the same dataset we have created the boxplot 
# Box Plots are a measure of how well distributed the data in a data set is. It divides the data set into three quartiles. This graph represents the minimum, maximum, median, first quartile and third quartile in the data set.


# ## MultiVariate Data Visualization
# ## Scatter Matrix Plot

# In[96]:


scatter_matrix(df, alpha=0.8, figsize=(9,9))
plt.show()


# In[ ]:


# Description:
# A scatter plot is created which includes the plots of all pairs of attributes 


# # Machine Learning and Supervised Linear Regression

# ## Load Data - Import Python Libraries and Module

# In[97]:


# Import Python Libraries: Numpy and Pandas 
import pandas as pd
import numpy as py

# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Import scikit-Learn module for the algorithm/modeL: Linear Regression
from sklearn.linear_model import LinearRegression
# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split
# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Specify location of the dataset
filename = 'C:/Users/sohai/housing_boston.csv'
# Specify the fields with their names
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# Load the data into a Pandas DataFrame
df = pd.read_csv(filename, names=names)
# VIP NOTES:
# Extract a sub-dataset from the original one -- > dataframe: df2
df2 = df[['RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'MEDV']]





# In[ ]:


# Description 
# First we import Python libraries 
# Then we load the dataset by proviidng which file to use by providing a path of that file 
# we then specify the fields to use 
# load the data into the pandas dataframe 
# we then define a second dataframe and give it values 


# ## Pre-process Dataset
# ## Clean Data: Find & Mark Missing Values

# In[98]:


# mark zero values as missing or NaN
df[['RM', 'PTRATIO', 'MEDV']] = df[['RM', 'PTRATIO', 'MEDV']].replace(0, np.NaN)
# count the number of NaN values in each
print(df.isnull().sum())


# In[ ]:


# Description
# Here we are clening the data as it can not have 0 values 
# in each column if there is an invalid value we mark it as missig value of NaN - not a number 


# ## Perform the Exploratory Data Analysis (EDA) on the dataset
# 

# In[99]:


# Get the dimensions or Shape of the dataset
# i.e. number of records/rows x number of variables/columns
print(df2.shape)

#DESCRIPTION
# we will be getting the shape of data set which will tell us the exact dimensions e.g. rows by columns and printing it 


# In[100]:


# Get the data types of all variabLes/attributes of the data set
# The results show
print(df2.dtypes)

#DESCRIPTION
# here we are printing the type of the variables in the dataset e.g. if it is an integer or float (has decimals)


# In[101]:


# Get several records/rows at the top of the dataset
# Get the first five records
print(df2.head(5))

#DESCRIPTION
#here we have specified that we need to display data from the first 5 rows therefore starting from 0 to 4 


# In[102]:


# Get the summary statistics of the numeric variables/attributes of the dataset
print(df2.describe())

# DESCRIPTION
# here we are printing the summary statistics of the data provided to us, which includes the mean, median, std dwviation, min ans maximum values 


# In[103]:


# Plot histrogram for each numeric
df2.hist(figsize=(12, 8))
pyplot.show()

# DESCRIPTION 
# here we are plotting the histogram for each numeric e.g.RM, RAD, MEDV etc 


# In[104]:


# Density plots
# IMPORTANT NOTES: 5 numeric variables -> at Least 5 plots -> Layout (2, 3): 2 rows, each row with 3 plots
df2.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()

#Description
# this is the density plot for the same dataset  2,3 means 2 rows and each row with  plots 


# In[46]:


df2.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# Description
# bos and whiskers plot is now created which shows the quartile range and outliers 


# In[105]:


# scatter plot matrix
scatter_matrix(df2, alpha=0.8, figsize=(15, 15))
pyplot.show()

#A scatter plot is created 


# ## Separate Dataset into Input & Output NumPy Arrays
# 

# In[48]:


# Store dataframe values into a numpy array
array = df2.values
# separate array into input and output components by slicing
# For X (input)[:, 5] --> all the rows, columns from 0 - 4 (5 - 1)
X = array[:,0:5]
# For Y (output)[:, 5] --> all the rows, column index 5 (Last column)
Y = array[:,5]


# Description
# Here we are storing the dataset in the multidimensional array 
# y here is a dependent variable 


# ## Split Input/Output Arrays into Training/Testing Datasets
# 

# In[107]:


# Split the dataset --> training sub-dataset: 67%; test sub-dataset:
test_size = 0.33
# Selection of records to include in which sub-dataset must be done randomly
# use this seed for randomization
seed = 7
# Split the dataset (both input & outout) into training/testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)

# here we are splitting dataset, 1/3rd of the data will be used for test ad the rest will be used for training 


# ## Build and Train the Model
# 

# In[52]:


# Build the model
model = LinearRegression()
# Train the model using the training sub-dataset
model.fit(X_train, Y_train)
# Print out the coefficients and the intercept
# print intercept and coefficients
print (model.intercept_)
print (model.coef_)


#Description
# here we are using the function linear regression to print the coefficients and intercept


# In[62]:


# If we want to print out the list of the coefficients with their correspondent variable name
# pair the feature names with the coefficients
names_2 = ['RM', 'AGE','DIS','RAD', 'PTRATIO']
coeffs_zip = zip(names_2, model.coef_)
# Convert iterator into set
coeffs = set(coeffs_zip)
#print (coeffs)
for coef in coeffs:
    print(coef, "\n")
    
#Description: Here we have listed a variable names_2 and we are printing the coefficients 
# zip() The function takes in iterables as arguments and returns an iterator. 
# set() set() method is used to convert any of the iterable to the distinct element and sorted sequence of iterable elements, commonly called Set


# ## Calculate R-Squared

# In[109]:


R_squared = model.score(X_test, Y_test)
print(R_squared)

#Desciption 
# Here we are calculating the value of R square value ; R-squared is a statistical measure of how close the data are to the fitted regression line
# the higher the Rsquare, the better it is 
# best models score above 83% - this tells us how good the independent cariables predict the dependent


# ## Prediction 

# In[57]:


model.predict([[6.0, 55, 5, 2, 16]])

# The suburb area has the following predictors:
# RM: average number of rooms per dwelling = 6.0
# AGE: proportion of owner-occupied units built prior to 1940 = 55
# DIS: weighted distances to five Boston employment centers = 5
# RAD: index of accessibility to radial highways = 2
# PTRATIO: pupil-teacher ratio by town = 16
# the model predict that the median value of owner-occupied homes in 1000 dollars in the above suburb should be around 23,000.


# ## Evaluate/Validate Algorithm/Model • Using K-Fold Cross-Validation
# 

# In[59]:


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



# Description:
# afte rwe train, we do the evaluation
# Use K-Fold to determine if the model is acceptable
# We pass the whole set because the system will divide for us -31 avg of all error (mean of square errors) this value would traditionally be positive value, but scikit reports as neg
# Square root would be between 5 and 6


#  # Machine Learning: Supervised – Logistic Regression
#  

# ## Import Python Libraries and Modules 

# In[8]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np


# In[9]:


# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[10]:


# Import scikit-Learn module for the algorithm/modeL: Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[11]:


# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split


# In[12]:


# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[13]:


# Import scikit-Learn module classification report to later use for information about how the system try to classify / lable each record
from sklearn.metrics import classification_report


# ## Load the data - Dataset: Iris.scv

# In[14]:


# Specify location of the dataset
filename = 'C:/Users/sohai/iris.csv'
# Load the data into a Pandas DataFrame
df = pd.read_csv(filename)

# Description: we are uploading the iris.csv file by providing file path and the reading the file. A datafram using pandas is created 


# In[15]:


## Preprocess Dataset - Clean Data: Find & Mark Missing Values


# In[16]:


# mark zero values as missing or NaN
df[[ 'SepalLengthCm' , 'SepalWidthCm' , 'PetalLengthCm ' ,'PetalWidthCm' ]] = df[['SepalLengthCm' , 'SepalWidthCm' ,'PetalLengthCm' , 'PetalWidthCm' ]].replace(0,np.NaN)
# count the number of NaN values in each column
print (df.isnull().sum())

# Description: data cleaning - we look at all values and any invalid value is marked as not a number or zero 



# ## Perform the exploratory data analysis (EDA) on the dataset
# 

# In[17]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)

# here the shape is being defined 


# In[18]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)

#Description - the data type of variables is printed 


# In[19]:


#return the first five records / rows of the data set
print(df.head(5))

#DEscription : first 5 rows are displayed 


# In[20]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


#Description: calculating summary statistics


# In[21]:


#class distribution i.e. how many records are in each class

print(df.groupby('Species').size())

#Description - here we are grouping by type species and showing the number of record in each group, also mentioning that data type is integer 


# In[22]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()

# Description - plotting histogram of the variables in the data set e.g. petal length, petal width etc.


# In[23]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()

# Description - plotting the density plot for the same variables as above 


# In[24]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()

# Description - plotting the box plot for the variables - it shows the outliers and quartile range 


# In[84]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()



# Description - Scatter plot is also constructed for the data set. here we have specified the size as well


# ## Separate Dataset into Input & Output NumPy arrays
# 

# In[85]:


# store dataframe values into a numpy array
array = df.values
# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 5 (6 - 1)
# these are the independent variables or predictors
X = array[:,1:5]
# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
Y = array[:,5]

# Description - Here we are seperating datset as trainign data (X) consists of independent variables
# Also beacuse the desired output which is (Y) consists of dependent variable or the ones we are trying to predict 


# ## Split Input/Output Arrays into Training/Testing Datasets

# In[86]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33
#selection of records to include in each data sub-dataset must be done randomly
seed = 7
#split the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ## Build and Train the Model

# In[87]:


#build the model
model = LogisticRegression()
# train the model using the training sub-dataset
model.fit(X_train, Y_train)
#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)

# Description - The code below build the Model, Train the model using the training sub-dataset, Print out the coefficients and the intercept, Print intercept and coefficients, if we want to print out the list of coefficients with their correspondent variable name, pair the feature names with the coefficients and convert iterator in to set
# The precision is the ratio tp / (tp + fp) --> where tp is the number of true positives and fp the number of false positives.
# The precision represents the ability of the classifier not to label a positive sample as negative
# The recall is the ratio tp / (tp + fn) --> where tp is the number of true positives and fn the number of false negatives
# The recall represents the ability of the classifier to find all the positive samples
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall --> where an F-beta score reaches its best value at 1 and worst score at 0 The F-beta score weights recall more than precision by a factor of beta beta == 1.0 means recall and precision are equally important
# The support is the number of occurrences of each class in y_true


# ## Score the accuracy of the model
# 

# In[88]:


#score the accuracy leve
result = model.score(X_test, Y_test)
#print out the results
print(("Accuracy: %.3f%%") % (result*100.0))

# Description: here we are calculating the accuracy 


# ## Classify/Predict Model
# 

# In[89]:


model.predict([[5.3, 3.0, 4.5, 1.5]])

#Description - we are predicting the flower type here 


# ## Evaluate the model using the 10-fold cross-validation technique.
# 

# In[92]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10
# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7
# split the whole dataset into folds
# In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The crossvalidation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once kfold = KFold(n_splits, random_state=seed)
# for logistic regression, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'

# Description:
#specify the number of time of repeated splitting, Fix the random seed, Must use the same see value so that the same subset can be obtained for each time the process is repeated, Split the whole data set into folds, for logistic regression, we can use accuracy level to evaluate the model/algorithm, train the model and run k-fold cross-validation to validate/evaluate the model, Print out the evaluation results


# In[93]:


# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print the evaluationm results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

##using the 10-fold cross-validation to evaluate the model / algorithm, the accuracy of this logistic regression model is 94.7%
# There is 94.7% chance that this new record is an Iris-versicolor


# In[ ]:




