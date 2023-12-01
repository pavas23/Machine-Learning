#!/usr/bin/env python
# coding: utf-8

# # ```CS F320 - FOUNDATIONS OF DATA SCIENCE```
# 
# ## ```ASSIGNMENT 2B - PCA Analysis and Determining Optimal Number of Components```
# 
# ### ```TEAM MEMBERS: ```
#         1. Pavas Garg - 2021A7PS2587H
#         2. Tushar Raghani - 2021A7PS1404H
#         3. Rohan Pothireddy - 2021A7PS0365H 
#     

# # ```Importing the Libraries```
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random


# ## `⏳ Loading the Dataset`
# 

# In[2]:


df = pd.read_csv("Hitters.csv")
df


# In[3]:


# checking categorical variables
df.select_dtypes(exclude=['number'])


# ## ```Dropping Categorical Variables```

# In[4]:


# Drop all categorical columns
df = df.select_dtypes(exclude='object')
df


# ## ```1. Data Understanding and Representation```

# In[5]:


print("Number of records in the given dataset are: ",len(df))
print("Number of features in the given dataset are: ",len(df.columns)-1)


# # ```🔬Preprocess and perform exploratory data analysis of the dataset obtained```

# ## `Replacing NULL values with mean`

# In[6]:


# to check if null values or NAN are present in the dataset
nan_count = df.isna().sum().sum()
null_count = df.isnull().sum().sum()
print("NAN count: ",nan_count)
print("NULL count: ",null_count)


# In[7]:


def find_mean(dataset, feature):
    n = len(dataset[feature])
    total_sum = 0
    count_valid_values = 0

    for val in dataset[feature]:
        if isinstance(val, (int, float)) and not np.isnan(val):
            total_sum += val
            count_valid_values += 1

    if count_valid_values == 0:
        return 0

    mean = total_sum / count_valid_values
    return mean


# In[8]:


for feature in df.columns:
    mean = find_mean(df, feature)
    print(feature)
    print(mean)


# In[9]:


pd.set_option('mode.chained_assignment', None)


# In[10]:


for feature in df.columns:
    mean = find_mean(df,feature)
    df[feature].fillna(mean,inplace=True)


# In[11]:


nan_count = df.isna().sum().sum()
null_count = df.isnull().sum().sum()
print("NAN count: ",nan_count)
print("NULL count: ",null_count)


# # `📊 Plotting Histograms`

# In[12]:


df.hist(bins=50,figsize=(15,15))
plt.show()


# In[13]:


# shifting the target attribute to the right most column of the dataset
column_to_shift = 'Salary'
shifted_column = df.pop(column_to_shift)
df[column_to_shift] = shifted_column


# In[14]:


df


# ## ```Feature Scaling```

# ### Normalization Method
# - Normalization is performed to transform the data to have a mean of 0 and standard deviation of 1
# - Normalization is also known as Z-Score Normalization
# 
# \begin{equation}
# z = \frac{(x-\mu)}{\sigma}
# \end{equation}

# In[15]:


# function for finding mean of a feature in a given dataset
def find_mean(dataset,feature):
    n = len(dataset[feature])
    sum = 0
    for val in dataset[feature]:
        sum += val
    return sum/n


# In[16]:


# function for finding standard deviation of a feature in a given dataset
def find_standard_deviation(dataset,feature):
    variance, squared_sum = 0,0
    n = len(dataset[feature])
    mean = find_mean(dataset,feature)
    for val in dataset[feature]:
        squared_sum += (val-mean)**2
    variance = squared_sum/n
    return math.sqrt(variance)


# In[17]:


# function for scaling a feature in given dataset
def normalize_feature(dataset,feature):
    mean = find_mean(dataset,feature)
    standard_deviation = find_standard_deviation(dataset,feature)
    normalized_feature = []
    for val in dataset[feature]:
        normalized_feature.append((val-mean)/standard_deviation)
    return normalized_feature


# In[18]:


# function for scaling (standardizing) the whole dataset
def normalize_dataset(dataset):
    df = dataset.drop(columns = 'Salary')
    normalized_df = pd.DataFrame()
    for feature in df.columns:
        normalized_result = normalize_feature(df,feature)
        normalized_df[feature] = normalized_result
        
# When copying columns from one DataFrame to another, you might get NaN values in the resulting DataFrame.
# The issue is caused because the indexes of the DataFrames are different.
# This causes the indexes for each column to be different.
# When pandas tries to align the indexes when assigning columns to the second DataFrame, it fails and inserts NaN values.
# One way to resolve the issue is to homogenize the index values.
# for eg [a,b,c,d] for df1 and indices for df2 are [1,2,3,4]
# that's why use df1.index = df2.index

    normalized_df.index = dataset.index 
    normalized_df['Salary'] = dataset['Salary']
    return normalized_df


# In[19]:


# normalizing the complete dataset
df = normalize_dataset(df)
df.head()


# In[20]:


# checking mean and variance of each feature after standardizing the dataset
dataset = df.drop(columns = 'Salary')
for feature in dataset:
    print("Mean of",feature,"is",round(find_mean(dataset,feature)))
    print("Standard Deviation of",feature,"is",round(find_standard_deviation(dataset,feature)))


# ## `Plot showing distribution of features after Normalization`

# In[21]:


# all features following a normal distribution with mean 0 and standard deviation of 1
sns.displot(df.drop(columns='Salary'), kind='kde',aspect=1,height=8)
plt.show()


# ## ```Dimensionality Reduction using PCA (Principal Component Analysis)```
# 
# - It is used to reduce the dimensionality of dataset by transforming a large set into a lower dimensional set that still contains most of the information of the large dataset
# 
# - Principal component analysis (PCA) is a technique that transforms a dataset of many features into principal components that "summarize" the variance that underlies the data
# 
# - PCA finds a new set of dimensions such that all dimensions are orthogonal and hence linearly independent and ranked according to variance of data along them
# 
# - Eigen vectors point in direction of maximum variance among data, and eigen value gives the importance of that eigen vector
# 
# ![PCA_Image](https://miro.medium.com/v2/resize:fit:1192/format:webp/1*QinDfRawRskupf4mU5bYSA.png)

# - Let $x_1$, $x_2$... $x_n$ be be N training examples, each having D features.
# 
# 
# - Mean of N training examples is given by $\bar{x}$, which can be computed as
# 
# $$
# \bar{x} = \frac{1}{N} \sum_{n=1}^{N} x_{n1} + \frac{1}{N} \sum_{n=1}^{N} x_{n2}... + \frac{1}{N} \sum_{n=1}^{N} x_{nd}
# $$
# 
# - Now suppose we have a graph with 2-dimensional points as follows:
# ![2D_points](https://i.stack.imgur.com/ym6ru.png)
# 
# - Our motive is to bring down the 2D points to 1D by projecting on a vector. We need to project the points on a 1D vector such that the variance between data points is maximum.
# 
# 
# - We need to compute unit vector such that the variance is as maximum as possible. 
# 
# 
# - We do some mathematical computations as follows:
# 
# $$
# cos\theta = \frac{OA}{OB}
# $$
# 
# $$
# \bar{u}cdot\bar{x_n} = (||u||)(||x||)cos\theta
#                      = (||u||)(OB)cos\theta = (||u||)(OA)
# $$
# 
# - The above equation gives us the below result
# 
# $$
# OA = \frac{\bar{u} \cdot \bar{x_n}}{\|u\|}
# $$
# 
# - We take projection on unit vector $||u||$ = 1
# 
# - Our final result is as follows:
# 
# $$
# OA = \bar{u}\cdot\bar{x_n}
# $$
# 
# - The mean of the projected points is given by 
# 
# $$
# \frac{1}{N}\sum_{n=1}^{N}{\bar{u}\cdot\bar{x_n}} = \bar{u}\cdot\sum_{n=1}^N\frac{x_n}{N} = \bar{u}\cdot\bar{x}
# $$
# 
# - Here, $\bar{x}$ is the mean of training points in their dimension
# 
# - We then compute variance as 
# 
# $$
# Variance = \frac{1}{N}\sum_{n=1}^N{(\bar{u}\cdot\bar{x_n} - \bar{u}\cdot\bar{x})^2}
# $$
# 
# - We then compute $\bar{u}$ which maximizes variance as much as possible such that $||u|| = 1$
# 
# - Consider $x_n$ and $\bar{x}$ to be matrices of $d$x$1$ size represented as follows
# 
# $$
# \bar{x_n}=\begin{bmatrix}
#   x_1 \\
#   x_2 \\
#   x_3 \\
#   \vdots \\
#   x_d \\
# \end{bmatrix}
# $$
# 
# 
# $$
# \bar{x}=\begin{bmatrix}
#   \bar{x_1} \\
#   \bar{x_2} \\
#   \bar{x_3} \\
#   \vdots \\
#   \bar{x_d} \\
# \end{bmatrix}
# $$
# 
# - Consider $\bar{u}$ to be a $1$x$d$ matrix represented by
# 
# $$
# \bar{u} = [{u_1}, {u_2}...{u_d}]
# $$
# 
# - We then observe that we need to maximize the following expression
# 
# $$
# max[\frac{1}{N}\sum_{n=1}^{N}(\bar{u}\cdot(\bar{x_n}-\bar{x}))(\bar{u}\cdot(\bar{x_n}-\bar{x})^{T}]
# $$
# 
# - While trying to maximize the above expression by expanding the same, we get
# 
# $$
# max[\frac{1}{N}\sum_{n=1}^{N}(\bar{u}(x_n - \bar{x})(x_n - \bar{x})^{T}\bar{u}^{T})]
# $$
# 
# - The above expression in turn becomes
# 
# $$
# max[\bar{u}\frac{1}{N}[\sum_{n=1}^{N}(x_n - \bar{x})(x_n - \bar{x})^{T}]\bar{u}^{T}]
# $$
# 
# - The above expression simplifies to 
# 
# $$
# max[\bar{u}S\bar{u}^T] 
# $$
# 
# $$
# ||u||=1
# $$
# - Here, $S$ is called covariance matrix 
# 
# 
# - Principal Component Analysis (PCA) gives linear combination of these features to get matured features
# 
# 
# - We then try to convert the above constraint optimization problem to an unconstrained optimization problem, as follows:
# 
# $$
# E(u,\lambda) = max[\bar{u}S\bar{u}^{T} + \frac{\lambda}{2}(1-\bar{u}\bar{u}^{T})]
# $$
# 
# - Taking derivation with respect to $\bar{u}$ and $\lambda$ and setting it to 0, we get final answer to be 
# 
# $$
# \bar{u}S\bar{u}^T = \lambda
# $$
# 
# - $\lambda$ is called the eigen value found from the equation
# 
# $$
# |A - \lambda{I}| = 0
# $$
# 
# - Let $u_1$, $u_2$,...$u_d$ be the eigen vectors, and $\lambda_1$, $\lambda_2$,...$\lambda_d$ be the eigen values, $A$ is a $d$x$d$ square matrix, we get 
# 
# $$
# A\gamma = \lambda\gamma
# $$
# 
# $$
# Au_1 = \lambda_1u_1
# $$
# 
# $$
# Au_2 = \lambda_2u_2
# $$
# 
# - Any of d $\bar{u}$ values are feasible solutions, we need to find optimal solution from the following set of equations
# 
# $$
# Su_1 = \lambda_1u_1 
# $$
# $$
# Su_2 = \lambda_2u_2 
# $$
# 
# $$
# .
# $$
# 
# $$
# .
# $$
# 
# $$
# .
# $$
# 
# $$
# Su_d = \lambda_du_d 
# $$
# 
# - The above set of equations simplifies to
# 
# $$
# u_1Su_1^{T} = \lambda_1
# $$
# 
# $$
# u_2Su_2^{T} = \lambda_2
# $$
# 
# $$
# .
# $$
# 
# $$
# .
# $$
# 
# $$
# .
# $$
# 
# $$
# u_dSu_d^{T} = \lambda_d
# $$
# 
# - For instance, if we project all points on eigen vector $u_1$ then variance comes out to be $\lambda_1$
# 
# 
# - $\lambda_1$, $\lambda_2$, ...., $\lambda_d$ are variances after projecting values/points on eigen vectors $u_1$, $u_2$,....,$u_d$. We need to find that eigen vector which has maximum variance, or simply, maximum $\lambda$.
# 
# 
# - For instance, consider the first eigen vector to be of the form 
# 
# $$
# u_1 = \begin{bmatrix}
#   \bar{u_{11}} \\
#   \bar{u_{12}} \\
#   \bar{u_{13}} \\
#   \vdots \\
#   \bar{u_{1d}} \\
# \end{bmatrix} 
# $$
# 
# - Transformed point is 
# 
# $$
# u_{11}x_{11} + u_{12}x_{12}+... + u_{1d}x_{1d}
# $$
# 
# - Transformation of a point from multidimensional space (d-dimensional in this case) to a uni-dimensional space is a linear transformation (where multiples are componenents of eigen vectors in PCA)

# In[22]:


# implementing PCA from scratch

# it will take dataset X and k components needed after PCA
def PCA(X,k):
    k_principal_components = [] # it will store first k eigen vectors
    mean = np.mean(X,axis=0)  # this will find mean for each row
    X = X - mean  # mean centering the data
    
    # finding the covariance matrix, will give a n*n matrix containing covariance of all features
    cov = np.cov(X.T)
    
    # finding eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # transpose eigenvector
    eigenvectors = eigenvectors.T
    
    # will give indexes according to eigen values, sorted in decreasing order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx]
    
    # for finding how much variance does each principal component capture
    explained_variance = eigenvalues / np.sum(eigenvalues)
    
    # slicing first k eigenvectors
    k_principal_components = eigenvectors[:k]
     
    # returning the transformed features
    # multiplyinh n*d matrix with d*k matrix to get transformed feature matrix of dimension n*k
    return np.dot(X,k_principal_components.T), explained_variance, k_principal_components


# In[23]:


k = 16
# this will return the new dataset
df_pca, explained_variance, k_principal_components = PCA(df.drop(columns="Salary"),k)
df_pca = pd.DataFrame(df_pca)
df_pca['Salary'] = df['Salary']
print("Shape of dataset is:", df.shape)
print("Shape of dataset after pca is:",df_pca.shape)


# In[24]:


df_pca


# ## ```Plot showing variance captured by each Principal Component```

# In[25]:


num_components = len(explained_variance)
components = np.arange(1, num_components + 1)
plt.figure(figsize=(10, 8))
plt.plot(components, explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(components)
plt.show()


# ## ```Plot to find out number of Principal Components needed inorder to capture 95% variance in data```

# In[26]:


# finding cumulative variance captured by principal components
y_var = np.cumsum(explained_variance)

plt.figure(figsize=(12,10))
plt.ylim(0.0,1.1)
plt.plot(components, y_var, marker='o', linestyle='-', color='green')

plt.xlabel('Number of Components',fontsize=16)
plt.ylabel('Cumulative variance (%)',fontsize=16)
plt.title('The number of components needed to capture 95% variance',fontsize=18)
plt.xticks(components) 

plt.axhline(y=0.95, color='red', linestyle='--')
plt.axhline(y=1.00, color='orange', linestyle='--')
plt.axvline(x=6.00, color='blue', linestyle='--')
plt.text(1, 0.95, '95% cut-off threshold', color = 'black', fontsize=16)
plt.text(1, 1, '100% variance if all features are included', color = 'black', fontsize=16)

plt.tight_layout()
plt.show()


# - ```The number of principal components required for efficient prediction might be 6 because we see that the first 6 principal components capture 95% of the complete variance of the data```
# 
# 
# - ```We will try a range of components on our regression model to find to determine most efficient number of principal components.```

# ## ```Model Training and MSE/RMSE Calculation```

# ## `Train-Test Split`

# In[27]:


def split_train_test(data,test_ratio):
    # np.random.seed() is very important as whenever we call the function it will randomly divide the indices
    # it might happen after many calls our model sees all the data and it leads to overfitting so to prevent it
    # seed function will randomly divide data only once and once the function is called it will not give other
    # permuatation of indices whenever called again,hence no overfitting
    np.random.seed(45)
    # it will give random permutation of indices from 0 to len(data)-1
    # now shuffled array will contain random number for eg [0,4,1,99,12,3...]
    shuffled = np.random.permutation(len(data))  
    test_set_size = int(len(data)*test_ratio)
    # it will give array of indices from index 0 to test_set_size-1
    test_indices = shuffled[:test_set_size]
    # it will give array of indices from index test_set_size till last
    train_indices = shuffled[test_set_size:]
    # it will return rows from data df corresponding to indices given in train and test indices array
    # so it is returning the train and test data respectively
    return data.iloc[train_indices], data.iloc[test_indices]


# In[28]:


train_set, test_set = split_train_test(df_pca,0.2)


# In[29]:


train_set.shape


# In[30]:


x_train = train_set.drop(columns = 'Salary')
x_test = test_set.drop(columns = 'Salary')
x_train.insert(0,'Ones',1)
x_test.insert(0,'Ones',1)
x_train.columns = range(len(x_train.columns))
x_test.columns = range(len(x_test.columns))
y_train = train_set['Salary']
y_test = test_set['Salary']


# In[31]:


x_train.head()


# In[32]:


x_test.head()


# ## `Gradient Descent Algorithm`
# 
# - We will use this equation to update our linear regression model parameters
# 
# $$
# \begin{equation}
# \theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
# \end{equation}
# $$
# 
# $$
# \begin{equation}
# \frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)}, \quad h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1}  + \theta_{2}x_{2}  +  ... +  \theta_{d}x_{d}
# \end{equation}
# $$
# 
# - Repeat until convergence
# $$
# \begin{equation}
# \theta_{j} = \theta_{j} - {\alpha}\sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)} ,\quad\text {$0 \leq j \leq d$}
# \end{equation}
# $$
# 
# - Such that it minimizes the cost function given by equation
# 
# $$
# \begin{equation}
# J(\theta) = {\frac{1}{2}}\sum_{i=1}^n{(h_{\theta}(x)^{(i)} - y^{(i)})^2}
# \end{equation}
# $$

# In[33]:


def give_train_data(x_train,x_test,k):
    first_k_x_train = x_train.iloc[:,:k+1]
    first_k_x_test = x_test.iloc[:,:k+1]
    return first_k_x_train,first_k_x_test


# In[34]:


def give_weight_vector(k):
    weight_vector = np.zeros(k+1)
    return weight_vector


# In[35]:


# function to find cost value, using the formula for J(theta)
def find_cost(y_actual,y_predicted):
    cost = 0
    for i in range(len(y_actual)):
        cost += (y_predicted[i] - y_actual[i])**2
    return (1/2)*cost


# In[36]:


def print_cost_function(iteration_x_axis_batch,cost_y_axis_batch):
    plt.plot(iteration_x_axis_batch,cost_y_axis_batch)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Value")
    plt.show()


# In[37]:


max_iterations = 10000


# In[38]:


def batch_gradient_descent(x_train,y_train,x_test,k,learning_rate,iteration_x_axis_batch,cost_y_axis_batch):
    prev_cost = 0
    
    # get x_train and y_train vectors
    x_train_batch, x_test_batch = give_train_data(x_train,x_test,k)
    
    # get the weight vector with degree weights
    weight_vector = give_weight_vector(k)
    
    for iteration in range(max_iterations):
        # will give the predicted value, after each iteration using updated weights
        
        y_predicted = np.dot(x_train_batch,weight_vector) 
        current_cost = find_cost(y_train,y_predicted)
            
        # this loop will update all the parameters one by one
        for theta_j in range(len(weight_vector)):
            
            # defining the xj vector for the column corresponding the weight theta_j
            xj_vector = x_train_batch.iloc[:,theta_j]
            
            # defining the vector representing the difference between predicted and actual values
            difference_actual_predicted_vector = (y_predicted-y_train).reshape(len(x_train_batch),-1)
            
            gradient =  np.dot(xj_vector,difference_actual_predicted_vector)
            weight_vector[theta_j] = weight_vector[theta_j] - learning_rate *gradient

        
        # adding cost to cost array after each iteration
        iteration_x_axis_batch.append(iteration)
        cost_y_axis_batch.append(current_cost)
    
    return weight_vector


# In[39]:


# function for finding the predicted value
def find_predicted_value(weight_vector,x_train):
    return np.dot(x_train,weight_vector)


# In[40]:


# function for finding mse and sse
def find_mse_rmse(y_actual,y_predicted):
    sse = 0
    for index in range(len(y_actual)):
        sse += (y_actual[index]-y_predicted[index])**2
    mse = sse/len(y_actual)
    rmse = mse**0.5
    return mse,rmse


# In[41]:


# scatter plot for predicted and actual values
def plot_graph_predicted_values(y_actual,y_predicted,length):
    plt.scatter([index for index in range(0,length)],y_predicted)
    plt.scatter([index for index in range(0,length)],y_actual,color='orange')
    plt.legend(['Predicted Values','Actual Values'])
    plt.show()


# In[42]:


def print_score(y_train_actual,x_train,y_test_actual,x_test,weight_vector,iteration_x_axis_batch,cost_y_axis_batch,k):
    print(f"Cost Function for {k} principal components:\n================================================\n")
    print_cost_function(iteration_x_axis_batch,cost_y_axis_batch)
    
    mse, rmse = find_mse_rmse(y_train_actual,find_predicted_value(weight_vector,x_train))
    print("Train Result:\n================================================\n")
    print("MSE for this regression model is: ",mse)
    print("RMSE for this regression model is: ",rmse)
    plot_graph_predicted_values(y_train_actual,find_predicted_value(weight_vector,x_train),len(x_train))
    
    print("Test Result:\n================================================\n")
    mse, rmse = find_mse_rmse(y_test_actual,find_predicted_value(weight_vector,x_test))
    print("MSE for this regression model is: ",mse)
    print("RMSE for this regression model is: ",rmse)
    plot_graph_predicted_values(y_test_actual,find_predicted_value(weight_vector,x_test),len(x_test))
    return rmse


# In[43]:


# for alpha = 0.0001
learning_rate = 0.0001

y_train = np.array(y_train)
y_test = np.array(y_test)
rmse_arr = []

# trying for different number of principal components
for k in range(1,17):
    iteration_x_axis_batch = []
    cost_y_axis_batch = []
    weight_vector = batch_gradient_descent(x_train,y_train,x_test,k,learning_rate,iteration_x_axis_batch,cost_y_axis_batch)
    x_train_k, x_test_k = give_train_data(x_train,x_test,k)
    rmse_arr.append(print_score(y_train,x_train_k,y_test,x_test_k,weight_vector,iteration_x_axis_batch,cost_y_axis_batch,k))


# ## ``` Plotting Number of Components vs RMSE```

# In[44]:


# graph showing rmse vs number of pca components
pca_components = [x for x in range(1,17)]
plt.figure(figsize=(10, 8))
plt.plot(pca_components,rmse_arr)
plt.xlabel('Number of PCA Components')
plt.ylabel('Test RMSE')
plt.title('Test RMSE vs PCA Components')
plt.axvline(x=4,color='orange',linestyle='--')
plt.axhline(y=rmse_arr[3],color='green',linestyle='--')
plt.xticks(pca_components)
plt.show()


# - Initially, as the number of ```components increases```, the ```RMSE decreases```. This is because a higher number of components capture more variance in the data, allowing the model to predict the underlying patterns better.
# 
# 
# - We see that ```minimum rmse is captured when we take 4 principal components```. The model captures enough information from the data without overfitting to the noise.
# 
# 
# - Beyond 4 number of components, we see that adding more components does not improve the performance. In fact, it ```leads to overfitting```, where the model starts ```capturing noise```in the data rather than genuine patterns.

# ## ```Testing the Most Efficient Model```

# In[45]:


# optimal model is one with 4 principal components as per graph


# In[46]:


iteration_x_axis_batch = []
cost_y_axis_batch = []
learning_rate = 0.0001

weight_vector = batch_gradient_descent(x_train,y_train,x_test,4,learning_rate,iteration_x_axis_batch,cost_y_axis_batch)
x_train_k , x_test_k = give_train_data(x_train,x_test,4)
print_score(y_train,x_train_k,y_test,x_test_k,weight_vector,iteration_x_axis_batch,cost_y_axis_batch,4)
print()


# In[47]:


y_pred = find_predicted_value(weight_vector,x_test_k)
y_pred


# In[48]:


# predicted value for a single point
y_pred[:1]


# ## ```Conclusion and Analysis```

# ### ```Significance of selecting an appropriate number of components```
# 
# - PCA reduces the dimensionality of the dataset by transforming it into a new set of uncorrelated variables. This takes care of capturing the required information and not overfitting by capturing the noise of the data.
# 
# 
# - A lower number of components simplifies the model and also makes the computation easy.
# 
# 
# - It is easier to understand and interpret the contribution of each principal component to the overall prediction.
