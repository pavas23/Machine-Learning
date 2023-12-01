#!/usr/bin/env python
# coding: utf-8

# # ```CS F320 - FOUNDATIONS OF DATA SCIENCE```
# 
# ## ```ASSIGNMENT 2A - Implementing PCA from Scratch```
# 
# ## ```TEAM MEMBERS: ```
#         1. Pavas Garg - 2021A7PS2587H
#         2. Tushar Raghani - 2021A7PS1404H
#         3. Rohan Pothireddy - 2021A7PS0365H 

# # ```Importing the Libraries```
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random


# # `⏳ Loading the Dataset`

# In[2]:


df = pd.read_csv("audi.csv")
df.head()


# In[3]:


df.shape


# In[4]:


# checking categorical variables
df.select_dtypes(exclude=['number']).head()


# ## ```Dropping Categorical Variables```

# In[5]:


# Drop all categorical columns
df = df.select_dtypes(exclude='object')
df


# ## ```Excluding the target variable```

# In[6]:


target_var = df["price"]


# In[7]:


df = df.drop(columns='price')


# ## ```1. Data Understanding and Representation```

# In[8]:


print("Number of records in the given dataset are: ",len(df))
print("Number of features in the given dataset are: ",len(df.columns))


# In[9]:


# representing data in matrix format, each row representing a car, columns representing features
feature_matrix = df.values


# In[10]:


print(feature_matrix)


# # ```🔬Preprocess and perform exploratory data analysis of the dataset obtained```

# # `📊 Plotting Histograms`

# In[11]:


df.hist(bins=50,figsize=(15,10))
plt.show()


# In[12]:


# understanding the given data
df.describe()


# ## ```Correlation Matrix```

# In[13]:


correlation = df.corr()
plt.subplots(figsize=(10,7))
heatmap = sns.heatmap(correlation,annot=True)
heatmap.set(title='Correlation Matrix')
plt.show()


# ## ```Plotting Box Plots```

# In[14]:


def plot_boxplot(dataframe,feature):
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green',marker='D',markeredgecolor='green')
    dataframe.boxplot(column=[feature],flierprops = red_circle,showmeans=True,meanprops=mean_shape,notch=True)
    plt.grid(False)
    plt.show()


# In[15]:


# red circles will be the outliers 
plot_boxplot(df,"mileage")


# In[16]:


# red circles are the outliers 
plot_boxplot(df,"engineSize")


# ## ```2. Implementing PCA using Covariance Matrices```

# - It is used to reduce the dimensionality of dataset by transforming a large set into a lower dimensional set that still contains most of the information of the large dataset
# 
# - Principal component analysis (PCA) is a technique that transforms a dataset of many features into principal components that "summarize" the variance that underlies the data
# 
# - PCA finds a new set of dimensions such that all dimensions are orthogonal and hence linearly independent and ranked according to variance of data along them
# 
# - Eigen vectors point in direction of maximum variance among data, and eigen value gives the importance of that eigen vector
# 
# ![PCA_Image](https://miro.medium.com/v2/resize:fit:1192/format:webp/1*QinDfRawRskupf4mU5bYSA.png)

# - First the dataset is centered by subtracting means from the feature values
# 
# - Means are calculated by 
# 
# $$
# \mu = \frac{1}{n} \sum_{i=1}^{n} x_i
# $$
# 
# - Centered features are then compted as 
# 
# $$
# x_i = x_i - \mu
# $$

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

# In[17]:


# calculating mean of each feature in the dataset
feature_means = df.mean()
feature_means


# In[18]:


# Centering the dataset by subtracting the mean from each feature. 
centered_features = df - feature_means


# In[19]:


centered_features.head()


# ## ```Covariance matrix of the centered dataset```

# In[20]:


# covariance matrix of centered feature values
covariance_matrix = np.cov(centered_features,rowvar=False)
plt.subplots(figsize=(10,7))
heatmap = sns.heatmap(covariance_matrix,annot=True)
heatmap.set(title='Covariance Matrix of the Centered Dataset')
plt.show()


# ## ```3. Eigenvalue Eigenvector Equation```

# For a square matrix $A$, if $\mathbf{v}$ is an eigenvector and $\lambda$ is the corresponding eigenvalue, the eigenvalue-eigenvector equation is given by 
#  
# $$
# A \mathbf{v} = \lambda \mathbf{v} 
# $$

# In[21]:


# finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
# transpose eigenvector
eigenvectors = eigenvectors.T
    
# will give indexes according to eigen values, sorted in decreasing order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[idx]


# ## ```4. Solving for Principal Components ```

# In[22]:


print(eigenvalues)


# In[23]:


print(eigenvectors)


# In[24]:


# for finding how much variance does each principal component capture
explained_variance = eigenvalues / np.sum(eigenvalues)


# In[25]:


print(explained_variance)


# In[26]:


# slicing first k eigenvectors
# let k = 5
k = 5
k_principal_components = eigenvectors[:k]


# In[27]:


print(k_principal_components)


# In[28]:


k_principal_components_eigenvalues = eigenvalues[:k]


# ## ```5. Sequential Variance Increase```

# In[29]:


# total variance covered by principal components
total_variance = np.sum(k_principal_components_eigenvalues)


# In[30]:


print(total_variance)


# In[31]:


# new features after applying PCA
pca_df = np.dot(centered_features,k_principal_components.T)
pca_df = pd.DataFrame(pca_df)
pca_df.head()


# ## ```Plot showing spread of data along first 2 Principal Components```

# In[32]:


plt.figure(figsize=(12,7))
sns.scatterplot(data=pca_df,x=0,y=1,color='orange')
plt.title("Scatter Plot",fontsize=16)
plt.xlabel('First Principal Component',fontsize=16)
plt.ylabel('Second Principal Component',fontsize=16)
plt.show()


# ## ```Plot showing variance captured by each Principal Component```

# In[33]:


num_components = len(explained_variance)
components = np.arange(1, num_components + 1)
plt.figure(figsize=(10, 8))
plt.plot(components, explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(components)
plt.show()


# In[34]:


# finding cumulative variance captured by principal components
y_var = np.cumsum(explained_variance)

plt.figure(figsize=(15,10))
plt.plot(components, y_var, marker='o', linestyle='-', color='green')
plt.title('The number of components needed to explain variance',fontsize=14)
plt.xlabel('Number of Components',fontsize=16)
plt.ylabel('Cumulative variance (%)',fontsize=16)
plt.xticks(components)

# line showing number of principal components required to capture most of the variance
plt.axvline(x=2.00, color='blue', linestyle='--')

plt.show()


# -  ```We can see that the complete variance is captured by the first 2 principal components and there is very insignificant sequential increase in the variance as we consider more principal components```

# ## ```Standardized Dataset```

# In[35]:


df_standardized = (df-df.mean())/(df.std())
df_standardized


# In[36]:


# covariance matrix of centered feature values
covariance_matrix = np.cov(df_standardized,rowvar=False)
plt.subplots(figsize=(10,7))
heatmap = sns.heatmap(covariance_matrix,annot=True)
heatmap.set(title='Covariance Matrix of the Standardized Dataset')
plt.show()


# In[37]:


# finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
# transpose eigenvector
eigenvectors = eigenvectors.T
    
# will give indexes according to eigen values, sorted in decreasing order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[idx]


# ## ```4. Solving for Principal Components ```

# In[38]:


print(eigenvalues)


# In[39]:


print(eigenvectors)


# In[40]:


# for finding how much variance does each principal component capture
explained_variance = eigenvalues / np.sum(eigenvalues)


# In[41]:


print(explained_variance)


# In[42]:


# slicing first k eigenvectors
# let k = 5
k = 5
k_principal_components = eigenvectors[:k]


# In[43]:


print(k_principal_components)


# In[44]:


k_principal_components_eigenvalues = eigenvalues[:k]


# ## ```5. Sequential Variance Increase```

# In[45]:


# total variance covered by principal components
total_variance = np.sum(k_principal_components_eigenvalues)


# In[46]:


print(total_variance)


# In[47]:


# new features after applying PCA
pca_df = np.dot(centered_features,k_principal_components.T)
pca_df = pd.DataFrame(pca_df)
pca_df.head()


# ## ```Plot showing spread of data along first 2 Principal Components```

# In[48]:


plt.figure(figsize=(12,7))
sns.scatterplot(data=pca_df,x=0,y=1,color='orange')
plt.title("Scatter Plot",fontsize=16)
plt.xlabel('First Principal Component',fontsize=16)
plt.ylabel('Second Principal Component',fontsize=16)
plt.show()


# ## ```Plot showing variance captured by each Principal Component```

# In[49]:


num_components = len(explained_variance)
components = np.arange(1, num_components + 1)
plt.figure(figsize=(10, 8))
plt.plot(components, explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(components)
plt.show()


# In[50]:


# finding cumulative variance captured by principal components
y_var = np.cumsum(explained_variance)

plt.figure(figsize=(15,10))
plt.plot(components, y_var, marker='o', linestyle='-', color='green')
plt.title('The number of components needed to explain variance',fontsize=14)
plt.xlabel('Number of Components',fontsize=16)
plt.ylabel('Cumulative variance (%)',fontsize=16)
plt.axhline(y=0.9,color='orange',linestyle='--')
plt.xticks(components)

# line showing number of principal components required to capture most of the variance
plt.axvline(x=3.00, color='blue', linestyle='--')

plt.show()


# -  ```We can see that the complete variance is captured by the first 3 principal components and there is very insignificant sequential increase in the variance as we consider more principal components```

# ## ```6. Visualization using Pair Plots```

# In[51]:


# pair plot of original features
original_pair_plot = sns.pairplot(pd.DataFrame(df,columns=df.columns),height=5)

# Set the title of the figur
original_pair_plot.fig.suptitle('Pair Plots of Original Features', y=1.02, fontsize=24)

# Display the pair plots
plt.show()


# ## ```Projecting Principal Components on Original Dataset```
# 
# - Projecting principal components onto these pair plots and visualizing them as vectors, as we have 5 principal components and each of these 5 vectors have 5 components, so for projecting principal components onto these plots, we will take ith and jth component of every eigenvector for plotting it on the pair plot for ith and jth feature in original dataset

# In[52]:


# projecting principal components onto these plots and visualizing them as vectors
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25,50))

# for going on every feature in original dataset
row, col = 0,0
for i in range(0,5):
    for j in range(i+1,5):
        # this will plot scatter plot for all points for ith and jth feature
        ax[row,col].scatter(df.iloc[:,i],df.iloc[:,j],label=f'original')
        ax[row,col].set_xlabel(df.columns[i])
        ax[row,col].set_ylabel(df.columns[j])
        # plotting each and every eigenvector
        for vec in eigenvectors:
            # plotting ith and jth component of each eigenvector
            ax[row,col].quiver(0, 0, vec[i], vec[j], angles='xy', scale_units='xy', scale=0.00001, color='r')
        col += 1
        if col > 1:
            col = 0
            row += 1
plt.show()


# ## ```Projecting Principal Components on Centered Dataset```
# 
# - We can observe that points on scatter plot are around projected principal components, because the data is centered around the mean.

# In[53]:


# projecting principal components onto these plots and visualizing them as vectors
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25,50))

# for going on every feature in original dataset
row, col = 0,0
for i in range(0,5):
    for j in range(i+1,5):
        # this will plot scatter plot for all points for ith and jth feature
        ax[row,col].scatter(centered_features.iloc[:,i],centered_features.iloc[:,j],label=f'original')
        ax[row,col].set_xlabel(df.columns[i])
        ax[row,col].set_ylabel(df.columns[j])
        # plotting each and every eigenvector
        for vec in eigenvectors:
            # plotting ith and jth component of each eigenvector
            ax[row,col].quiver(0, 0, vec[i], vec[j], angles='xy', scale_units='xy', scale=0.00001, color='orange')
        col += 1
        if col > 1:
            col = 0
            row += 1
plt.show()


# ## ```Projecting Principal Components on Standardized Dataset```
# 
# - We can see that points on scatter plot are around the vectors, as here we have properly scaled dataset

# In[54]:


df_std = (df-df.mean())/df.std()

# projecting principal components onto these plots and visualizing them as vectors
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(25,50))

# for going on every feature in original dataset
row, col = 0,0
for i in range(0,5):
    for j in range(i+1,5):
        # this will plot scatter plot for all points for ith and jth feature
        ax[row,col].scatter(df_std.iloc[:,i],df_std.iloc[:,j],label=f'original')
        ax[row,col].set_xlabel(df.columns[i])
        ax[row,col].set_ylabel(df.columns[j])
        # plotting each and every eigenvector
        for vec in eigenvectors:
            # plotting ith and jth component of each eigenvector
            ax[row,col].quiver(0, 0, vec[i], vec[j], angles='xy', scale_units='xy', scale=0.001, color='g')
        col += 1
        if col > 1:
            col = 0
            row += 1
plt.show()


# ## ```7. Conclusion and Interpretation```
# 
# - We can see that maximum variance is captured by first two principal components, which helps us in reducing dimension of dataset from 5 features to 2 features while retaining most of the information.
# 
# 
# - PCA is a dimensionality reduction technique, and dimensionality reduction is the process of reducing the number of features in a dataset while retaining as much information as possible. This can be done to reduce the complexity of a model, improve the performance of a learning algorithm, or make it easier to visualize the data.
# 
# 
# - PCA converts a set of correlated features in the high dimensional space into a series of uncorrelated features in the low dimensional space. These uncorrelated features are also called principal components.
# 
# 
# - We can see that points on scatter plot are around the vectors when we take projections of principal components for a standardized dataset, as here we have properly scaled dataset, unlike when we took projection on original dataset.
