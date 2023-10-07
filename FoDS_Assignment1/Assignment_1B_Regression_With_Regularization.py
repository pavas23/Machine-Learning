#!/usr/bin/env python
# coding: utf-8

# # ```ASSIGNMENT 1B - REGRESSION WITH REGULARIZATION```

# ## ```TEAM MEMBERS```
#     1. Pavas Garg - 2021A7PS2587H
#     2. Tushar Raghani - 2021A7PS1404H
#     3. Rohan Pothireddy - 2021A7PS0365H

# # ```Importing the Libraries```
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random


# ## `⏳ Loading the Dataset`
# 

# In[2]:


dataset = pd.read_csv("Fish Data - A2.csv")


# In[3]:


dataset.head()


# In[4]:


print("Number of records in the given dataset are: ",len(dataset))
print("Number of features in the given dataset are: ",len(dataset.columns)-1)


# # ```TASK 1: DATA PREPROCESSING```

# # ```🔬Preprocess and perform exploratory data analysis of the dataset obtained```

# # `📊 Plotting Histograms`

# In[5]:


dataset.hist(bins = 50,figsize=(15,10))
plt.show()


# # `Feature Scaling`

# ### Normalization Method
# - Normalization is performed to transform the data to have a mean of 0 and standard deviation of 1
# - Normalization is also known as Z-Score Normalization
# 
# \begin{equation}
# z = \frac{(x-\mu)}{\sigma}
# \end{equation}

# In[6]:


# to check if null values or NAN are present in the dataset
nan_count = dataset.isna().sum().sum()
null_count = dataset.isnull().sum().sum()
print("NAN count: ",nan_count)
print("NULL count: ",null_count)


# In[7]:


# function for finding mean of a feature in a given dataset
def find_mean(dataset,feature):
    n = len(dataset[feature])
    sum = 0
    count_nan_null_values = 0
    for val in dataset[feature]:
        if not isinstance(val, (int, float)):
            count_nan_null_values += 1
        else:
            sum += val
    return sum/n-count_nan_null_values


# In[8]:


find_mean(dataset,"Width")


# In[9]:


find_mean(dataset,"Height")


# In[10]:


# function to replace NULL values with mean
def replace_with_mean(dataset,feature,mean):
   dataset[feature].fillna(mean,inplace=True)


# In[11]:


nan_count = dataset.isna().sum().sum()
null_count = dataset.isnull().sum().sum()
print("NAN count: ",nan_count)
print("NULL count: ",null_count)


# In[12]:


# replacing values which are 0 with the mean
mean = find_mean(dataset,"Weight")
dataset['Weight'] = dataset['Weight'].replace(0,mean)


# In[13]:


for feature in dataset.columns:
    mean = find_mean(dataset,feature)
    replace_with_mean(dataset,feature,mean)


# In[14]:


# function for finding standard deviation of a feature in a given dataset
def find_standard_deviation(dataset,feature):
    variance, squared_sum = 0,0
    n = len(dataset[feature])
    mean = find_mean(dataset,feature)
    for val in dataset[feature]:
        squared_sum += (val-mean)**2
    variance = squared_sum/n
    return math.sqrt(variance)


# In[15]:


# function for scaling a feature in given dataset
def normalize_feature(dataset,feature):
    mean = find_mean(dataset,feature)
    standard_deviation = find_standard_deviation(dataset,feature)
    normalized_feature = []
    for val in dataset[feature]:
        normalized_feature.append((val-mean)/standard_deviation)
    return normalized_feature


# In[16]:


# function for scaling (normalizing) the whole dataset
def normalize_dataset(dataset):
    df = dataset.drop(columns = "Weight")
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
    normalized_df["Weight"] = dataset["Weight"]
    return normalized_df


# ## `Plot showing distribution of features before normalization`

# In[17]:


# all features following a normal distribution with mean 0 and standard deviation of 1
sns.displot(dataset.drop(columns="Weight"), kind='kde',aspect=1,height=8)
plt.show()


# In[18]:


# normalizing the complete dataset
dataset = normalize_dataset(dataset)
dataset.head()


# In[19]:


# checking mean and variance of each feature after standardizing the dataset
df = dataset.drop(columns = "Weight")
for feature in df:
    print("Mean of",feature,"is",round(find_mean(dataset,feature)))
    print("Standard Deviation of",feature,"is",round(find_standard_deviation(dataset,feature)))


# ## `Plot showing distribution of features after normalization`

# In[20]:


# all features following a normal distribution with mean 0 and standard deviation of 1
sns.displot(dataset.drop(columns="Weight"), kind='kde',aspect=1,height=8)
plt.show()


# # ```Correlation Matrix```

# In[21]:


correlation = dataset.corr()
plt.subplots(figsize=(10,7))
heatmap = sns.heatmap(correlation,annot=True)
heatmap.set(title='Correlation Matrix')
plt.show()


# ## `Plotting Box Plots`

# In[22]:


def plot_boxplot(dataframe,feature):
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green',marker='D',markeredgecolor='green')
    dataframe.boxplot(column=[feature],flierprops = red_circle,showmeans=True,meanprops=mean_shape,notch=True)
    plt.grid(False)
    plt.show()


# In[23]:


# red circles will be the outliers 
plot_boxplot(dataset,"Width")


# In[24]:


# red circles are the outliers 
plot_boxplot(dataset,"Height")


# # ```Generating features for higher degree polynomials```

# In[25]:


dataset.head()


# In[26]:


def give_features(dataset,degree):
    features_deg = []
    feature_name = []
    for i in range(0,degree+1):
        j = degree - i
        vector_temp = []
        for ind in range(0,len(dataset)):
            x1 = dataset['Height'][ind]
            x2 = dataset['Width'][ind]
            vector_temp.append((x1**i)*(x2**j))
        features_deg.append(vector_temp)
        feature_name.append(f"X1^{i}_X2^{j}")
    return features_deg, feature_name


# In[27]:


def give_dataset_with_all_features(dataset):
    df_new = pd.DataFrame()
    # this will give polynomial terms for all polynomials with degree from 2 to 9
    for degree in range(1,10):
        features_degree, feature_name = give_features(dataset,degree)
        # iterating over each feature associated with that degree
        for ind in range(len(features_degree)):
            name = feature_name[ind]
            df_new[name] = features_degree[ind]
            
    return df_new


# In[28]:


df_temp = give_dataset_with_all_features(dataset)
# now add a column of ones
ones = [1 for i in range(len(dataset))]
df_new = pd.DataFrame()
df_new[f"X1^{0}_X2^{0}"] = ones
for feature in df_temp.columns:
    df_new[feature] = df_temp[feature]
df_new['Weight'] = dataset['Weight']
dataset = df_new
dataset.head()


# # ```Train-Test Split```

# In[29]:


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


# In[30]:


train_set, test_set = split_train_test(dataset,0.2)


# In[31]:


print(len(train_set))


# In[32]:


print(len(test_set))


# In[33]:


train_set.head()


# In[34]:


test_set.head()


# # ```TASK 2: POLYNOMIAL REGRESSION```
# 
# - We will use this equation to update our regression model parameters
# 
# $$
# \begin{equation}
# \theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
# \end{equation}
# $$
# 
# $$
# \begin{equation}
# \frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i=1}^n(h_{\theta}(x) - y^{(i)})x_{j}^{(i)} + 
# \frac{\lambda}{2}{q}{\theta_{j}^{(q-1)}}, \quad h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1}  + \theta_{2}x_{2}  +  ... +  \theta_{d}x_{d}
# \end{equation}
# $$
# 
# - Repeat until convergence
# $$
# \begin{equation}
# \theta_{j} = \theta_{j} - {\alpha}{({\sum_{i=1}^n(h_{\theta}(x) - y^{(i)})x_{j}^{(i)}+ \frac{\lambda}{2}{q}{\theta_{j}^{(q-1)}}})} ,\quad\text {$0 \leq j \leq d$}
# \end{equation}
# $$
# 
# - Such that it minimizes the cost function given by equation
# 
# $$
# \begin{equation}
#  J(w) = \frac{1}{2} \sum_{n=1}^{N} \left( y(x_n, w) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{M} \left(  \mathbf{w_{j}}^q \right)
# \end{equation}
# $$
# 
# - Where,
# $$
# \| \mathbf{w} \|_1 |= w_0^q + w_1^q + \ldots + w_d^q \quad\text {,for  d  features } 
# $$

# # ```Functions for finding error```

# In[35]:


# function to find cost value, using the formula
def find_cost_regression(y_actual,y_predicted,theta_vector,Lambda,q):
    cost = 0
    for ind in range(len(y_predicted)):
        cost = cost + int((y_predicted[ind] - y_actual[ind])**2)
    cost = (1/2)*cost
    # adding the penalty term to the cost
    cost += (Lambda*0.5)*np.asarray([abs(w)**q for w in theta_vector])
    return cost[0] # returns the cost value instead of cost array


# In[36]:


# function for finding mse and sse
def find_sse_mse(y_actual,y_predicted):
    sse = 0
    for index in range(len(y_actual)):
        sse += (y_actual[index]-y_predicted[index])**2
    mse = sse/len(y_actual)
    return sse, mse


# In[37]:


# scatter plot for predicted and actual values
def plot_graph_predicted_values(y_actual,y_predicted,length):
    plt.scatter([index for index in range(0,length)],y_predicted)
    plt.scatter([index for index in range(0,length)],y_actual,color='orange')
    plt.legend(['Predicted Values','Actual Values'])
    plt.show()


# In[38]:


# function for finding the predicted value
def find_predicted_value(weight_vector,x_train):
    return np.dot(x_train,weight_vector)


# In[39]:


def print_cost_function(iteration_x_axis_batch,cost_y_axis_batch):
    plt.plot(iteration_x_axis_batch,cost_y_axis_batch)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Value")
    plt.show()


# In[40]:


def print_score(y_train_actual,x_train,y_test_actual,x_test,weight_vector,iteration_x_axis_batch,cost_y_axis_batch):
    print("Cost Function:\n================================================\n")
    print_cost_function(iteration_x_axis_batch,cost_y_axis_batch)
    
    sse_train, mse_train = find_sse_mse(y_train_actual,find_predicted_value(weight_vector,x_train))
    print("Train Result:\n================================================\n")
    print("SSE for this regression model is: ",sse_train)
    print("MSE for this regression model is: ",mse_train)
    plot_graph_predicted_values(y_train_actual,find_predicted_value(weight_vector,x_train),len(x_train))
    
    print("Test Result:\n================================================\n")
    sse_test, mse_test = find_sse_mse(y_test_actual,find_predicted_value(weight_vector,x_test))
    print("SSE for this regression model is: ",sse_test)
    print("MSE for this regression model is: ",mse_test)
    plot_graph_predicted_values(y_test_actual,find_predicted_value(weight_vector,x_test),len(x_test))
    
    return mse_train, mse_test


# # ```Function for train data```

# In[41]:


def give_train_data(degree):
    # features for polynomial with given degree
    num_columns_needed = int((degree+1)*(degree+2)*0.5)
    
    y_train = train_set["Weight"].to_numpy()
    x_train = train_set.iloc[:,:num_columns_needed].to_numpy()
    y_test = test_set["Weight"].to_numpy()
    x_test = test_set.iloc[:,:num_columns_needed].to_numpy()
    
    # initial weights before gradient descent
    weight_vector = np.random.randn(num_columns_needed)
    
    return x_train, y_train, x_test, y_test, weight_vector


# # ```Batch Gradient Descent```

# In[42]:


# defining max_iterations
max_iterations = 10000
learning_rate = 0.0000001
mse_train_degree_batch = []
mse_train_degree_stochastic = []
mse_test_degree_batch = []
mse_test_degree_stochastic = []


# In[43]:


def batch_gradient_descent(degree,learning_rate,Lambda,q,iteration_x_axis_batch,cost_y_axis_batch):
    prev_cost = 0
    
    # get x_train and y_train vectors
    x_train_batch, y_train_batch, x_test_batch, y_test_batch, weight_vector = give_train_data(degree)
    
    for iteration in range(max_iterations):
        # will give the predicted value, after each iteration using updated weights
        
        y_predicted = np.dot(x_train_batch,weight_vector) 
        current_cost = find_cost_regression(y_train_batch,y_predicted,weight_vector,Lambda,q)
            
        # this loop will update all the parameters one by one
        for theta_j in range(len(weight_vector)):
             
            # defining the xj vector for the column corresponding the weight theta_j
            xj_vector = x_train_batch[:,theta_j]
            
            # defining the vector representing the difference between predicted and actual values
            difference_actual_predicted_vector = (y_predicted-y_train_batch).reshape(len(x_train_batch),-1)
            
            gradient =  np.dot(xj_vector,difference_actual_predicted_vector)
            gradient_regression = gradient + (Lambda/2)*q*((abs(weight_vector[theta_j]))**(q-1)) # adding gradient due to penalty term
            
            # for the bias term, don't penalize it
            if(theta_j == 0):
                weight_vector[theta_j] = weight_vector[theta_j] - learning_rate*gradient
            else:
                weight_vector[theta_j] = weight_vector[theta_j] - learning_rate*gradient_regression
            

        # adding cost to cost array after each iteration
        iteration_x_axis_batch.append(iteration)
        cost_y_axis_batch.append(current_cost)
    
    return weight_vector


# # ```Stochastic Gradient Descent```

# In[44]:


def stochastic_gradient_descent(degree,learning_rate,Lambda,q,iteration_x_axis_stochastic,cost_y_axis_stochastic):
    prev_cost = 0
    
    # get x_train and y_train vectors
    x_train_stochastic, y_train_stochastic, x_test_stochastic, y_test_stochastic, weight_vector = give_train_data(degree)
    
    for iteration in range(500):
        # will give the predicted value, after each iteration using updated weights
        
        y_predicted = np.dot(x_train_stochastic,weight_vector) 
        current_cost = find_cost_regression(y_train_stochastic,y_predicted,weight_vector,Lambda,q)
            
        # this loop will update all the parameters one by one
        for theta_j in range(len(weight_vector)):
             
            # this will iterate over each training data point and update the theta_j after each iteration
            for index in range(len(x_train_stochastic)):
                xj = x_train_stochastic[index,theta_j]
                y_predicted_itr = find_predicted_value(weight_vector,x_train_stochastic[index])
                difference_actual_predicted = (y_predicted_itr-y_train_stochastic[index])
                gradient = difference_actual_predicted*xj
                gradient_regression = gradient + (Lambda/2)*q*((abs(weight_vector[theta_j]))**(q-1)) # adding gradient due to penalty term
                
                # update theta_j after each and every data point
                # for the bias term, don't penalize it
                if(theta_j == 0):
                    weight_vector[theta_j] = weight_vector[theta_j] - learning_rate*gradient
                else:
                    weight_vector[theta_j] = weight_vector[theta_j] - learning_rate*gradient_regression
                
                # adding cost to cost array after each updation
                y_predicted = np.dot(x_train_stochastic,weight_vector)
                current_cost = find_cost_regression(y_train_stochastic,y_predicted,weight_vector,Lambda,q)
                iteration_x_axis_stochastic.append(iteration)
                cost_y_axis_stochastic.append(current_cost)

    return weight_vector


# # ```Funtion for printing result of regression```

# In[45]:


q_array = [0.5,1,2,4]


# In[46]:


def print_result(learning_rate,degree,Lambda,q,mse_degree_batch,mse_degree_stochastic):
    iteration_x_axis_batch = []
    cost_y_axis_batch = []

    # for calling batch gradient descent function
    weight_vector_batch = batch_gradient_descent(degree,learning_rate,Lambda,q,iteration_x_axis_batch,cost_y_axis_batch)

    iteration_x_axis_stochastic = []
    cost_y_axis_stochastic = []

    # for calling stochastic gradient descent function
    weight_vector_stochastic =  stochastic_gradient_descent(degree,learning_rate,Lambda,q,iteration_x_axis_stochastic,cost_y_axis_stochastic)

    # for printing the results
    x_train_batch, y_train_batch, x_test_batch, y_test_batch, weight_vector_temp = give_train_data(degree)

    print("Batch Gradient Descent:\n================================================\n")
    mse_train_batch, mse_test_batch = print_score(y_train_batch,x_train_batch,y_test_batch,x_test_batch,weight_vector_batch,iteration_x_axis_batch,cost_y_axis_batch)
    mse_degree_batch.append(mse_test_batch)

    print("Stochastic Gradient Descent:\n================================================\n")
    mse_train_stochastic, mse_test_stochastic = print_score(y_train_batch,x_train_batch,y_test_batch,x_test_batch,weight_vector_stochastic,iteration_x_axis_stochastic,cost_y_axis_stochastic)
    mse_degree_stochastic.append(mse_test_stochastic)
    
    return mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic


# # ```Test run for polynomial with degree 1```

# In[47]:


mse_degree_1_batch = []
mse_degree_1_stochastic = []


# ### ```taking q as 0.5```

# In[48]:


# constants
degree = 1
Lambda = 0.000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_1_batch,mse_degree_1_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[49]:


degree = 1
Lambda = 0.001
q = 1

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_1_batch,mse_degree_1_stochastic)


# ### ```taking q as 2```

# In[50]:


degree = 1
Lambda = 0.00001
q = 2

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_1_batch,mse_degree_1_stochastic)


# ### ```taking q as 4```

# In[51]:


degree = 1
Lambda = 0.00001
q = 4

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_1_batch,mse_degree_1_stochastic)


# ## ```Plot comparing MSE values with different q for degree 1 polynomial```

# In[52]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_1_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_1_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 0.5 and for stochastic gradient descent best model here is one with q as 2```

# # ```Test run for polynomial with degree 2```

# In[53]:


mse_degree_2_batch = []
mse_degree_2_stochastic = []


# ### ```taking q as 0.5```

# In[54]:


# constants
degree = 2
Lambda = 0.01
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_2_batch,mse_degree_2_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[55]:


# constants
degree = 2
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_2_batch,mse_degree_2_stochastic)


# ### ```taking q as 2```

# In[56]:


# constants
degree = 2
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_2_batch,mse_degree_2_stochastic)


# ### ```taking q as 4```

# In[57]:


# constants
degree = 2
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_2_batch,mse_degree_2_stochastic)


# ## ```Plot comparing MSE values with different q for degree 2 polynomial```

# In[58]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_2_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_2_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 4 and for stochastic gradient descent best model here is one with q as 0.5```

# # ```Test run for polynomial with degree 3```

# In[59]:


mse_degree_3_batch = []
mse_degree_3_stochastic = []


# ### ```taking q as 0.5```

# In[60]:


# constants
degree = 3
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_3_batch,mse_degree_3_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[61]:


# constants
degree = 3
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_3_batch,mse_degree_3_stochastic)


# ### ```taking q as 2```

# In[62]:


# constants
degree = 3
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_3_batch,mse_degree_3_stochastic)


# ### ```taking q as 4```

# In[63]:


# constants
degree = 3
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_3_batch,mse_degree_3_stochastic)


# ## ```Plot comparing MSE values with different q for degree 3 polynomial```

# In[64]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_3_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_3_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 0.5 and for stochastic gradient descent best model here is one with q as 0.5```

# # ```Test run for polynomial with degree 4```

# In[65]:


mse_degree_4_batch = []
mse_degree_4_stochastic = []


# ### ```taking q as 0.5```

# In[66]:


# constants
degree = 4
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_4_batch,mse_degree_4_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[67]:


# constants
degree = 4
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_4_batch,mse_degree_4_stochastic)


# ### ```taking q as 2```

# In[68]:


# constants
degree = 4
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_4_batch,mse_degree_4_stochastic)


# ### ```taking q as 4```

# In[69]:


# constants
degree = 4
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_4_batch,mse_degree_4_stochastic)


# ## ```Plot comparing MSE values with different q for degree 4 polynomial```

# In[70]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_4_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_4_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 1 and for stochastic gradient descent best model here is one with q as 1```

# # ```Test run for polynomial with degree 5```

# In[71]:


mse_degree_5_batch = []
mse_degree_5_stochastic = []


# ### ```taking q as 0.5```

# In[72]:


# constants
degree = 5
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_5_batch,mse_degree_5_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[73]:


# constants
degree = 5
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_5_batch,mse_degree_5_stochastic)


# ### ```taking q as 2```

# In[74]:


# constants
degree = 5
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_5_batch,mse_degree_5_stochastic)


# ### ```taking q as 4```

# In[75]:


# constants
degree = 5
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_5_batch,mse_degree_5_stochastic)


# ## ```Plot comparing MSE values with different q for degree 5 polynomial```

# In[76]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_5_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_5_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 1 and for stochastic gradient descent best model here is one with q as 1```

# # ```Test run for polynomial with degree 6```

# In[77]:


mse_degree_6_batch = []
mse_degree_6_stochastic = []


# ### ```taking q as 0.5```

# In[78]:


# constants
degree = 6
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_6_batch,mse_degree_6_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[79]:


# constants
degree = 6
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_6_batch,mse_degree_6_stochastic)


# ### ```taking q as 2```

# In[80]:


# constants
degree = 6
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_6_batch,mse_degree_6_stochastic)


# ### ```taking q as 4```

# In[81]:


# constants
degree = 6
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_6_batch,mse_degree_6_stochastic)


# ## ```Plot comparing MSE values with different q for degree 6 polynomial```

# In[82]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_6_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_6_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 2 and for stochastic gradient descent best model here is one with q as 2```

# # ```Test run for polynomial with degree 7```

# In[83]:


mse_degree_7_batch = []
mse_degree_7_stochastic = []


# ### ```taking q as 0.5```

# In[84]:


# constants
degree = 7
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_7_batch,mse_degree_7_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[85]:


# constants
degree = 7
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_7_batch,mse_degree_7_stochastic)


# ### ```taking q as 2```

# In[86]:


# constants
degree = 7
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_7_batch,mse_degree_7_stochastic)


# ### ```taking q as 4```

# In[87]:


# constants
degree = 7
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_7_batch,mse_degree_7_stochastic)


# ## ```Plot comparing MSE values with different q for degree 7 polynomial```

# In[88]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_7_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_7_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 1 and for stochastic gradient descent best model here is one with q as 1```

# # ```Test run for polynomial with degree 8```

# In[89]:


mse_degree_8_batch = []
mse_degree_8_stochastic = []


# ### ```taking q as 0.5```

# In[90]:


# constants
degree = 8
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(learning_rate,degree,Lambda,q,mse_degree_8_batch,mse_degree_8_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[91]:


# constants
degree = 1
Lambda = 0.001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_1_batch,mse_degree_1_stochastic)
degree = 8
Lambda = 0.000000001
q = 1

print_result(learning_rate,degree,Lambda,q,mse_degree_8_batch,mse_degree_8_stochastic)


# ### ```taking q as 2```

# In[92]:


# constants
degree = 8
Lambda = 0.000000001
q = 2

print_result(learning_rate,degree,Lambda,q,mse_degree_8_batch,mse_degree_8_stochastic)


# ### ```taking q as 4```

# In[93]:


# constants
degree = 8
Lambda = 0.000000001
q = 4

print_result(learning_rate,degree,Lambda,q,mse_degree_8_batch,mse_degree_8_stochastic)


# ## ```Plot comparing MSE values with different q for degree 8 polynomial```

# In[94]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_8_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_8_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 1 and for stochastic gradient descent best model here is one with q as 1```

# # ```Test run for polynomial with degree 9```

# In[95]:


mse_degree_9_batch = []
mse_degree_9_stochastic = []


# ### ```taking q as 0.5```

# In[97]:


# constants
degree = 9
Lambda = 0.000000001
q = 0.5

mse_train_batch, mse_test_batch, mse_train_stochastic, mse_test_stochastic = print_result(0.00000000001,degree,Lambda,q,mse_degree_9_batch,mse_degree_9_stochastic)
mse_train_degree_batch.append(mse_train_batch)
mse_train_degree_stochastic.append(mse_train_stochastic)
mse_test_degree_batch.append(mse_test_batch)
mse_test_degree_stochastic.append(mse_test_stochastic)


# ### ```taking q as 1```

# In[98]:


# constants
degree = 9
Lambda = 0.000000001
q = 1

print_result(0.00000000001,degree,Lambda,q,mse_degree_9_batch,mse_degree_9_stochastic)


# ### ```taking q as 2```

# In[99]:


# constants
degree = 9
Lambda = 0.000000001
q = 2

print_result(0.00000000001,degree,Lambda,q,mse_degree_9_batch,mse_degree_9_stochastic)


# ### ```taking q as 4```

# In[100]:


# constants
degree = 9
Lambda = 0.000000001
q = 4

print_result(0.00000000001,degree,Lambda,q,mse_degree_9_batch,mse_degree_9_stochastic)


# ## ```Plot comparing MSE values with different q for degree 9 polynomial```

# In[101]:


plt.title(f"Graph with learning rate = {learning_rate}")
plt.plot(q_array,mse_degree_9_batch,label='Batch Gradient Descent')
plt.plot(q_array,mse_degree_9_stochastic,label='Stochastic Gradient Descent')
plt.xlabel("Q Value")
plt.ylabel("MSE Value")
plt.legend()
plt.show()


# ### ```we see that for batch gradient descent best model here is one with least MSE, which is with q as 0.5 and for stochastic gradient descent best model here is one with q as 4```

# ## ```Train-Test MSE vs Degree of Polynomial Plot```

# ## ```Batch Gradient Descent Plot```

# In[104]:


# Plot the training and testing error vs degree of polynomial
degrees = [i for i in range(1,10)]

# for batch gradient descent
plt.figure(figsize=(10, 8))
plt.title('Training/Testing Error vs. Degree of Polynomial for Batch Gradient Descent')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Training/Testing Error')
plt.plot(degrees,mse_train_degree_batch,label='Training Error')
plt.plot(degrees,mse_test_degree_batch,label='Testing Error')
plt.legend()
plt.show()


# ### ``` Best Model for Batch Gradient Descent is one with degree 7, as test error is minimum```

# ## ```Stochastic Gradient Descent Plot```

# In[137]:


# for stochastic gradient descent
plt.figure(figsize=(10, 8))
plt.title('Training/Testing Error vs. Degree of Polynomial for Stochastic Gradient Descent')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Training/Testing Error')
plt.plot(degrees,mse_train_degree_stochastic,label='Training Error')
plt.plot(degrees,mse_test_degree_stochastic,label='Testing Error')
plt.legend()
plt.show()


# ### ```Best Model for Stochastic Gradient Descent is one with degree 7,  as test error is minimum```

# ## ```Tabulating training and testing errors for polynomials```

# In[108]:


# batch gradient descent
df_batch = pd.DataFrame()
df_batch["Degree of Polynomial"] = degrees
df_batch["MSE for Training Data"] = mse_train_degree_batch
df_batch["MSE for Testing Data"] = mse_test_degree_batch
df_batch


# In[110]:


# stochastic gradient descent
df_stochastic = pd.DataFrame()
df_stochastic["Degree of Polynomial"] = degrees
df_stochastic["MSE for Training Data"] = mse_train_degree_stochastic
df_stochastic["MSE for Testing Data"] = mse_test_degree_stochastic
df_stochastic


# ## ```Observation on overfitting```
# - We observe that as we increase the degree of polynomial, the difference between test and train error increases, which shows there is overfitting, and our model fits completely on train data, giving large values of error for testing data
# 
# - We also observe that for lower degree polynomials both test and train errors have higer values because of underfitting the data

# # ```TASK 3: GRAPH PLOTTING```

# In[111]:


from mpl_toolkits.mplot3d import Axes3D

def generate_surface_plot(degree,learning_rate,Lambda,q):

    iteration_x_axis_batch = []
    cost_y_axis_batch = []

    # Assuming you have predictions for a meshgrid
    # Replace these with your actual predictions and meshgrid
    # Example data (replace with actual data):
    min_width = train_set["X1^0_X2^1"].min()
    max_width = train_set["X1^0_X2^1"].max()
    min_height = train_set["X1^1_X2^0"].min()
    max_height = train_set["X1^1_X2^0"].max()	

    # Assuming you have predictions for a meshgrid
    # Replace these with your actual predictions and meshgrid
    # Example data (replace with actual data):
    x1_mesh, x2_mesh = np.meshgrid(np.linspace(min_height, max_height, 250),np.linspace(min_width, max_width, 250))
    # predicted_weights = your_model.predict(np.c_[x1_mesh.ravel(), x2_mesh.ravel()]).reshape(x1_mesh.shape)

    x1_flat = x1_mesh.ravel()
    x2_flat = x2_mesh.ravel()

    # Create an array of pairs of meshgrid points (width, height)
    meshgrid_points = np.c_[x1_flat, x2_flat]
    df = pd.DataFrame(meshgrid_points, columns=['Height', 'Width'])
    df_temp = give_dataset_with_all_features(df)
    # now add a column of ones
    ones = [1 for i in range(len(df))]
    df_new = pd.DataFrame()
    df_new[f"X1^{0}_X2^{0}"] = ones
    for feature in df_temp.columns:
        df_new[feature] = df_temp[feature]
    df = df_new

    num_columns_needed = int((degree+1)*(degree+2)*0.5)
    weight_vector = np.random.randn(num_columns_needed)
    weight_vector = batch_gradient_descent(degree,learning_rate,Lambda,q,iteration_x_axis_batch,cost_y_axis_batch)
    
    df = df.iloc[:,:num_columns_needed].to_numpy()
    y_predicted = find_predicted_value(weight_vector,df)
    y_pred = np.reshape(y_predicted,[250,250])

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x1_mesh, x2_mesh, y_pred, cmap='viridis')

    #The viridis color map transitions smoothly from a light color (e.g., light yellowish-green) for low values to darker colors (e.g., dark purple) for higher values.

    # Label the axes
    ax.set_xlabel('Width of Fish')
    ax.set_ylabel('Height of Fish')
    ax.set_zlabel('Predicted Weight of Fish')

    # Show the plot
    plt.show()


# # ```Plotting Surface Plots for every Polynomial```

# ## ```Surface Plots for four regularized linear regression models```

# In[112]:


generate_surface_plot(1,0.00000000001,0.000000001,0.5)


# In[113]:


generate_surface_plot(1,0.00000000001,0.000000001,1)


# In[114]:


generate_surface_plot(1,0.00000000001,0.000000001,2)


# In[115]:


generate_surface_plot(1,0.00000000001,0.000000001,4)


# ## ```Surface Plot for Degree 2 Polynomial```

# In[116]:


generate_surface_plot(2,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 3 Polynomial```

# In[117]:


generate_surface_plot(3,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 4 Polynomial```

# In[118]:


generate_surface_plot(4,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 5 Polynomial```

# In[119]:


generate_surface_plot(5,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 6 Polynomial```

# In[120]:


generate_surface_plot(6,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 7 Polynomial```

# In[121]:


generate_surface_plot(7,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 8 Polynomial```

# In[122]:


generate_surface_plot(8,0.00000000001,0.000000001,0.5)


# ## ```Surface Plot for Degree 9 Polynomial```

# In[123]:


generate_surface_plot(9,0.00000000001,0.000000001,0.5)


# ## DESCRIPTION OF MODEL USED
# 1. We first started off generating nine polynomial regression models using Batch and Stochastic Gradient Descent, in which we varied the value of q among 0.5,1,2,4, and for each case, we plotted the graph for cost function v/s epochs (iterations). 
# 2. We also found SSE and MSE for each case
# 3. Among these models developed, we found the best model for both Batch and Stochastic Gradient Descent by plotting a graph between training and testing data v/s degree of the polynomial, for both the gradient descent methods.
# 4. We then plotted surface plots for each of the nine polynomial regression models, and then analyzed the outcome of the same.

# ## ALGORITHMS USED
# 
# 1. We used Batch Gradient Descent and Stochastic Gradient Descent algorithms with regularization. 
# 2. We observe a smoother cost curve in Batch Gradient Descent as compared to that of Stochastic Gradient Descent.
# 
# ## Implementation of Regularization
# 
# 1. We used regularization while implementing the Gradient Descent algorithm.
# 2. Regularization adds a penalty term in the gradient expression, which is used to update our weights.

# # ```TASK 4: COMPARATIVE ANALYSIS```

# - We observe that polynomial with degree 7 is the best fit polynomial among the nine models developed

# In[126]:


mse_for_best_fit_train = mse_train_degree_batch[6]
mse_for_best_fit_test = mse_test_degree_batch[6]
print("MSE for polynomial with degree 7 is (Train): ",mse_for_best_fit_train)
print("MSE for polynomial with degree 7 is (Test): ",mse_for_best_fit_test)


# In[127]:


# for the four regularized linear regression models, we see that MSE values were much higher than that for best fit polynomial
print("MSE for polynomial with degree 0 is (Train): ",mse_train_degree_batch[0])
print("MSE for polynomial with degree 0 is (Test): ",mse_test_degree_batch[0])


# In[135]:


# so for all four regularized linear models with different values of q namely 0.5, 1, 2 and 4 have much higher MSE value
degree = [1,2,3,4,7]
mse_test_val = []
for i in range(len(mse_degree_1_batch)-1):
    mse_test_val.append(mse_degree_1_batch[i])
mse_test_val.append(mse_for_best_fit_test)
plt.title("MSE for Linear Models and Best fit Polynomial Model")
plt.scatter(degree,mse_test_val)
plt.show()


# In[136]:


# here 1,2,3,4 represent MSE for linear model with q value as 0.5, 1, 2, 4 which have much higher MSE as compared
# best fit polynomial with degree 7


# ### ```END OF ASSIGNMENT 1B - REGRESSION WITH REGULARIZATION```
