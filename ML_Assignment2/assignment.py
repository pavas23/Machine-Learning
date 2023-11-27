#!/usr/bin/env python
# coding: utf-8

# # BITS F464 - Semester 1 - MACHINE LEARNING
# 
# ## ASSIGNMENT 2 – DECISION TREES AND SUPPORT VECTOR MACHINES
# 
# ### Team number: 24
# 
# #### Full names of all students in the team:
# Pavas Garg, Tushar Raghani, Rohan Pothireddy, Kolasani Amit Vishnu
# 
# #### Id number of all students in the team:
#  2021A7PS2587H, 2021A7PS1404H, 2021A7PS0365H, 2021A7PS0151H

# # 1. Preprocess and perform exploratory data analysis of the dataset obtained

# ## ```Importing The Libraries```

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random


# ## ```Importing the Dataset```

# In[2]:


df = pd.read_csv("communities.data",header=None)
df.head()


# # ```1. 🔬Preprocessing of Dataset```

# In[3]:


# regular expression model
import re

# Read the .names file
with open('communities.names', 'r') as file:
    names_content = file.read()

# Find the start and end positions of the attribute names
start_pos = names_content.find('@attribute') + len('@attribute')
end_pos = names_content.find('@data')

# Extract the attribute names section
attributes_section = names_content[start_pos:end_pos]

# Split the attribute names into a list
attributes_list = [line.split()[1] for line in attributes_section.split('\n') if line.strip()]

# Special handling for the first attribute to ensure correct extraction
first_attribute_line = names_content[names_content.find('@attribute'):names_content.find('\n', names_content.find('@attribute'))]
first_attribute_name = re.search('@attribute\s+(\S+)\s', first_attribute_line).group(1)
attributes_list[0] = first_attribute_name

# Print the extracted attribute names
print("Attribute Names:")
print(attributes_list)


# In[4]:


# adding attribute names for the dataset
attribute_names = attributes_list
df.columns = attribute_names

print("Number of observations in this dataset are: ", len(df))
print("Number of features in this dataset are: ", len(df.columns)-1)
df.head()


# In[5]:


# understanding the given data
df.describe()


# In[6]:


# to suppress specific warning 
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)


# # ```Missing Values```

# In[7]:


# replacing missing values represented by ? by NA
df.replace('?', pd.NA, inplace=True)


# In[8]:


# heatmap for missing data visualization
plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),cmap="YlGnBu",cbar_kws={'label': 'Missing Data'})
plt.show()


# In[9]:


# printing columns which have null values
[col for col in df.columns if df[col].isnull().any()]


# In[10]:


# Set the threshold for null values 
threshold = 1000

# Count null values in each column
missing_values = df.isna().sum()
print("Count of missing values:")
print(missing_values[missing_values > 0])

# Filter columns that exceed the threshold
columns_to_drop = missing_values[missing_values > threshold].index

# Display columns with null values exceeding the threshold
print("\nColumns with null values exceeding 1000:")
print(columns_to_drop)

# Drop columns exceeding the threshold
df = df.drop(columns=columns_to_drop)


# In[11]:


# function to replace NULL values with mean
def replace_with_mean(df,feature,mean):
   df[feature].fillna(mean,inplace=True)


# In[12]:


# checking null count
null_count = df.isnull().sum().sum()
print("NULL count: ",null_count)


# In[13]:


# printing features with missing values
missing_values = df.isna().sum()
print(missing_values[missing_values>0])


# In[14]:


# We see that the column OtherPerCap has only one missing value, so we replace it with the mean
mean_value = pd.to_numeric(df['OtherPerCap'], errors='coerce').mean()
replace_with_mean(df,'OtherPerCap',mean_value)


# In[15]:


null_count = df.isnull().sum().sum()
print("NULL count: ",null_count)


# # ```Categorical Variables```

# In[16]:


# checking for categorical features
categorical_columns = df.select_dtypes(include=['category', 'object']).columns
print(len(categorical_columns))


# In[17]:


categorical_columns


# In[18]:


# checking community name feature
df['communityname'].value_counts()


# In[19]:


# as we can see that community name doesn't play important role in predicting crime rate so we can drop it
df = df.drop('communityname',axis=1)


# In[20]:


type_of_values = [type(val) for val in df["OtherPerCap"]]
print(type_of_values[:5])


# In[21]:


# converting string type to float
df["OtherPerCap"] = [float(val) for val in df["OtherPerCap"]]
df["OtherPerCap"]


# In[22]:


df.shape


# # ```Correlation Matrix```
# 
# - This correlation matrix shows top 10 features which are highly correlated with target feature.

# In[23]:


# correlation matrix
correlation_matrix = df.select_dtypes(include='number').corr()
plt.subplots(figsize=(20,15))

# finding correlation with target attribute
correlation_with_target = correlation_matrix["ViolentCrimesPerPop"]

# sort features according to correlation 
sorted_features = correlation_with_target.abs().sort_values(ascending=False)
top_n_features = sorted_features.index[:10]
subset_df = df[top_n_features]

sns.heatmap(subset_df.corr(), cmap="coolwarm", annot=True, linewidths=.5)
plt.show()


# ## ```Plotting Correlation Graphs for Strongly Related Features```

# In[24]:


from pandas.plotting import scatter_matrix
attributes = ["PctKids2Par","PctIlleg","ViolentCrimesPerPop"]
scatter_matrix(df[attributes],figsize=(12,8))
plt.show()


# # ```Discretizing continuous target variable```

# In[25]:


target_variable = df.columns[-1]

# Plot the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(df[target_variable], bins=30, kde=False, color='blue', edgecolor='black')

# Set labels and title
plt.xlabel(target_variable)
plt.ylabel('Frequency')
plt.title('Distribution of {}'.format(target_variable))

# Show the plot
plt.show()


# Converting target label "ViolentCrimesPerPop" to discrete class labels
# 
# We will make 4 classes 
# 
# - ViolentCrimesPerPop less than or equal to 0.25 will belong to class label "low"
# - ViolentCrimesPerPop greater than 0.25 and less than or equal to 0.5  will belong to class label "medium"
# - ViolentCrimesPerPop greater than 0.5 and less than or equal to 0.75  will belong to class label "high"
# - ViolentCrimesPerPop greater than 0.75 and less than or equal to 1 will belong to class label "very high"

# In[26]:


# we will do label encoding for target variable
crime_rate_values = np.array(df["ViolentCrimesPerPop"])

for ind in range(len(crime_rate_values)):
    if crime_rate_values[ind] <= 0.25:
        crime_rate_values[ind] = 1
    elif crime_rate_values[ind] > 0.25 and crime_rate_values[ind] <= 0.5:
        crime_rate_values[ind] = 2
    elif crime_rate_values[ind] > 0.5 and crime_rate_values[ind] <= 0.75:
        crime_rate_values[ind] = 3
    else:
        crime_rate_values[ind] = 4
        
df["ViolentCrimesPerPop"] = crime_rate_values


# In[27]:


# plotting pie chart for target attribute
crime_rate = df["ViolentCrimesPerPop"]

categories = ["low","medium","high","very high"]
frequency_crime_rate = [0,0,0,0]

for val in crime_rate:
    if val == 1:
        frequency_crime_rate[0] += 1
    elif val == 2:
        frequency_crime_rate[1] += 1
    elif val == 3:
        frequency_crime_rate[2] += 1
    elif val == 4:
        frequency_crime_rate[3] += 1
        
dictionary = dict(zip(categories, frequency_crime_rate))

# plotting pie chart
df_crime_rate = pd.DataFrame({'Crime Rate': frequency_crime_rate,},index=categories)
plot = df_crime_rate.plot.pie(y='Crime Rate', figsize=(5, 5))


# # ```Visualizing Class Imbalance```

# In[28]:


df["ViolentCrimesPerPop"].value_counts()


# In[29]:


sns.countplot(x = 'ViolentCrimesPerPop',data = df)
plt.show()


# ```Here we can observe that number of samples in dataset for very high crime rate is much less than number of samples which correspond to low crime rate, hence there is class imbalance.```

# # ```Train-Test Split```

# In[30]:


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


# In[31]:


train_set, test_set = split_train_test(df,0.2)


# In[32]:


train_set.head()


# In[33]:


len(train_set)


# In[34]:


test_set.head()


# In[35]:


len(test_set)


# In[36]:


# changing indexing to 0 based indexing
train_set.index = range(len(train_set))
train_set


# In[37]:


# changing indexing to 0 based indexing
test_set.index = range(len(test_set))
test_set


# # ```Handling Data Imbalance```
# 
# Methods to handle data imbalance are:
# - Undersampling the majority class
# - Oversampling the minority class (by duplicating items)
# - Oversampling minority class (using SMOTE - Synthetic Minority Oversamping Technique, by K Nearest Neighbors algorithm)
# - Ensemble Method 
# - Focal Loss

# #### SMOTE (Synthetic Minority Oversampling Technique)
# 
# - It aims to balance class distribution by randomly increasing minority class examples by replicating them.
# - SMOTE synthesises new minority instances between existing minority instances. It generates the virtual training records by linear interpolation for the minority class. 
# - These synthetic training records are generated by randomly selecting one or more of the k-nearest neighbors for each example in the minority class.
# - SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
# 
# 
# ![SMOTE_Visualization](data:image/png;base64,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)

# In[38]:


train_set["ViolentCrimesPerPop"].value_counts()


# # ```Imbalanced Train Data```

# In[39]:


sns.countplot(x='ViolentCrimesPerPop',data = train_set)
plt.show()


# In[40]:


pip install imblearn


# In[41]:


import collections
x = train_set.drop(columns='ViolentCrimesPerPop')
y = train_set["ViolentCrimesPerPop"]
counter = collections.Counter(y)
print(counter.items())


# In[42]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_smt, y_smt = smote.fit_resample(x,y)
y_smt = pd.DataFrame(y_smt)


# In[43]:


# updating the train_set, to a balanced train_set after using smote
train_set_smote = pd.concat([x_smt,y_smt],axis=1)
train_set = train_set_smote
print(len(train_set))


# In[44]:


train_set


# # ```Balanced Train Data```

# In[45]:


# now we have balanced train_set
sns.countplot(x='ViolentCrimesPerPop',data = train_set)
plt.show()


# ## ```Box Plots for checking correlation between numerical features and target```
# 
# - Box-plot interpretation, these plots gives us idea about distribution of numerical features for each of the classes in target variable, if the distribution looks similar that means the numerical feature has no effect on target variable, hence numerical feature is not correlated to target feature.
# 
# 
# - We can see that for feature "PctBornSameState" rectangular boxes are in similar lines. It means whether the crime rate was low or high is not effected by this feature.
# 
# 
# - Whereas for features "PctKids2Par" and "PctIlleg" we can see that rectangular boxes are in different lines, which affect the crime rate.
# 
# 
# - We will find which all features to drop after performing ANOVA Test.

# In[46]:


def make_box_plot(dataset,feature):    
    low_crime_rate = dataset[dataset['ViolentCrimesPerPop'] == 1][feature]
    medium_crime_rate = dataset[dataset['ViolentCrimesPerPop'] == 2][feature]
    high_crime_rate = dataset[dataset['ViolentCrimesPerPop'] == 3][feature]
    very_high_crime_rate = dataset[dataset['ViolentCrimesPerPop'] == 4][feature]
    
    fig = plt.figure(figsize =(7,6))
    fig = plt.figure(figsize =(7,6))
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.yaxis.grid(True, linestyle='-', which='major', color='grey',alpha=0.5)
    
    plt.boxplot([low_crime_rate, medium_crime_rate, high_crime_rate, very_high_crime_rate], labels=['low', 'medium', 'high', 'very high'],vert=True)
    plt.ylabel(feature)
    plt.title(f"Boxplot of {feature} for ViolentCrimesPerPop")
    plt.show()


# In[47]:


make_box_plot(train_set,"PctBornSameState")


# In[48]:


make_box_plot(train_set,"PctKids2Par")


# In[49]:


make_box_plot(train_set,"PctIlleg")


# # ```Feature Selection```

# # ANOVA Test
# 
# ANOVA (Analysis of Variance) Test is used for feature selection when input variables are numerical and output variable is categorical.
# 
# The variance of a feature determines how much it is impacting the response variable. If the variance is low, it implies there is no impact of this feature on response and vice-versa.
# 
# F-Distribution is a probability distribution generally used for the analysis of variance.
# 
# H0: Two variances are equal
# 
# H1: Two variances are not equal
# 
# F-value is the ratio of two Chi-distributions divided by its degrees of Freedom.
# 
# \begin{equation}
# \ F = (\chi^2_{1}/n_{1}-1) / (\chi^2_{2}/n_{2}-1)
# \end{equation}
# 
# where $\chi^2_{1}$ and $\chi^2_{2}$ are chi distributions and $n_{1}$, $n_{2}$ are its respective degrees of freedom
# 
# Analysis of Variance is a statistical method, used to check the means of two or more groups that are significantly different from each other. It assumes Hypothesis as
# 
# H0: Means of all groups are equal (There is no relation between the given variables as the mean values of the numeric Predictor variable is same for all the groups in the categorical target variable)
# 
# H1: At least one mean of the groups are different (There is some relation between given variables)
# 
# ```SST``` represents the total variation in the data. It is calculated as the sum of the squared differences between each data point and the overall mean of the data.
# 
# <center>SST $= \sum(y_{i} - 𝑦̄)^2 $ </center>
# 
# ```SSB``` represents the variation between different groups or categories in your data. It is calculated as the sum of the squared differences between the group means and the overall mean, weighted by the number of data points in each group.
# 
# <br>
# <center>SSB $= \sum(n_{j}(𝑦̄_{j} - 𝑦̄)^2) $ </center>
# 
# ```SSE``` represents the variation within each group or category. It is calculated as the sum of the squared differences between individual data points and their respective group means.
# 
# <center> SSE $= \sum\sum((y_{ij} - 𝑦̄_{j})^2) $ </center>
# 

# In[50]:


# defining anova function
def anova_feature_selection(X,y):
    n_features = X.shape[1]
    f_scores = [] # intializing array to store f value for each feature
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # to find value of f score for each feature
    for i in range(n_features):
        # feature_values is a column vector
        feature_values = [X.iloc[j][i] for j in range(X.shape[0])]
        
        ssb = 0
        sse = 0
        
        for c in unique_classes:
            # it will take datapoints belonging to class c, lets say n_j points
            class_values = []
            for j in range(len(y)):
                if y[j] == c:
                    class_values.append(feature_values[j])
                    
            # calculating class mean for class c
            class_mean = np.mean(class_values)
            
            # class size as n_j
            class_size = len(class_values)
            
            sse += np.sum([(x-class_mean)**2 for x in class_values])
            ssb += class_size * (class_mean - np.mean(feature_values))**2
        
        # for handling the case when division is undefined 0/0 form
        if sse == 0:
            epsilon = 1e-6
            sse += epsilon
            
        f_scores.append((ssb/(n_classes-1))/(sse/(X.shape[0]-n_classes)))
    
    return f_scores


# In[51]:


x = train_set.drop(columns='ViolentCrimesPerPop')
y = train_set["ViolentCrimesPerPop"]
print("Number of numerical features before applying ANOVA test is: ",len(x.columns))


# In[52]:


# this will give f-score values for all features
f_scores = anova_feature_selection(x,y)
f_scores[:10]


# In[53]:


import json
f_scores = pd.Series(f_scores,index=x.columns)
f_scores_dict = {}
for ind in range(len(f_scores)):
    f_scores_dict[train_set.columns[ind]] = f_scores[ind]
f_scores_dict = dict(sorted(f_scores_dict.items(), key = lambda x: x[1], reverse = True))
print(json.dumps(f_scores_dict, indent = 4))


# In[54]:


# plot showing f values for numerical features
f_scores.sort_values(ascending=False,inplace =True)
f_scores.plot.bar()
plt.show()


# ### According to ANOVA test we reject H0 if 
# 
# \begin{equation}
# F_{\text{observed}}  \ge  F_{\text{critical}}
# \end{equation}
# 
# where, 
# \begin{equation}
# F_{\text{critical}} \space\space  \text{is calculated by taking degree of freedom as k-1,n-k where k is the number of classes (k=4) at level of significance ${\alpha}$ = 0.05}
# \end{equation}
# 
# \begin{equation}
# F_{\text{critical}} \space\space  \text{comes out to be 2.6069}
# \end{equation}
# 
# so for features where,
# 
# \begin{equation}
# F_{\text{observed}} \space\space  \text{is greater than or equal to 2.6069 are included, rest of the features are dropped}
# \end{equation}

# In[55]:


# so features with high f values are included,
# and we drop features which have f values less than 2.6069
count_of_drop_features = 0
for f_val in f_scores:
    if f_val < 2.6069:
        count_of_drop_features += 1
        
print("Number of features that should be dropped are:",count_of_drop_features)


# In[56]:


num_features_included = len(f_scores) - count_of_drop_features
f_scores = f_scores[:num_features_included]
print("Number of numerical features remaining after applying f test are:",len(f_scores))


# In[57]:


train_set_new = pd.DataFrame()

columns_to_copy = []
# including remaining numerical features
count = 0
for feature in f_scores_dict:
    if count == num_features_included:
        break
    columns_to_copy.append(feature)
    count += 1

train_set_new = pd.concat([train_set[columns_to_copy].copy(),train_set["ViolentCrimesPerPop"]],axis=1)
train_set = train_set_new
print(train_set.shape)
features_after_f_test = train_set.columns
features_after_f_test


# In[58]:


train_set


# In[59]:


features_after_f_test


# # ``Outlier Detection``
# 
# ### What is an Outlier?
# 
# - A data point which is significantly far from other data points
# - Inter-Quartile Range Method to remove Outliers (IQR)
# - IQR = Q3 - Q1
# - Upper_Limit = Q3 + 1.5*IQR
# - Lower_Limit = Q1 - 1.5*IQR

# In[60]:


def plot_boxplot(dataframe,feature):
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green',marker='D',markeredgecolor='green')
    dataframe.boxplot(column=[feature],flierprops = red_circle,showmeans=True,meanprops=mean_shape,notch=True)
    plt.grid(False)
    plt.show()


# ## ```Plotting Individual Box Plots```

# In[61]:


plot_boxplot(train_set,"ViolentCrimesPerPop")


# ## ```Plotting Box Plot for multiple features (before outlier removal)```

# In[62]:


# plotting box plots for first 8 features
def plot_boxplot_multiple_features():
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green',marker='D',markeredgecolor='green')
    fig,axis = plt.subplots(1,len(train_set.iloc[:,:8].select_dtypes(include='number').columns),figsize=(20,10))
    for i,ax in enumerate(axis.flat):
        ax.boxplot(train_set.iloc[:,:8].select_dtypes(include='number').iloc[:,i],flierprops=red_circle,showmeans=True,meanprops=mean_shape,notch=True)
        ax.set_title(train_set.iloc[:,:8].select_dtypes(include='number').columns[i][0:20]+"..",fontsize=15,fontweight=20)
        ax.tick_params(axis='y',labelsize=14)

    plt.tight_layout()


# In[63]:


# red circles are the outliers 
plot_boxplot_multiple_features()


# In[64]:


train_set


# In[65]:


# function to return list of indices which are outliers for that feature
def find_outlier_IQR(dataframe,feature):
    q1 = dataframe[feature].quantile(0.25)
    q3 = dataframe[feature].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr
    outlier_indices = dataframe.index[(dataframe[feature] < lower_limit) | (dataframe[feature] > upper_limit)]
    return outlier_indices


# In[66]:


# creating a list to store indices of outliers, for all features
outlier_index_list = []
for feature in train_set.select_dtypes(include='number').columns:
    # skipping the target variable
    if feature == "ViolentCrimesPerPop":
        continue
    outlier_index_list.extend(find_outlier_IQR(train_set.select_dtypes(include='number'),feature))


# In[67]:


# checking the outlier list
print(len(set(outlier_index_list)))
print(outlier_index_list[:10])


# In[68]:


# function to remove outliers and which will return a clean datafram without the outliers
def remove_outliers(dataframe,outlier_index_list):
    outlier_index_list = sorted(set(outlier_index_list)) # use a set to remove duplicate values of indices
    dataframe = dataframe.drop(outlier_index_list)
    return dataframe


# In[69]:


print(len(train_set))


# In[70]:


train_set = remove_outliers(train_set,outlier_index_list)


# In[71]:


# checking the len after outlier removal
print(len(train_set))


# ## ```Plotting Box Plot for multiple features (after outlier removal)```

# In[72]:


plot_boxplot_multiple_features() # we can observe the difference now


# In[73]:


train_set.shape


# In[74]:


train_set.head()


# # ```Feature Scaling```
# 
# ### Standardization Method
# - Standardization is performed to transform the data to have a mean of 0 and standard deviation of 1
# - Standardization is also known as Z-Score Normalization
# 
# \begin{equation}
# z = \frac{(x-\mu)}{\sigma}
# \end{equation}

# In[75]:


# function for finding mean of a feature in a given dataset
def find_mean(dataset,feature):
    n = len(dataset[feature])
    sum = 0
    for val in dataset[feature]:
        sum += val
    return sum/n


# In[76]:


# function for finding standard deviation of a feature in a given dataset
def find_standard_deviation(dataset,feature):
    variance, squared_sum = 0,0
    n = len(dataset[feature])
    mean = find_mean(dataset,feature)
    for val in dataset[feature]:
        squared_sum += (val-mean)**2
    variance = squared_sum/n
    return math.sqrt(variance)


# In[77]:


# function for scaling a feature in given dataset
def standardize_feature(dataset,feature,mean_value,standard_deviation_value):
    # to check if mean and standard deviation values are given incase of test dataset or have to be calculated for train dataset
    if mean_value == -1:
        mean_value = find_mean(dataset,feature)
    if standard_deviation_value == -1:
        standard_deviation_value = find_standard_deviation(dataset,feature)

    standardized_feature = []
    for val in dataset[feature]:
        if standard_deviation_value == 0:
            standard_deviation_value = 1e6
        standardized_feature.append((val-mean_value)/standard_deviation_value)
    return standardized_feature, mean_value, standard_deviation_value


# In[78]:


# function for scaling (standardizing) the whole dataset
def standardize_dataset(dataset,mean_array,standard_deviation_array):
    standardized_df = pd.DataFrame()
    mean_calculated = []
    standard_deviation_calculated = []
    for ind in range(len(dataset.columns)):
        standardized_result, m, sd = standardize_feature(dataset,dataset.columns[ind],mean_array[ind],standard_deviation_array[ind])
        standardized_df[dataset.columns[ind]] = standardized_result
        mean_calculated.append(m)
        standard_deviation_calculated.append(sd)
    return standardized_df, mean_calculated, standard_deviation_calculated


# # ```Plot showing distribution of features before standardization```

# In[79]:


x_axis_limits = (0,10)
sns.displot(train_set.select_dtypes(include='number'), kind='kde',aspect=1,height=8,warn_singular=False,facet_kws={'xlim': x_axis_limits})
plt.show()


# # ```Standardizing the train dataset```

# In[80]:


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# In[81]:


# standardizing the train dataset
train_set_new = train_set.select_dtypes(include='number')
train_set_new = train_set_new.drop(columns = ['ViolentCrimesPerPop'])
train_mean_array = [-1 for i in range(len(train_set_new.columns))]
train_standard_deviation_array = [-1 for i in range(len(train_set_new.columns))]
train_set_new, mean_calculated, standard_deviation_calculated = standardize_dataset(train_set_new,train_mean_array,train_standard_deviation_array)
train_set_new


# In[82]:


# checking mean and variance of each feature after standardizing the train dataset
for feature in train_set_new:
    print("Mean of",feature,"is",round(find_mean(train_set_new,feature)))
    print("Standard Deviation of",feature,"is",round(find_standard_deviation(train_set_new,feature)))


# # ```Plot showing distribution of features after standardization```

# In[83]:


# all features following a normal distribution with mean 0 and standard deviation of 1
sns.displot(train_set_new, kind='kde',aspect=1,height=8)
plt.show()


# In[84]:


# replacing standardized features with original features in train_set dataframe
train_set_new.index = train_set.index
train_set_new["ViolentCrimesPerPop"] = train_set["ViolentCrimesPerPop"]
train_set = train_set_new
train_set


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

# In[85]:


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

k = 36

# this will return the new dataset
train_set_pca, explained_variance, k_principal_components = PCA(train_set.drop(columns="ViolentCrimesPerPop"),k)
train_set_pca = pd.DataFrame(train_set_pca)
print("Shape of train_set is:", train_set.shape)
print("Shape of train_set_pca is:",train_set_pca.shape)


# In[86]:


train_set_pca


# In[87]:


# plotting first two principal components
PC1 = train_set_pca[0]
PC2 = train_set_pca[1]

target = train_set["ViolentCrimesPerPop"]
label = ["low","medium","high","very high"]

labels = []
for points in target:
    labels.append(label[int(points)-1])
    
zipped = list(zip(PC1,PC2,target,labels))
pc_df = pd.DataFrame(zipped, columns=['PC1','PC2','Target','Label'])
pc_df


# ## ```Plot showing spread of data along first 2 Principal Components```

# In[88]:


plt.figure(figsize=(12,7))
forest_green = (34/255, 139/255, 34/255)
gold = (255/255,215/255,0/255)
colors = [forest_green if label == 0 else gold for label in target]
sns.scatterplot(data=pc_df,x="PC1",y="PC2",hue="Label",palette={'low':gold,'medium':forest_green,'high':'blue',"very high":'red'})
plt.title("Scatter Plot",fontsize=16)
plt.xlabel('First Principal Component',fontsize=16)
plt.ylabel('Second Principal Component',fontsize=16)
plt.show()


# ## ```Plot showing spread of data along first 3 Principal Components```

# In[89]:


PC3 = train_set_pca[2]

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111,projection='3d')
 
for l in np.unique(target):
    ix = np.where(target==l)
    ix = np.asarray(ix)[0]
    ax.scatter(PC1[ix],PC2[ix],PC3[ix],label=label[int(l)-1],c=[gold if int(l) == 1 else forest_green if int(l) == 2 else 'blue' if int(l) == 3 else 'red'])

ax.set_xlabel("PC1",fontsize=12)
ax.set_ylabel("PC2",fontsize=12)
ax.set_zlabel("PC3",fontsize=12)
 
ax.view_init(30, 125)
ax.legend()
plt.title("3D Plot",fontsize=16)
plt.show()


# ## ```Plot showing variance captured by each Principal Component```

# In[90]:


num_components = len(explained_variance)
components = np.arange(1, num_components + 1)
plt.figure(figsize=(10, 8))
plt.plot(components, explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.xticks(components)
plt.show()


# ## ```Plot to find out number of Principal Components needed inorder to capture 95% variance in data```

# In[91]:


# finding cumulative variance captured by principal components
y_var = np.cumsum(explained_variance)

plt.figure(figsize=(25,15))
plt.ylim(0.0,1.1)
plt.plot(components, y_var, marker='o', linestyle='-', color='green')

plt.xlabel('Number of Components',fontsize=16)
plt.ylabel('Cumulative variance (%)',fontsize=16)
plt.title('The number of components needed to explain variance',fontsize=18)
plt.xticks(components) 

plt.axhline(y=0.95, color='red', linestyle='--')
plt.axhline(y=1.00, color='orange', linestyle='--')
plt.axvline(x=36.00, color='blue', linestyle='--')
plt.text(1, 0.95, '95% cut-off threshold', color = 'black', fontsize=16)
plt.text(1, 1, '100% variance if all features are included', color = 'black', fontsize=16)

plt.tight_layout()
plt.show()


# In[92]:


# therefore for capturing 95% of variance in dataset we should take first 30 principal components
# adding the output or target attribute to train_set
train_set_pca.index = train_set.index
train_set_pca["ViolentCrimesPerPop"] = train_set["ViolentCrimesPerPop"]
train_set_pca
train_set = train_set_pca
train_set.shape


# In[93]:


train_set


# ## ```Correlation Matrix for Train Dataset```
# 
# - We can see that we have transformed our train dataset into unrelated components by using PCA, as correlation between new features is almost 0

# In[94]:


corr = train_set.corr()
plt.subplots(figsize=(12,8))
corr_new = corr[corr>=0.00001]
sns.heatmap(corr_new)
plt.show()


# # ```Modifying Test Data```

# In[95]:


# original test dataset
test_set


# ## ```Transforming test data into same feature space as train data```

# In[96]:


# removing features, which were eliminated after ANOVA test
print(features_after_f_test)


# In[97]:


# remove features other than those left after ANOVA test
test_set_new = pd.DataFrame()
for feature in features_after_f_test:
    test_set_new[feature] = test_set[feature]
test_set = test_set_new
test_set


# # ```Feature Scaling for Test Data```

# ## ```Plot showing distribution of features before standardization```

# In[98]:


x_axis_limits = (0,10)
sns.displot(test_set.select_dtypes(include='number'), kind='kde',aspect=1,height=8,warn_singular=False,facet_kws={'xlim': x_axis_limits})
plt.show()


# # ```Standardizing the test dataset```
# 
# - We are standardizing test data set with mean and standard deviation values obtained from trained dataset

# In[99]:


# standardizing the test dataset
test_set_new = test_set.select_dtypes(include='number')
test_set_new = test_set_new.drop(columns = ['ViolentCrimesPerPop'])
test_set_new, mean_calculated, standard_deviation_calculated = standardize_dataset(test_set_new,mean_calculated,standard_deviation_calculated)
test_set_new


# In[100]:


# checking mean and variance of each feature after standardizing the test dataset
# here mean of all features might not be 0 because we have standardized using mean of train dataset
for feature in test_set_new:
    print("Mean of",feature,"is",round(find_mean(test_set_new,feature)))
    print("Standard Deviation of",feature,"is",round(find_standard_deviation(test_set_new,feature)))


# ## ```Plot showing distribution of features after standardization```

# In[101]:


# all features following a normal distribution with mean 0 and standard deviation of 1
sns.displot(test_set_new, kind='kde',aspect=1,height=8)
plt.show()


# In[102]:


# replacing standardized features with original features in test_set dataframe
test_set_new.index = test_set.index
test_set_new["ViolentCrimesPerPop"] = test_set["ViolentCrimesPerPop"]
test_set = test_set_new
test_set


# # ```PCA for test data```

# In[103]:


# taking projection of test data on principal component vectors,
# to get same number of features in test data as of train data
test_set_pca = np.dot(test_set.drop(columns="ViolentCrimesPerPop"),k_principal_components.T)
test_set_pca = pd.DataFrame(test_set_pca)
test_set_pca.index = test_set.index
test_set_pca["ViolentCrimesPerPop"] = test_set["ViolentCrimesPerPop"]
test_set_pca


# In[104]:


# finally we have test_set in same dimension as our train_set
test_set = test_set_pca
test_set.shape


# # ```Implementing Machine Learning Models```
# 
# - Decision tree model with entropy implementation
# - Adaboost
# - Multiclass SVM

# ## ```Classification Report```
# 
# - True Positive (TP): Predictions which are predicted by classifier as ```positive``` and are actually ```positive```
# 
# 
# - False Positive (FP): Predictions which are predicted by classifier as ```positive``` and are actually ```negative```
# 
# 
# - True Negative (TN): Predictions which are predicted by classifier as ```negative``` and are actually ```negative```
# 
# 
# - False Negative (FN): Predictions which are predicted by classifier as ```negative``` and are actually ```positive```
# 
# 
# 
# 
# - ```Accuracy```
# 
#     It describes the number of correct predictions over all predictions.
#       
# $$
# \frac{TP+TN}{TP+FP+TN+FN} = \frac{\text{Number of correct  predictions}}{\text{Number of total predictions}}
# $$
#     
# - ```Precision```
# 
#     Precision is a measure of how many of the positive predictions made are correct.
#     
# $$
# \frac{TP}{TP+FP} = \frac{\text{Number of correctly predicted positive instances}}{\text{Number of total positive predictions made}}
# $$
# 
# - ```Recall```
# 
#     Recall is a measure of how many of the positive cases the classifier correctly predicted, over all the positive cases in the data.
#     
# $$
# \frac{TP}{TP+FN} = \frac{\text{Number of correctly predicted positive instances}}{\text{Number of total positive instances in dataset}}
# $$
# 
# - ```F1 Score```
# 
#     F1-Score is a measure combining both precision and recall. It is harmonic mean of both.
# 
# $$
# \frac{2*{\text{Precision}}*{\text{Recall}}}{\text{Precision}+\text{Recall}}
# $$
# 
# - ```Support```
# 
#     Support may be defined as the number of samples of the true response that lies in each class of target values  
# $$
# $$
# - ```Confusion Matrix```
#     
#     It is a table with 4 different combinations of predicted and actual values.
#     
#     
#     
# 
#                         |                 | Predicted Positive | Predicted Negative  |
#                         |-----------------|--------------------|---------------------|
#                         | Actual Positive | True Positives     | False Negatives     |
#                         | Actual Negative | False Positives    | True Negatives      |
# 
# - ```ROC Curve```
# 
#     The ROC curve is produced by calculating and plotting the true positive rate against the false positive rate  for a single classifier at a variety of thresholds.
#     
#     The Area Under the Curve (AUC) is the measure of the ability of a binary classifier to distinguish between classes and is used as a summary of the ROC curve.The higher the AUC, the better the model’s performance at distinguishing between the positive and negative classes.
# 
# $$
#     \text{True Positive Rate (TPR) }  =  \frac{TP}{TP+FN}
# $$
# 
# $$
#     \text{False Positive Rate (FPR) }  =  \frac{FP}{FP+TN}
# $$
# 
# $$
# $$

# In[105]:


# Machine Learning Algorithms
MLA = {} # dictionary for comparing models
MLA_predictions = {} # dictionary for comparing ROC curve of models


# In[106]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize

# function to show performance metrics for classification models
def performance_metrics(name,y_test,y_pred):
    # finding accuracy score
    accuracy = accuracy_score(y_test, y_pred)
   
    # calculating classification report
    clf_report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True))
    print("Test Result:\n================================================")        
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}\n")
   
    # confusion matrix
    sns.heatmap(confusion_matrix(y_test,y_pred),fmt='',cmap = 'YlGnBu')
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
   
    MLA[name] = [name,accuracy_score(y_test, y_pred)*100,precision_score(y_test,y_pred,average='weighted'),recall_score(y_test,y_pred,average='weighted')]
    MLA_predictions[name] = y_pred

    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred_bin = label_binarize(y_pred, classes=np.unique(y))
   
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y_test_bin.shape[1]
   
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
   
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
   
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
   
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i + 1, roc_auc[i]))
   
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multiple Classes')
    plt.legend(loc="lower right")
    plt.show()


# ## ```Splitting train and test data into x and y components```

# In[107]:


x_train = train_set.drop(columns='ViolentCrimesPerPop')
y_train = train_set['ViolentCrimesPerPop']
x_test = test_set.drop(columns='ViolentCrimesPerPop')
y_test = test_set['ViolentCrimesPerPop']


# In[108]:


# converting column names in train and test dataset to strings


# In[109]:


new_column_names = []
for val in train_set.columns:
    new_column_names.append(str(val))
train_set.columns = new_column_names
test_set.columns = new_column_names


# # ```2. Decision tree model with entropy implementation```
# 
# - Decision Tree is a Supervised learning technique that can be used for classification problems.
# 
# 
# - A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.
# 
# 
# - Decision trees can model nonlinear relationships in the data, capturing complex patterns that linear models may miss.
# 
# 
# - Decision trees are relatively robust to irrelevant features, as they are less likely to be selected for splitting.
# 
# 
# - Decision trees are prone to overfitting, especially when the tree is deep. Overfitting occurs when the model captures noise in the training data, leading to poor generalization on new, unseen data.

# ## ```2.1 Implementation of the Model```

# In[110]:


# creating validation data set, which will be used in pruning of decision tree
train_set_new, val_set = split_train_test(train_set,0.2)


# In[111]:


train_set_new.shape


# In[112]:


val_set.shape


# In[113]:


x_train_new = train_set_new.drop(columns='ViolentCrimesPerPop')
y_train_new = train_set_new['ViolentCrimesPerPop']


# ## ```Helper functions for Decision Tree```

# In[114]:


import random
from pprint import pprint


# In[115]:


# function to check if y_values are pure i.e all belong to a single class or not
def check_purity(y_values):
    y_values = np.array(y_values)
    unique_classes = np.unique(y_values)
    if(len(unique_classes) == 1):
        return True
    return False


# In[116]:


# function to classify data based on majority count in y_values
def classify_data(y_values):
    y_values = np.array(y_values)
    unique_classes, counts_unique_classes = np.unique(y_values, return_counts = True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification


# In[117]:


# function to find all possible potential splits for all features
def get_potential_splits(x_train,random_subspace=None):
    # random_subspace is number of features that we will randomly select from all features
    potential_splits = {} # dictionary that will contain splits for each feature
    num_columns = x_train.shape[1]
    
    # for creating a list from 0 to num_columns
    column_indices = list(range(num_columns))
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(column_indices,random_subspace)
    
    # for each feature we will have multiple splits
    for column_index in column_indices:
        potential_splits[column_index] = []
        values = x_train.iloc[:,column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values

    return potential_splits


# In[118]:


# split data function
def split_data(train_dataset,split_column,split_value):
    # this will take all values for that particular column
    split_column_values = train_dataset.iloc[:,split_column]
    
    # splitting data into binary split
    left_child_data = train_dataset[split_column_values <= split_value]
    right_child_data = train_dataset[split_column_values > split_value]

    return left_child_data, right_child_data


# ### ```Entropy```
# 
# Entropy is a measurement of uncertainty (sometimes called impurity) has the following formula
# 
# $$
# H(X) = - \sum_{j} p_{j} log_{2}(p_{j})
# $$
# 
# \begin{equation}
# \text{where} \space {p_{j} \space \text{is the probability of class j}}
# \end{equation}
# 
# For two class problem, it takes the following form
# 
# $$
# H(X) = - plog_{2}(p)  -(1-p)log_{2}(1-p)
# $$
# 
# 
# When we are looking for a value to split our decision tree node, we want to minimize entropy.

# In[119]:


# entropy function, to find impurity measure for parent node
def calculate_entropy(train_dataset):
    # to find number of samples belonging to class 1 and class 2
    y_train = np.array(train_dataset.iloc[:,-1])
    _, counts = np.unique(y_train,return_counts=True)
    total_samples = counts.sum()
    
    # this will contain p_j i.e probability of class j, where j=1,2
    probability_of_class = counts / total_samples
    entropy = -sum((probability_of_class * np.log2(probability_of_class)))
    
    return entropy


# ### ```Overall Entropy```
# 
# Overall Entropy for children nodes, can be calculated using the formula
# 
# $$
# H(children) = \sum_{j}^{2}( \frac{n_{j}}{N} \cdot H_{j})
# $$
# 
# $$
#   \text{where} \space n_{j} \space \text{is number of training samples associated with node j such that}
# $$
# 
# $$
# N = \sum_{j}^{2} n_{j}
# $$
# 
# $$
#   \text{and} \space H_{j} \space\text{is entropy of child node j}
# $$

# In[120]:


# overall entropy function
def calculate_overall_entropy(left_child_data,right_child_data):
    n1 = len(left_child_data) # number of data points belonging to left child
    n2 = len(right_child_data) # number of data points belonging to right child
    N = n1 + n2 # total number of data points
    overall_entropy = ((n1/N)*calculate_entropy(left_child_data)) + ((n2/N)*calculate_entropy(right_child_data))
    return overall_entropy


# ### ```Information Gain```
# 
# Information gain is a measure which tells how much information we have gained after splitting a node at a particular value.
# 
# $$
#    \text{Information Gain} = H(Parent) - H(Children)
# $$
# 
# $$
# \text{where} \space H(Parent) \space\text{is entropy of Parent node or entropy before split and} \space H(Children) \space\text{is overall entropy of both left and right child nodes or entropy after split.}
# $$
# 
# We need to find the split, which results in highest information gain.

# In[121]:


# function for finding information gain
def calculate_information_gain(train_dataset,left_child_data,right_child_data):
    return calculate_entropy(train_dataset) - calculate_overall_entropy(left_child_data,right_child_data)


# In[122]:


# function for finding best split, by taking highest information gain
def find_best_split(train_dataset):
    information_gain = -1e9
    
    if(len(train_dataset.columns) == 31):
        x_train = train_dataset.drop(columns="ViolentCrimesPerPop")
    else:
        x_train = train_dataset
        
    # find all possible splits
    potential_splits = get_potential_splits(x_train)

    # iterating over all splits to find best split, which gives minimum overall entropy
    for column_index in potential_splits:
        for value in potential_splits[column_index]:

            # this will give data points for left and right child
            left_child_data, right_child_data = split_data(x_train,column_index,value)
            information_gain_current = calculate_information_gain(train_dataset,left_child_data,right_child_data)

            if information_gain_current > information_gain:
                information_gain = information_gain_current
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# # ```Decision Tree Algorithm```
# 
# - We will represent our decision tree in form of a dictionary {}, where {key:[yes,no]} represents a node, in which key will be a question and yes corresponds to left child node and no corresponds to right child node
# 
# - Also yes and no can contain dictionaries, which will represent the left and right subtrees
# 
# ### ```Pre Pruning of Decision Tree```
# 
# Pre pruning is nothing but stopping the growth of decision tree on an early stage before it perfectly fits the entire training data. For that we can limit the growth of trees by setting constrains. We can limit parameters like max_depth, min_samples etc.

# In[123]:


def decision_tree_algorithm(train_dataset,counter=0,min_samples=2,max_depth=5,random_subspace=None):
    
    # converting train_dataset to numpy 2d array
    if counter == 0:
        global column_headers
        column_headers = train_dataset.columns
        # first call of this function, so convert train_dataset to numpy 2d array
        data = train_dataset.values
    else:
        data = train_dataset
    
    x_train = train_dataset.drop(columns="ViolentCrimesPerPop")
    y_train = train_dataset.iloc[:,-1]
    
    # base case, if node is pure, or contains less than min_samples or depth of tree has reached max_depth so stop
    if (check_purity(y_train)) or (len(train_dataset) < min_samples) or (counter == max_depth):
        classification = classify_data(y_train)
        return classification
    
    else:
        # recursive part of decision tree algorithm
        counter += 1 # increment the depth of decision tree

        # helper functions
        potential_splits = get_potential_splits(x_train,random_subspace)
        split_column, split_value = find_best_split(x_train)
        left_child_data, right_child_data = split_data(train_dataset,split_column,split_value)
        
        if len(left_child_data) == 0 or len(right_child_data) == 0:
            classification = classify_data(y_train)
            return classification
        
        # instantiate sub tree
        feature_name = column_headers[split_column]
        question = "{} <= {}".format(feature_name,split_value)
        sub_tree = {question : []}
        
        # find answers
        yes_answer = decision_tree_algorithm(left_child_data,counter,min_samples,max_depth,random_subspace)
        no_answer = decision_tree_algorithm(right_child_data,counter,min_samples,max_depth,random_subspace)
        
        # if both answers are same then, assign sub_tree as yes_answer instead of left and right child
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


# In[124]:


tree = decision_tree_algorithm(train_set_new,counter=0,min_samples=2,max_depth=10)
tree


# In[125]:


# classify example
def classify_example(example,tree):
    question = list(tree.keys())[0]
    feature_name, comparision_operator, value = question.split()
    
    # ask a question
    if example[int(feature_name)] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
        
    # base case, answer is a leaf node
    if not isinstance(answer,dict):
        return answer
    else:
        # recursive part
        residual_tree = answer
        return classify_example(example,residual_tree)


# In[126]:


def make_predictions(x_test,tree):
    # find predicted values for each test data point
    predicted_output = []
    for ind in range(len(x_test)):
        predicted_output.append(classify_example(x_test.iloc[ind],tree))
    
    return predicted_output


# In[127]:


def calculate_accuracy(y_predicted,y_test):
    # finding accuracy
    correct_predictions = 0
    for ind in range(len(y_test)):
        if y_test.iloc[ind] == y_predicted[ind]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(y_test)
    return accuracy


# In[128]:


y_predicted = make_predictions(x_test,tree)
calculate_accuracy(y_predicted,y_test)*100


# ## ```Post Pruning in Decision Tree```
# 
# - This is done after construction of decision tree, pruning is done in a bottom-up fashion.
# 
# 
# - It is done by replacing a subtree with a new leaf node with a class label which is majority class of records.
# 
# 
# - We are using validation dataset for post pruning of decision tree

# In[129]:


def filter_df(df,question):
    feature, _ , value = question.split()
    df_yes =  df[df[feature] <= float(value)]
    df_no =  df[df[feature] > float(value)]
    
    return df_yes, df_no


# In[130]:


def pruning_result(tree,x_train,y_train,x_val,y_val):
    
    # setting leaf with value as majority label in train set
    leaf = y_train.value_counts().index[0]
    errors_leaf = sum(y_val != leaf)
    errors_decision_node = sum(y_val != make_predictions(x_val,tree))

    # returning leaf node or subtree whichever has less error
    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree


# In[131]:


def post_pruning(tree,train_set,val_set):
    
    # find question for root node of tree
    question = list(tree.keys())[0]
    
    # this will give left and right subtrees
    yes_answer, no_answer = tree[question]
    
    x_train = train_set.drop(columns='ViolentCrimesPerPop')
    y_train = train_set['ViolentCrimesPerPop']
    x_val = val_set.drop(columns='ViolentCrimesPerPop')
    y_val = val_set['ViolentCrimesPerPop']

    # base case
    # both left and right subtrees are leaf nodes, so either it will return the same tree, or replace this tree with a leaf node
    if not isinstance(yes_answer,dict) and not isinstance(no_answer,dict):
        return pruning_result(tree,x_train,y_train,x_val,y_val)
        
    # recursive part
    else:
        df_train_yes, df_train_no = filter_df(train_set,question)
        df_val_yes, df_val_no = filter_df(val_set,question)
        
        # left subtree, this will do post pruning on left subtree recursively
        if isinstance(yes_answer,dict):
            yes_answer =  post_pruning(yes_answer,df_train_yes,df_val_yes)
            
        # right subtree, this will do post pruning on right subtree recursively
        if isinstance(no_answer,dict):
            no_answer = post_pruning(no_answer,df_train_no,df_val_no)
        
        # new tree, so that it doesn't replace the existing tree
        tree = {question:[yes_answer,no_answer]}
        
    # we are calling pruning_result here, because the tree after post_pruning can be replaced by a single leaf node
    return pruning_result(tree,x_train,y_train,x_val,y_val)


# In[132]:


pruned_tree = post_pruning(tree,train_set_new,val_set)
pruned_tree


# ## ```Visualizing Decision Boundaries after Post Pruning```
# 
# - We can see that unpruned tree has overfitting, it overfits the train dataset, but pruned tree prevents overfitting

# In[133]:


def plot_decision_boundaries(tree, x_min, x_max, y_min, y_max):
    forest_green = (34/255, 139/255, 34/255)
    gold = (255/255,215/255,0/255)
    color_keys = {1:"blue",2:"orange",3:"green",4:"red"}
    
    # recursive part
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        yes_answer, no_answer = tree[question]
        feature, _, value = question.split()
    
        if feature == "0":
            plot_decision_boundaries(yes_answer, x_min, float(value), y_min, y_max)
            plot_decision_boundaries(no_answer, float(value), x_max, y_min, y_max)
        else:
            plot_decision_boundaries(yes_answer, x_min, x_max, y_min, float(value))
            plot_decision_boundaries(no_answer, x_min, x_max, float(value), y_max)
        
    # "tree" is a leaf
    else:
        plt.fill_between(x=[x_min, x_max], y1=y_min, y2=y_max, alpha=0.2, color=color_keys[tree])
        
    return


# In[134]:


def create_plot(df,tree=None,title=None):
    sns.lmplot(data=df,x=df.columns[0], y=df.columns[1], hue="ViolentCrimesPerPop", fit_reg=False, height=4, aspect=1.5, legend=False)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["low","medium","high","very high"])
    plt.title(title)
    
    if tree or tree == False: # root of the tree might just be a leave with "False"
        x_min, x_max = round(df["0"].min()), round(df["0"].max())
        y_min, y_max = round(df["1"].min()), round(df["1"].max())

        plot_decision_boundaries(tree, x_min, x_max, y_min, y_max)
    
    return


# In[135]:


# plotting decision boundaries for train_set using unpruned tree
create_plot(train_set_new,tree,"Unpruned Decision Tree")


# In[136]:


# plotting decision boundaries for train_set using pruned decision tree
create_plot(train_set_new,pruned_tree,"Pruned Decision Tree")


# ## ```Accuracy difference before and after pruning decision tree```

# In[137]:


# before pruning decision tree
y_predicted = make_predictions(x_test,tree)
calculate_accuracy(y_predicted,y_test)*100


# In[138]:


# after pruning decision tree
y_predicted = make_predictions(x_test,pruned_tree)
calculate_accuracy(y_predicted,y_test)*100


# ## ```2.2 Insights drawn (plots, markdown explanations)```

# ## ```Comparing Accuracy of Decision Tree for different depths```

# In[139]:


depth_array = [3,5,8,10,15,20,30]
accuracy_array = []
predictions_array = []

for depth_val in depth_array:
    plt.close('all')
    
    # generate the decision tree
    tree = decision_tree_algorithm(train_set_new,counter=0,min_samples=2,max_depth=depth_val)
    # do pruning of decision tree obtained
    pruned_tree = post_pruning(tree,train_set_new,val_set)
    
    # plot decision boundaries
    print("\n================================================\n")
    print("Decision boundary on the training set before and after pruning\n\n")
    
    create_plot(train_set_new,tree,"Unpruned Decision Tree")
    plt.pause(1)
    create_plot(train_set_new,pruned_tree,"Pruned Decision Tree")
    plt.pause(1)
    
    # find accuracy on test dataset after pruning
    y_predicted = make_predictions(x_test,pruned_tree)
    predictions_array.append(y_predicted)
    acc_val = calculate_accuracy(y_predicted,y_test)*100
    
    print(f"Accuracy for decision tree with max depth as {depth_val} is {acc_val}")
    accuracy_array.append(acc_val)


# In[140]:


# to find highest accuracy
plt.plot(depth_array,accuracy_array)
plt.xlabel("Depth of decision tree")
plt.ylabel("Accuracy")
plt.show()


# In[141]:


# finding highest accuracy
max_index = 0
highest_accuracy = accuracy_array[0]

for i in range(len(accuracy_array)):
    if accuracy_array[i] > highest_accuracy:
        highest_accuracy = accuracy_array[i]
        max_index = i

best_predictions = predictions_array[max_index]


# ## ```Visualizing results predicted by Decision Tree with actual values```

# In[142]:


def plot_scatter_plot(values,title):
    for ind in range(len(best_predictions)):
        if(np.array(values)[ind] == 1):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='yellow')
        elif(np.array(values)[ind] == 2):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='green')
        elif(np.array(values)[ind] == 3):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='blue')
        else:
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='red')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["low","medium","high","very high"])
    plt.title(title)
    plt.show()


# In[143]:


# plotting scatter plot for prediction made by Decision Tree using first 2 features of test data set
plot_scatter_plot(best_predictions,"Decision Tree Result")


# In[144]:


# plotting scatter plot for actual values of data points using first 2 features of test data set
plot_scatter_plot(y_test,"Actual Values")


# # ```Classification Report for Decision Tree Model```

# In[145]:


performance_metrics("Decision Tree",y_test,best_predictions)


# # ```3. Adaboost```
# 
# Adaboost is one of the early successful algorithm wihtin the Boosting branch of Machine Learning.
#  
# - AdaBoost for classification is a supervised Machine Learning algo.
# - It consists of iteratively training multiple stumps using feature data $x$ and target label $y$
# - After training, the model's ovearll prefictions come from a weighted sum of the stumps predictions
# 
# - For instances, consider $n$ stumps, $P_i$ be the prediction of $i_{th}$ stump. whose corresponding weight is $w_{i}$ then:
# 
# 
# $$
# \text{Final Adaboost Prediction} = w_{{stump}1}*P{1} + w_{{stump}2}*P{2} + ... + w_{{stump}n}*P{n}
# $$
# 
# 
# ### Algorithm of Adaboost
# 
# - AdaBoost trains a sequence of models with augmented sample weights generating 'confidence' coefficients $\alpha$ for each individual classifiers based on errors.
# 
# 1. For a datset with N number of samples, we initialize weight of each data point with $w_i$ = $\frac{1}{N}$ . The weight of a datapoint represents the probability of selecting it during sampling. 
# 
# 
# 2. Training Weak Clasifiers: After sampling the dataset, we train a weak classifier $K_m$ using the training dataset. Here $K_m$ is just a decision tree stump
# 
# - For $m$ = 1 to $M$: 
# - sample the dataset using the weights $w_{i}^{(m)}$ to obtain the training samples $x_{i}$
# - Fit a classifier $K_m$ using all the training samples $x_i$
# - Here $K_m$ is the prediction of $m_{th}$ stump
# 
# 3. Calcuating Update Parameters from Error: Stumps in AdaBoost progressively learn from previous stump's mistakes through adjusting the weights of the dataset to accomodate the mistakes made by stumps at each iteration. 
# - The weights of the misclassified data will be increased, and the weights of correctly classified data will be decreased. 
# - As a result, as we go into further iterations, the training data will mostly comprise of data that is often misclassified by our stumps. 
# - When data is misclassified, $y*K_m(x) = -1$, then the weight update will be positive and the weights will exponentially increase for that data. When the data is correctly satisfied, $y*K_m(x) = 1$, the opposite happens.
# 
# 
# - Let $\epsilon$ be the ratio of sum of weights for misclassified samples to that of the total sum of weights for samples. Compute the value of $\epsilon$ as follows:
# 
# $$
# \epsilon = \frac{\sum_{y_i \neq K_m(x_i)}w_i^{(m)}}{\sum_{y_i}w_i^{(m)}}
# $$
# 
# 
# - In the above equation, $y_i$ is the ground truth value of the target varibale and $w_i^{(m)}$ is the weight of the sample $i$ at iteration $m$
# - Larger $\epsilon$ means largers misclassification percentage, makes the confidence $\alpha_m$ decrease exponentially.
# - Let confidence that the AdaBoost places on $m_{th}$ stump's ability to classify the data be $\alpha_m$. Compute $\alpha_m$ value as follows:
# 
# 
# $$
# \alpha_{m} = \frac{1}{2}\log{(\frac{1-\epsilon}{\epsilon}})
# $$
# 
# 
# - Update all the weights as follows:
# $$
# w_{i}^{(m+1)} = w_{i}^{(m)}.e^{-\alpha_myK_m(x)} 
# $$
# 
# 
# 4. New predictions can be computed by 
# $$
# K(x) = sign[\sum_{m=1}^{M}\alpha_{m}.K_{m}(x)]
# $$
# 
# - Low error means the $\alpha$ value is larger, and thus has higher importance in voting.
# 
# - AdaBoost predicts on -1 or 1. As long as the weighted sum is above 0, it predicts 1, else, it predicts -1.

# ## ```3.1 Implementation of the Model```

# In[146]:


# class for decision stump
class DecisionStump:
    # constructor will initialize value, feature_ind to split at, and threshold and alpha for this stump
    def __init__(self):
        self.value = 1
        self.feature_ind = None
        self.threshold = None
        self.alpha = None

    # function to predict outcome for this particular stump
    def predict(self,x_train):
        n_samples = x_train.shape[0]
        x_train_column = x_train[:, self.feature_ind]
        predictions = np.ones(n_samples)
        
        if self.value == 1:
            predictions[x_train_column < self.threshold] = -1
        else:
            predictions[x_train_column > self.threshold] = -1

            
        return predictions


# In[147]:


# class for implementing binary adaboost classifier
class Adaboost:
    
    # n_clf are the number of decision stumps, clfs will store each of the classifier
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []
        
    def train_model(self,x_train,y_train):
        n_samples, n_features = x_train.shape

        # initialize weights to 1/N
        w = []
        for i in range(n_samples):
            w.append(1/n_samples)
        w = np.array(w)
            
        # to store each decision stump
        self.clfs = []

        # iterate through classifiers
        for i in range(self.n_clf):

            # initializing a new decision stump
            clf = DecisionStump()
            min_error = float("inf")

            # finding best feature and threshold value
            for feature_i in range(n_features):
                
                x_train_column = x_train[:,feature_i]
                thresholds = np.unique(x_train_column)

                for threshold in thresholds:
                    # predict with value 1
                    p = 1
                    
                    # initializing predictions to 1
                    predictions = np.ones(n_samples)
                    predictions[x_train_column < threshold] = -1

                    # calculating error
                    misclassified = w[y_train != predictions]
                    error = sum(misclassified)/sum(w)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.value = p
                        clf.threshold = threshold
                        clf.feature_ind = feature_i
                        min_error = error

            # calculate alpha
            epsilon = 1e-10 # to prevent divide by 0
            clf.alpha = 0.5 * np.log((1.0 - (min_error + epsilon)) / (min_error + epsilon))

            # calculate predictions and update weights
            predictions = clf.predict(x_train)
            w *= np.exp(-clf.alpha * y_train * predictions)

            # Save classifier
            self.clfs.append(clf)
            
    # function to find output
    def decision_function(self,x_train):
        clf_preds = [clf.alpha * clf.predict(x_train) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return y_pred
    
    # function to classify
    def predict(self, x_train):
        return np.sign(self.decision_function(x_train))
    


# In[148]:


class MulticlassAdaboost:
    
    # initializing classifiers array
    def __init__(self,n_stumps):
        self.classifiers = []
        self.n_stumps = n_stumps

    def train_model(self, x_train, y_train):
        self.classes = np.unique(y_train)
        self.classifiers = []

        for i, class_label in enumerate(self.classes):
            # treating the current class as positive and the rest as negative
            binary_labels = np.where(y_train == class_label, 1, -1)
            
            # create a binary Adaboost classifier
            # n_clf is number of stumps in that classifier
            classifier = Adaboost(n_clf=self.n_stumps)
            
            # training the ith classifier
            classifier.train_model(x_train, binary_labels)
            self.classifiers.append(classifier)
            
    # to predict the output
    def predict(self,x_test):
        
        # for each binary classifier, get the raw decision scores
        decision_scores = np.array([clf.decision_function(x_test) for clf in self.classifiers])

        # Choose the class with the highest decision score as the final prediction
        predictions = np.argmax(decision_scores, axis=0)+1 # to adjust the 0 based indexing
        return predictions


# In[149]:


def find_accuracy(y_true,y_pred):
    count = 0
    for ind in range(len(y_true)):
        if y_pred[ind] == y_true[ind]:
            count += 1
    
    return count/len(y_true)


# # ```Training Multiclass Adaboost Model```

# In[150]:


# creating instance of that class
multiclass_adaboost = MulticlassAdaboost(n_stumps = 5)
multiclass_adaboost.train_model(np.array(x_train), np.array(y_train))


# In[151]:


# finding predictions
y_pred = multiclass_adaboost.predict(np.array(x_test))


# In[152]:


# finding accuracy for this machine learning model
print(find_accuracy(y_test,y_pred))


# ## ```3.2 Insights drawn (plots, markdown explanations)```

# ## ```Comparing Accuracy of Adaboost for different number of decision stumps```

# In[153]:


num_decision_stumps = [1,3,5,7,8,10,15,20,25,30,50]
accuracy_array = []
predictions_array = []

for val in num_decision_stumps:
    # creating object for this class
    multiclass_adaboost = MulticlassAdaboost(n_stumps = val)
    multiclass_adaboost.train_model(np.array(x_train), np.array(y_train))
    
    y_pred = multiclass_adaboost.predict(np.array(x_test))
    predictions_array.append(y_pred)
    
    acc_val = find_accuracy(y_test,y_pred)
    accuracy_array.append(acc_val)
    
    print(f"Accuracy for Adaboost with {val} decision stumps is: {acc_val}\n")


# In[154]:


plt.plot(num_decision_stumps,accuracy_array)
plt.xlabel("Number of decision stumps")
plt.ylabel("Accuracy")
plt.show()


# In[155]:


# finding highest accuracy
max_index = 0
highest_accuracy = accuracy_array[0]

for i in range(len(accuracy_array)):
    if accuracy_array[i] > highest_accuracy:
        highest_accuracy = accuracy_array[i]
        max_index = i

best_predictions = predictions_array[max_index]


# ## ```Visualizing results predicted by Adaboost with actual values```

# In[156]:


def plot_scatter_plot(values,title):
    for ind in range(len(best_predictions)):
        if(np.array(values)[ind] == 1):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='yellow')
        elif(np.array(values)[ind] == 2):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='green')
        elif(np.array(values)[ind] == 3):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='blue')
        else:
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='red')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["low","medium","high","very high"])
    plt.title(title)
    plt.show()


# In[157]:


# plotting scatter plot for prediction made by Adaboost using first 2 features of test data set
plot_scatter_plot(best_predictions,"Adaboost Result")


# In[158]:


# plotting scatter plot for actual values of data points using first 2 features of test data set
plot_scatter_plot(y_test,"Actual Values")


# ## ```Classification Report for Adaboost Model```

# In[159]:


performance_metrics("Adaboost",y_test,best_predictions)


# # ```4. Multiclass SVM```
# 
# - Aim is to find optimal hyperplane for data points that maximizes the margin.
# 
# 
# - We will get output as set of weights, one for each feature, whose linear combination predicts the value of target attribute.
# 
# 
# - To maximize the margin, we reduce the number of weights that are non-zero to a few.
# 
# <br>
# 
# ![svm](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)
# 
# <br>
# 
# - We solve the following constraint optimization problem
# 
# $$
# \max_{w, b} \frac{2}{\|w\|} \space\space\space \text{such that} \space\space\space \forall i: y_i (w \cdot x_i + b) \geq 1
# $$
# 
# - which can be written as, 
# 
# $$
# \min_{w, b} \frac{\|w\|^{2}}{2} \space\space\space \text{such that} \space\space\space \forall i: y_i (w \cdot x_i + b) \geq 1
# $$
# 
# - Corresponding Lagrangian is
# 
# $$
# \mathcal{L}(w, b, \alpha_{i}) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (w \cdot x_i + b) - 1 ]
# \space\space\space \forall i: \alpha_i \geq 0
# $$
# 
# - The above indicates primal form of optimization problem, for which we can write equivalent Lagrangian Dual Problem
# 
# 
# $$
# \max_{\alpha} \left[ \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \right]
# \space\space\space \text{such that} \space\space \forall i: \alpha_i \geq 0 \space \text{and} \space \sum_{i=1}^{N} \alpha_i y_i = 0
# $$
# 
# 
# - The dual optimization problem for the SVM is a quadratic programming problem. The solution to this quadratic programming problem yields the Lagrange multipliers for each point in the dataset.
# 
# 
# - The weight vector is given by
# 
# $$
# w = \sum_{i=1}^{N} \alpha_i y_i x_i
# $$
# 
# - And bias term is
# 
# $$
# b = y_i - \sum_{j=1}^{N} \alpha_j y_j (x_j \cdot x_i)
# $$
# 
# - Only points with α > 0 define the hyperplane (contribute to the sum). These are called support vectors.
# 
# 
# $$
# y = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i (x_i \cdot x) + b\right)
# $$
# 
# - Here, α are the Lagrange multipliers, y are the class labels of the support vectors, x are the support vectors, x is the new example, and b is the bias term.

# ## ```Kernel Transformation```
# 
# - When the decision boundary between classes is not linear in the input feature space, we use non-linear Support Vector Machines (SVM) in which we use Kernel functions to map non-linear model in input space to a linear model in higher dimensional feature space.
# 
# 
# - The kernel trick allows the SVM to operate in this higher-dimensional space without explicitly computing the transformation.
# 
# 
# - The general form of the decision function in a non-linear SVM is
# 
# 
# $$
# y = \text{sign}\left(\sum_{i \in \text{SV}}^{N} \alpha_i y_i K(x_i, x) + b\right)
# $$
# 
# - We will use the following Kernel functions
# 
# 
#     - Linear Kernel
# 
# $$
# K(x_i, x_j) = x_i \cdot x_j
# $$
# 
#     - Polynomial Kernel
#    
# $$
# K(x_i, x_j) = (1 + x_i \cdot x_j)^p
# $$
# 
#     - Radial Basis Function (RBF) or Gaussian Kernel
#     
# $$
# K(x_i, x_j) = \exp\left(-\frac{1}{2\sigma^2} \|x_i - x_j\|^2\right)
# $$
# 
# 

# ## ```4.1 Implementation of the Model```

# In[160]:


# defining kernel functions

# linear kernel
def linear_kernel(xi,xj,dummy1=1,dummy2=1):
    return xi.dot(xj.T)
    
# polynomial kernel function
def polynomial_kernel(xi,xj,p=2,dummy=1):
    return (1 + xi.dot(xj.T))**p

# gaussian kernel function
def gaussian_kernel(xi,xj,dummy=1,sigma=0.1):
    return np.exp(-0.5*(1 /sigma ** 2) * np.linalg.norm(xi[:, np.newaxis] - xj[np.newaxis, :], axis=2) ** 2)


# In[161]:


class Support_Vector_Machine:
   
    # initial constructor for SVM class
    def __init__(self, kernel='polynomial', degree=2, sigma=0.1, epoches=5000, learning_rate= 0.001):
        self.alpha = None
        self.b = 0
        self.degree = degree
        self.C = 1
        self.sigma = sigma
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.kernel = kernel
        
        if self.kernel == 'linear':
            self.kernel = linear_kernel
        elif self.kernel == 'polynomial':
            self.kernel = polynomial_kernel
        elif self.kernel == 'gaussian':
            self.kernel =  gaussian_kernel
            
    # function for training binary SVM Model
    def train(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = np.random.random(x_train.shape[0])
        self.b = 0
        self.ones = np.ones(x_train.shape[0])
        
        # calculating yi*yj*K(xi, xj)
        y_mul_kernal = np.outer(y_train,y_train) * self.kernel(x_train,x_train,self.degree,self.sigma)
        
        # performing gradient descent
        for i in range(self.epoches):
            
            # 1 – yk ∑αj*yj*K(xj, xk)
            gradient = self.ones - y_mul_kernal.dot(self.alpha) 
            
            # α = α + η*(1 – yk ∑αj*yj*K(xj, xk)) to maximize
            self.alpha += self.learning_rate * gradient 
            
            # checking if 0<α<C
            self.alpha[self.alpha < 0] = 0
            self.alpha[self.alpha > self.C] = self.C 
            
            # ∑αi – (1/2) ∑i ∑j αi αj yi yj K(xi, xj)
            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal) 
           
            alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
       
        # for intercept b, we will only consider α which are 0<α<C
        b_values = []        
        for index in alpha_index:
            b_values.append(y_train[index] - (self.alpha * y_train).dot(self.kernel(x_train, x_train[index])))
            
        # finding b value
        # take mean or average of yi – ∑αj*yj*K(xi,xj)
        self.b = np.mean(b_values) 
        
    # function to predict values
    def decision_function(self,x_test):
        return (self.alpha * self.y_train).dot(self.kernel(self.x_train, x_test)) + self.b


# ## ```Training Multiclass SVM Model```

# In[162]:


def find_predictions(kernel='linear',degree=2):
    
    # finding unique classes
    unique_classes = np.unique(y_train)
    # to store predictions made by each of the model
    dictionary_of_models = {}
    predictions = []
    
    # we will iterate over each class and train one vs rest model for that class
    # points belonging to that class will be labeled as 1 and rest as -1
    for index in unique_classes:
        class_label = []  # Reset class_label for each class
        for label in y_train:
            if label == index:
                class_label.append(1)
            else:
                class_label.append(-1)
                
        # creating instance of model
        svm_model = Support_Vector_Machine(kernel=kernel,degree=degree)
        
        # training svm model for class index vs rest
        svm_model.train(np.array(x_train),class_label)
        
        # adding predictions
        predictions.append(svm_model.decision_function(np.array(x_test)))
        
    return predictions


# In[163]:


def find_final_predictions(kernel='linear',degree=2):
    predictions = find_predictions(kernel,degree=degree)
    
    # finding final predictions
    predictions = np.array(predictions).T
    final_predictions = np.argmax(predictions, axis=1) + 1  # Add 1 to align with class labels
    
    return final_predictions


# In[164]:


# function to find accuracy
def give_accuracy(y_test,y_pred):
    count = 0
    for i in range(len(y_test)):
        if(y_test[i]==y_pred[i]):
            count+=1

    accuracy = count/len(y_test)
    return accuracy


# ## ```4.2 Insights drawn (plots, markdown explanations)```

# # ```Comparing Accuracy with different Kernel functions```

# In[165]:


accuracy_overall = []
predictions_array_overall = []


# ## ```Linear Kernel Function```

# In[166]:


final_predictions = find_final_predictions('linear')
predictions_array_overall.append(final_predictions)
accuracy = give_accuracy(y_test,final_predictions)
accuracy_overall.append(accuracy)
print("Accuracy for SVM with linear kernel is: ",accuracy)


# ## ```Polynomial Kernel Function```

# In[167]:


accuracy_array = []
predictions_array = []
degree_array = [2,3,4,5,10,20]

for degree_val in degree_array:
    final_predictions = find_final_predictions('polynomial',degree=degree_val)
    predictions_array.append(final_predictions)
    accuracy = give_accuracy(y_test,final_predictions)
    accuracy_array.append(accuracy)
    print(f"Accuracy for SVM with polynomial kernel (degree={degree_val}) is: {accuracy}")


# In[168]:


plt.plot(degree_array,accuracy_array)
plt.xlabel("Degree of Polynomial Kernel")
plt.ylabel("Accuracy")
plt.plot()


# In[169]:


# finding highest accuracy
max_index = 0
highest_accuracy = accuracy_array[0]

for i in range(len(accuracy_array)):
    if accuracy_array[i] > highest_accuracy:
        highest_accuracy = accuracy_array[i]
        max_index = i

best_predictions_polynomial = predictions_array[max_index]


# In[170]:


accuracy_overall.append(accuracy_array[max_index])
predictions_array_overall.append(best_predictions_polynomial)


# ## ```Gaussian Kernel Function```

# In[171]:


final_predictions = find_final_predictions('gaussian')
predictions_array_overall.append(final_predictions)
accuracy = give_accuracy(y_test,final_predictions)
accuracy_overall.append(accuracy)
print("Accuracy for SVM with gaussian kernel is: ",accuracy)


# In[172]:


# finding best prediction
max_index = 0
highest_accuracy = accuracy_overall[0]

for i in range(len(accuracy_overall)):
    if accuracy_overall[i] > highest_accuracy:
        highest_accuracy = accuracy_overall[i]
        max_index = i

best_predictions = predictions_array_overall[max_index]


# ## ```Visualizing results predicted by Multiclass SVM with actual values```

# In[173]:


def plot_scatter_plot(values,title):
    for ind in range(len(best_predictions)):
        if(np.array(values)[ind] == 1):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='yellow')
        elif(np.array(values)[ind] == 2):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='green')
        elif(np.array(values)[ind] == 3):
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='blue')
        else:
            plt.scatter(x_test.iloc[ind][0],x_test.iloc[ind][1],color='red')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["low","medium","high","very high"])
    plt.title(title)
    plt.show()


# In[174]:


# plotting scatter plot for prediction made by SVM using first 2 features of test data set
plot_scatter_plot(best_predictions,"SVM Result")


# In[175]:


# plotting scatter plot for actual values of data points using first 2 features of test data set
plot_scatter_plot(y_test,"Actual Values")


# ## ```Classification Report for SVM Model```

# In[176]:


performance_metrics("SVM",y_test,best_predictions)


# # ```Comparison of insights drawn from the models```

# In[177]:


MLA_columns = ['Name', 'Test Accuracy', 'Precision', 'Recall']
MLA_compare = pd.DataFrame(columns=MLA_columns)
index = 0

# MLA_compare
count = 0
for val in MLA:
    MLA_compare.loc[count] = MLA[val]
    count += 1
MLA_compare.sort_values(by='Test Accuracy', ascending=False, inplace=True)
MLA_compare


# # ```Test Accuracy Comparison```

# In[178]:


plt.subplots(figsize=(15,6))
sns.barplot(x="Name", y="Test Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Test Accuracy Comparison')
plt.show()


# # ```Precision Comparison```

# In[179]:


plt.subplots(figsize=(15,6))
sns.barplot(x="Name", y="Precision",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Precision Comparison')
plt.show()


# # ```Recall Comparison```

# In[180]:


plt.subplots(figsize=(15,6))
sns.barplot(x="Name", y="Recall",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Recall Comparison')
plt.show()


# # Best Model
# 
# ```As we can observe SVM has highest accuracy and recall score among other machine learning models, so SVM performs better than other models on this given dataset.```

# # ```5. References```
# 

# - https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# 
# - https://pandas.pydata.org/
# 
# - https://matplotlib.org/
# 
# - https://www.geeksforgeeks.org/machine-learning/
# 
# - https://seaborn.pydata.org/
