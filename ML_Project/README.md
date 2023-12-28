# Machine Learning for Sustainable Development Goals (SDGs)

In this project, we had to implement the machine learning algorithms given below from scracth in Python, compare the performance of different machine learning models, and collect insights from our analysis by varying different parameters.

## ```Dataset```

```Analysing Economic Growth Trends```

This project aims to analyse historical economic growth trends using the German Credit Data dataset. The insights from this analysis can include identifying factors that influence economic growth, predicting future economic trends, and assessing the impact of credit data on economic stability.

## ```Preprocessing and exploratory data analysis```

- Visualizing Numerical Features
- Visualizing Class Imbalance
- Checking Missing Values
- Converting Categorical Variables
- Correlation for strongly related features
- Discretizing continuous target variable
- Handling Data Imbalance using SMOTE (Synthetic Minority Oversampling Technique)
- Feature Selection using ANOVA Test and Chi-Sqaured Test
- Outlier Detection using IQR
- Feature Scaling using Standardization Method
- Dimensionality Reduction using PCA (Principal Component Analysis)

We implemented the following Machine Learning Algorithms as a part of this project

## ```Random Forest```

- Random forest is bagging of multiple decision trees. It has lower variance and reduced overfitting as compared to a single decision tree which has higher variance and is prone to overfitting.

- Random forest is more robust to outliers and noise as well.

- Main reason why random forest is so powerful is because of randomness, as each decision tree will be constructed on a bootstrapped subset of our data.

- Firstly, we will select n sample points from our train data with replacement, and then we will select a feature subspace out of all features.

- We will take majority vote from all decision trees for classifying our test data.


### ```Entropy```

Entropy is a measurement of uncertainty (sometimes called impurity) has the following formula

$$
H(X) = - \sum_{j} p_{j} log_{2}(p_{j})
$$

$$
\begin{equation}
\text{where} \space {p_{j} \space \text{is the probability of class j}}
\end{equation}
$$

For two class problem, it takes the following form

$$
H(X) = - plog_{2}(p)  -(1-p)log_{2}(1-p)
$$


When we are looking for a value to split our decision tree node, we want to minimize entropy.

### ```Overall Entropy```

Overall Entropy for children nodes, can be calculated using the formula

$$
H(children) = \sum_{j}^{2}( \frac{n_{j}}{N} \cdot H_{j})
$$

$$
  \text{where} \space n_{j} \space \text{is number of training samples associated with node j such that}
$$

$$
N = \sum_{j}^{2} n_{j}
$$

$$
  \text{and} \space H_{j} \space\text{is entropy of child node j}
$$

### ```Information Gain```

Information gain is a measure which tells how much information we have gained after splitting a node at a particular value.

$$
   \text{Information Gain} = H(Parent) - H(Children)
$$

$$
\text{where} \space H(Parent) \space\text{is entropy of Parent node or entropy before split and} \space H(Children) \space\text{is overall entropy of both left and right child nodes or entropy after split.}
$$

We need to find the split, which results in highest information gain.

### ```Visualizing Decision Boundaries after Post Pruning```

<p align="center">
  <img src="https://github.com/pavas23/Machine-Learning/assets/97559428/989cdb7f-7313-40b2-b242-d17f84327e6f" alt="Unpruned Decision Tree">
  <br>
  <em>Unpruned Decision Tree</em>
</p>

<p align="center">
  <img src="https://github.com/pavas23/Machine-Learning/assets/97559428/b574aa45-bdf8-45ec-920c-4e08604c6ee3" alt="Pruned Decision Tree">
  <br>
  <em>Pruned Decision Tree</em>
</p>

## ```Fishers Linear Discriminant```


- Fisher's Linear Discriminant is a supervised learning classifier used to find a set of weights that can help us draw a decision boundary and classify the given labelled data.

- Main aim is to find a boundary vector that maximizes the separation between two classes of the given projected data.

- The following formulae are used:
  
- Covariance matrix is given by ($\tilde{X}$
 is the demeaned data matrix, $S$ is the Covariance matrix)
 
$$
S= \frac{1}{n} \tilde{X}^T \tilde{X}
$$

- Demeaned data matrix is the matrix whose each element is of the form $X_i$ = $(x_i - \mu)$


- We then compute the within-class covariance $S_w$ as follows 

$$
S_w = S_{class1} + S_{class2}
$$

- $S_w^{-1}$ is the inverse of the combined covariances of the two classes



- Consider $m_1$ and $m_2$ to be the means of classes 1 and 2 respectively. We then calculate the difference of means $m_{difference}$ and the sum of means $m_{sum}$ to compute the projection vector and threshold value

$$
m_{difference} = m_1 - m_2
$$

$$
m_{sum} = m_1 + m_2
$$

- We then compute the $projection$ $vector$ which is proportional to the dot product of the transpose of the mean difference and the inverse of within class covariance

$$
\text{Projection vector } \alpha\  (m_{difference})^T \cdot (S_W^{-1})
$$

- We compute the $Threshold$ value as follows
  
$$
Threshold = 0.5*(projection\ vector)\cdot(m_{sum})
$$



- We compute the mean $m$ of the means of the two classes
  
$$
m = (m_1 + m_2)/2
$$

- Consider $l^{T}$ to be
  
$$
l^{T} = (m_{difference})^T\cdot S^{-1}
$$

- Consider $y_0$ to be
  
$$
y_0 = l^Tx_0
$$

- If $E[y_0]$ is near to $m_1$, then it will belong to $Class\ 1$, else it will belong to $Class\ 2$

- Assign a datapoint $x_0$ to $Class\ 1$ if $y_0 > m$, else assign the datapoint to $Class\ 2$

- Here $m$ is given by
  
$$
m = 0.5*(m_1 - m_2)^TS^{-1}(m_1+m_2)
$$


