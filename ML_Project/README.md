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

<p align="center">
  <img src="https://github.com/pavas23/Machine-Learning/assets/97559428/48152caf-9f07-40f8-bc6e-4a45f0a41306" alt="Fisher Linear Discriminant Analysis Result">
  <br>
  <em>Fisher Linear Discriminant Analysis Result</em>
</p>

<br>

## ```Multilayer Perceptron```

- Multilayer Perceptron is a type of ```Artificial Neural Networks (ANN)```, which is listed under supervised learning algorithms.


- It is basically a feed-forward neural network algorithm.


- It is called Multilayer Perceptron as there are multiple hidden layers between the input and the output values, which helps the model understand the complex mappings between the inputs and outputs.


- The overall structure of the Multilayer Perceptron is as follows:
    1. Input Layer: The input layer contains ```nodes(neurons)```, each of which corresponds to each feature of the input data. The number of nodes(neurons) in the input layer denotes the dimensionality of the input data.
    2. Hidden Layer(s): Data from input layer is passed through the hidden layers, which makes the neural network to understand the complex inferences from the given input dataset.
    3. Output Layer: The output layer helps produce the the network's predictions.
    4. Loss Function: The difference between the observed values and the expected values is called the loss, which is computed by the loss function. The main aim of during training is to ```minimize the loss as much as possible```.
    
    
- The Multilayer Perceptron works as follows:
    1. Initialize weights and biases: Initializing weights and biases with random values
    2. Forward Propagation: In this step, data moves sequentially from the input layer, through the hidden layers, and reaches the output layer, with the weights that are assigned to each link (connecting nodes) are adjusted while training.
    3. Backward Propagation: During the feed-forward process, the errors are computed by calculating the differences between the expected and the observed values. The ```backpropagation process starts by propagating the error backward through the network```. Main aim of this step is to observe the contribution of each weight to overall error.
    4. Continue doing until the end of epochs is reached.

Hidden layer(s): 
For each layer, the activations and outputs are calculated as: 

$$
L_{j}^{l} = \sum_{i} w_{j,i}^{l} x_{i}^{l} = w_{j,0}^{l} x_{0}^{l} + w_{j,1}^{l} x_{1}^{l} + w_{j,2}^{l} x_{2}^{l} + \ldots + w_{j,n}^{l} x_{n}^{l}
$$

$$
Y_{j}^{l} = g^{l}(L_{j}^{l})\\
\{ y_i, x_{i1}, \ldots, x_{ip} \}_{i=1}^{n}
$$

- $x_{i}^{l}$ denotes the $i^{th}$ input to layer $l$ coming from the $l^{th}$ layer,
- $y_{j}^{l}$ represents the output of the $j^{th}$ neuron in layer $l$,
- $w_{j,i}^{l}$ signifies the weight of the connection between the $i^{th}$ neuron in layer $l$ and the $j^{th}$ neuron in the same layer $l$,
- $L_{j}^{l}$ represents the net activation of the $j^{th}$ neuron in layer $l$,
- $g^{l}(\cdot)$ stands for the activation function of layer $l$.

<br>

## ```K Nearest Neighbors```

- KNN Algorithm compares a point in test dataset with the values in the given training dataset and based on its closeness or similarities in given range of K neighbors it assigns the point to the majority class.

- It is called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.

- We have different metrics that can be used to calculate distance


- ```Minkowski Distance```
    
    The Minkowski distance is a generalization of several distance metrics, including the Euclidean distance and the Manhattan distance. It is a way to measure the distance between two points in a multi-dimensional space.
    
$$
d(p,q) = ({\sum_{i=1}^n{|q_i - p_i|^p}})^{\frac{1}{p}}
$$

- ```Euclidean Distance```

    The Euclidean distance, also known as the L2 distance, is a common distance metric used to measure the straight-line distance between two points in a multi-dimensional space.
  
$$
d(p,q) = \sqrt{\sum_{i=1}^n{(q_i - p_i)^2}}
$$

- ```Manhattan Distance```
    
    The Manhattan distance, also known as the L1 distance or taxicab distance, is a distance metric used to measure the distance between two points in a multi-dimensional space by summing the absolute differences of their coordinates. 
    
$$
d(p,q) = \sum_{i=1}^n{|q_i - p_i|}
$$

- ```Chebyshev Distance```
    
    The Chebyshev distance, also known as the chessboard distance, is a distance metric used to measure the maximum absolute difference between the coordinates of two points in a multi-dimensional space.
    
$$
d(p,q) = max_{i}{|q_i - p_i|}
$$

- ```Cosine Distance```
    
    Cosine distance is a similarity metric commonly used to measure the cosine of the angle between two non-zero vectors in a multi-dimensional space. It quantifies the similarity between the two vectors by examining the cosine of the angle formed by the vectors' representations.
    
$$
d(p,q) = 1 - \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|}
$$

<br>

## ```Comparison of insights drawn from the models```

- After performing the chi-squared test on all the categorical features, we see that the p-values for all our categorical features range from 0 to 195.48. According to the chi-squared test, we rejected the features which did not have much of an effect on our target attribute, i.e., Customer Classification.

- We have also used the ANOVA test to find out the numerical features which affect the target attribute, i.e., Customer Classification.

- Therefore the number of categorical features which affect the target attribute are given below


```python
  Index(['Duration in Month', 'Age in years', 'Present residence since',
       'Number of existing credits at this bank', 'Customer Classification',
       'Checking Account_A14', 'Credit History_A34',
       'Personal status and sex_A93', 'Present employment since_A75',
       'Purpose_A43', 'Present employment since_A74', 'Purpose_A41',
       'Other installment plans_A143', 'Property_A121', 'Checking Account_A11',
       'Property_A123', 'Savings Account/Bonds_A65', 'Housing_A152',
       'Savings Account/Bonds_A63', 'Savings Account/Bonds_A64', 'Job_A172',
       'Personal status and sex_A94', 'Telephone_A192', 'Job_A173',
       'Checking Account_A13', 'Job_A174', 'Present employment since_A73',
       'Purpose_A49', 'Other debtors / guarantors_A101',
       'Other debtors / guarantors_A103', 'Credit History_A32',
       'Credit History_A33', 'Housing_A151', 'Property_A122', 'Purpose_A42',
       'Savings Account/Bonds_A62', 'Purpose_A48', 'Housing_A153',
       'Checking Account_A12'],
      dtype='object')
```


  - We see that the p-values of 'Duration in Month', 'Age in years', 'Present residence since','Number of existing credits at this bank', 'Customer Classification','Checking Account_A14', 'Credit History_A34' are higher. Hence their impact on assessing the credit on economic stability is higher

 
```python
{
    "Present residence since": 29.976250818691852,
    "Number of existing credits at this bank": 25.756499112950923,
    "Age in years": 11.886909705995478,
    "Duration in Month": 4.693614687806769,
    "Installment rate in percentage of disposable income": 1.6838002436053603,
    "Credit Amount": 1.0817072208587106,
    "Number of people being liable to provide maintenance for": 0.0
}
```


- We can see that the f-scores of "Present Residence since", and "Number of existing credits at this bank" are very high compared to the other features. Hence these features influence in assessing the impact of credit data on economic stability.

<br>

<div align="center">

| Name | Test Accuracy | Precision | Recall | AUC |
| ------------- | ------------- | --- | --- | --- |
| Fishers Linear Discriminant |	80.000000 | 0.827586 | 0.914286 | 0.707143 |
| KNN | 80.000000 | 0.833333 | 0.904762 | 0.714881 |
| Multilayer Perceptron	 | 75.862069 | 0.836538 | 0.828571 | 0.701786 |
| Random Forest	| 74.482759 | 0.854167 | 0.780952 | 0.715476 |

</div>
