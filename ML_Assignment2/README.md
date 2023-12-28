# Decision Tree and Support Vector Machine

In this assignment, we had to implement the machine learning algorithms given below from scracth in Python, compare the performance of different machine learning models, and collect insights from our analysis by varying different parameters.

## ```Dataset```

```Communities and Crime```

The data combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR.
The crime dataset contains 128 socio-economic features from the US 1990 Census. The target is the crime rate per community.

Number of Instances: 1994

Number of Attributes: 128

## ```Preprocessing and exploratory data analysis```

- Checking Missing Values
- Converting Categorical Variables
- Correlation for strongly related features
- Discretizing continuous target variable
- Handling Data Imbalance using SMOTE (Synthetic Minority Oversampling Technique)
- Feature Selection using ANOVA Test
- Outlier Detection using IQR
- Feature Scaling using Standardization Method
- Dimensionality Reduction using PCA (Principal Component Analysis)

We implemented the following Machine Learning Algorithms as a part of the assignment

## ```Decision tree model with entropy implementation```

- Decision Tree is a Supervised learning technique that can be used for classification problems.

- A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.

- Decision trees can model nonlinear relationships in the data, capturing complex patterns that linear models may miss.

- Decision trees are relatively robust to irrelevant features, as they are less likely to be selected for splitting.

- Decision trees are prone to overfitting, especially when the tree is deep. Overfitting occurs when the model captures noise in the training data, leading to poor generalization on new, unseen data.

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
  <img src="https://github.com/pavas23/Machine-Learning/assets/97559428/0b4cecf9-11de-42e6-a6ce-2797bf01f8cb" alt="Unpruned Decision Tree">
  <br>
  <em>Unpruned Decision Tree</em>
</p>

<p align="center">
  <img src="https://github.com/pavas23/Machine-Learning/assets/97559428/9cf9048f-3f88-4e10-b40c-5f3e556137a2" alt="Pruned Decision Tree">
  <br>
  <em>Pruned Decision Tree</em>
</p>


## ```Adaboost```

Adaboost is one of the early successful algorithm wihtin the Boosting branch of Machine Learning.
 
- AdaBoost for classification is a supervised Machine Learning algo.
- It consists of iteratively training multiple stumps using feature data $x$ and target label $y$
- After training, the model's ovearll prefictions come from a weighted sum of the stumps predictions

- For instances, consider $n$ stumps, $P_i$ be the prediction of $i_{th}$ stump. whose corresponding weight is $w_{i}$ then:


$$
\text{Final Adaboost Prediction} = w_{{stump}1}*P{1} + w_{{stump}2}*P{2} + ... + w_{{stump}n}*P{n}
$$


### Algorithm of Adaboost

- AdaBoost trains a sequence of models with augmented sample weights generating 'confidence' coefficients $\alpha$ for each individual classifiers based on errors.

1. For a datset with N number of samples, we initialize weight of each data point with $w_i$ = $\frac{1}{N}$ . The weight of a datapoint represents the probability of selecting it during sampling. 


2. Training Weak Clasifiers: After sampling the dataset, we train a weak classifier $K_m$ using the training dataset. Here $K_m$ is just a decision tree stump

- For $m$ = 1 to $M$: 
- sample the dataset using the weights $w_{i}^{(m)}$ to obtain the training samples $x_{i}$
- Fit a classifier $K_m$ using all the training samples $x_i$
- Here $K_m$ is the prediction of $m_{th}$ stump

3. Calcuating Update Parameters from Error: Stumps in AdaBoost progressively learn from previous stump's mistakes through adjusting the weights of the dataset to accomodate the mistakes made by stumps at each iteration. 
- The weights of the misclassified data will be increased, and the weights of correctly classified data will be decreased. 
- As a result, as we go into further iterations, the training data will mostly comprise of data that is often misclassified by our stumps. 
- When data is misclassified, $y*K_m(x) = -1$, then the weight update will be positive and the weights will exponentially increase for that data. When the data is correctly satisfied, $y*K_m(x) = 1$, the opposite happens.


- Let $\epsilon$ be the ratio of sum of weights for misclassified samples to that of the total sum of weights for samples. Compute the value of $\epsilon$ as follows:

$$
\epsilon = \frac{\sum_{y_i \neq K_m(x_i)}w_i^{(m)}}{\sum_{y_i}w_i^{(m)}}
$$


- In the above equation, $y_i$ is the ground truth value of the target varibale and $w_i^{(m)}$ is the weight of the sample $i$ at iteration $m$
- Larger $\epsilon$ means largers misclassification percentage, makes the confidence $\alpha_m$ decrease exponentially.
- Let confidence that the AdaBoost places on $m_{th}$ stump's ability to classify the data be $\alpha_m$. Compute $\alpha_m$ value as follows:


$$
\alpha_{m} = \frac{1}{2}\log{(\frac{1-\epsilon}{\epsilon}})
$$


- Update all the weights as follows:
  
$$
w_{i}^{(m+1)} = w_{i}^{(m)}.e^{-\alpha_myK_m(x)} 
$$


4. New predictions can be computed by
   
$$
K(x) = sign[\sum_{m=1}^{M}\alpha_{m}.K_{m}(x)]
$$

- Low error means the $\alpha$ value is larger, and thus has higher importance in voting.

- AdaBoost predicts on -1 or 1. As long as the weighted sum is above 0, it predicts 1, else, it predicts -1.

<br/>

## ```Multiclass SVM```

- Aim is to find optimal hyperplane for data points that maximizes the margin.


- We will get output as set of weights, one for each feature, whose linear combination predicts the value of target attribute.


- To maximize the margin, we reduce the number of weights that are non-zero to a few.

<br>

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png">
</div>

<br>

- We solve the following constraint optimization problem

$$
\max_{w, b} \frac{2}{\|w\|} \space\space\space \text{such that} \space\space\space \forall i: y_i (w \cdot x_i + b) \geq 1
$$

- which can be written as, 

$$
\min_{w, b} \frac{\|w\|^{2}}{2} \space\space\space \text{such that} \space\space\space \forall i: y_i (w \cdot x_i + b) \geq 1
$$

- Corresponding Lagrangian is

$$
\mathcal{L}(w, b, \alpha_{i}) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (w \cdot x_i + b) - 1 ]
\space\space\space \forall i: \alpha_i \geq 0
$$

- The above indicates primal form of optimization problem, for which we can write equivalent Lagrangian Dual Problem


$$
\max_{\alpha} \left[ \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) \right]
\space\space\space \text{such that} \space\space \forall i: \alpha_i \geq 0 \space \text{and} \space \sum_{i=1}^{N} \alpha_i y_i = 0
$$


- The dual optimization problem for the SVM is a quadratic programming problem. The solution to this quadratic programming problem yields the Lagrange multipliers for each point in the dataset.


- The weight vector is given by

$$
w = \sum_{i=1}^{N} \alpha_i y_i x_i
$$

- And bias term is

$$
b = y_i - \sum_{j=1}^{N} \alpha_j y_j (x_j \cdot x_i)
$$

- Only points with α > 0 define the hyperplane (contribute to the sum). These are called support vectors.


$$
y = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i (x_i \cdot x) + b\right)
$$

- Here, α are the Lagrange multipliers, y are the class labels of the support vectors, x are the support vectors, x is the new example, and b is the bias term.

<br>

## ```Kernel Transformation```

- When the decision boundary between classes is not linear in the input feature space, we use non-linear Support Vector Machines (SVM) in which we use Kernel functions to map non-linear model in input space to a linear model in higher dimensional feature space.


- The kernel trick allows the SVM to operate in this higher-dimensional space without explicitly computing the transformation.


- The general form of the decision function in a non-linear SVM is


$$
y = \text{sign}\left(\sum_{i \in \text{SV}}^{N} \alpha_i y_i K(x_i, x) + b\right)
$$

- We will use the following Kernel functions


- Linear Kernel

$$
K(x_i, x_j) = x_i \cdot x_j
$$

- Polynomial Kernel
   
$$
K(x_i, x_j) = (1 + x_i \cdot x_j)^p
$$

- Radial Basis Function (RBF) or Gaussian Kernel
    
$$
K(x_i, x_j) = \exp\left(-\frac{1}{2\sigma^2} \|x_i - x_j\|^2\right)
$$

## ```Comparison of insights drawn from the models```

| Name |	Test Accuracy |	Precision	| Recall |
| ------------- | ------------- | --- | --- |
| SVM |	60.301508 |	0.598199 |	0.603015 |
| Decision Tree | 	58.040201 | 0.627750 |	0.580402 |
| Adaboost | 	52.512563 | 	0.686176 |	0.525126 |


