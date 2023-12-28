## Assignment 2-A : Implementing PCA from Scratch and Applying it to Car Data

The objective of this assignment is to gain a deeper understanding of Principal Component Analysis (PCA) by implementing it using NumPy and Pandas libraries and applying it to the 'Car_data' dataset to reduce dimensionality and visualize principal components.

## ```Implementing PCA using Covariance Matrices```

- It is used to reduce the dimensionality of dataset by transforming a large set into a lower dimensional set that still contains most of the information of the large dataset

- Principal component analysis (PCA) is a technique that transforms a dataset of many features into principal components that "summarize" the variance that underlies the data

- PCA finds a new set of dimensions such that all dimensions are orthogonal and hence linearly independent and ranked according to variance of data along them

- Eigen vectors point in direction of maximum variance among data, and eigen value gives the importance of that eigen vector

<div align = "center">
<img src = "https://miro.medium.com/v2/resize:fit:1192/format:webp/1*QinDfRawRskupf4mU5bYSA.png">
</div>

- First the dataset is centered by subtracting means from the feature values

- Means are calculated by 

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

- Centered features are then compted as 

$$
x_i = x_i - \mu
$$


- Let $x_1$, $x_2$... $x_n$ be be N training examples, each having D features.


- Mean of N training examples is given by $\bar{x}$, which can be computed as

$$
\bar{x} = \frac{1}{N} \sum_{n=1}^{N} x_{n1} + \frac{1}{N} \sum_{n=1}^{N} x_{n2}... + \frac{1}{N} \sum_{n=1}^{N} x_{nd}
$$

- Now suppose we have a graph with 2-dimensional points as follows:

<div align="center">
<img src = "https://i.stack.imgur.com/ym6ru.png">
</div>

- Our motive is to bring down the 2D points to 1D by projecting on a vector. We need to project the points on a 1D vector such that the variance between data points is maximum.


- We need to compute unit vector such that the variance is as maximum as possible. 


- We do some mathematical computations as follows:

$$
cos\theta = \frac{OA}{OB}
$$

$$
\bar{u} \cdot \bar{x_n} = (||u||)(||x||)cos\theta
                     = (||u||)(OB)cos\theta = (||u||)(OA)
$$

- The above equation gives us the below result

$$
OA = \frac{\bar{u} \cdot \bar{x_n}}{\|u\|}
$$

- We take projection on unit vector $||u||$ = 1

- Our final result is as follows:

$$
OA = \bar{u}\cdot\bar{x_n}
$$

- The mean of the projected points is given by 

$$
\frac{1}{N}\sum_{n=1}^{N}{\bar{u}\cdot\bar{x_n}} = \bar{u}\cdot\sum_{n=1}^N\frac{x_n}{N} = \bar{u}\cdot\bar{x}
$$

- Here, $\bar{x}$ is the mean of training points in their dimension

- We then compute variance as 

$$
Variance = \frac{1}{N}\sum_{n=1}^N{(\bar{u}\cdot\bar{x_n} - \bar{u}\cdot\bar{x})^2}
$$

- We then compute $\bar{u}$ which maximizes variance as much as possible such that $||u|| = 1$

- Consider $x_n$ and $\bar{x}$ to be matrices of $d$ x $1$ size represented as follows

$$
\bar{x_n}=\begin{bmatrix}
  x_1 \\
  x_2 \\
  x_3 \\
  \vdots \\
  x_d \\
\end{bmatrix}
$$


$$
\bar{x}=\begin{bmatrix}
  \bar{x_1} \\
  \bar{x_2} \\
  \bar{x_3} \\
  \vdots \\
  \bar{x_d} \\
\end{bmatrix}
$$

- Consider $\bar{u}$ to be a $1$ x $d$ matrix represented by

$$
\bar{u} = [{u_1}, {u_2}...{u_d}]
$$

- We then observe that we need to maximize the following expression

$$
max[\frac{1}{N}\sum_{n=1}^{N}(\bar{u}\cdot(\bar{x_n}-\bar{x}))(\bar{u}\cdot(\bar{x_n}-\bar{x})^{T}]
$$

- While trying to maximize the above expression by expanding the same, we get

$$
max[\frac{1}{N}\sum_{n=1}^{N}(\bar{u}(x_n - \bar{x})(x_n - \bar{x})^{T}\bar{u}^{T})]
$$

- The above expression in turn becomes

$$
max[\bar{u}\frac{1}{N}[\sum_{n=1}^{N}(x_n - \bar{x})(x_n - \bar{x})^{T}]\bar{u}^{T}]
$$

- The above expression simplifies to 

$$
max[\bar{u}S\bar{u}^T] 
$$

$$
||u||=1
$$
- Here, $S$ is called covariance matrix 


- Principal Component Analysis (PCA) gives linear combination of these features to get matured features


- We then try to convert the above constraint optimization problem to an unconstrained optimization problem, as follows:

$$
E(u,\lambda) = max[\bar{u}S\bar{u}^{T} + \frac{\lambda}{2}(1-\bar{u}\bar{u}^{T})]
$$

- Taking derivation with respect to $\bar{u}$ and $\lambda$ and setting it to 0, we get final answer to be 

$$
\bar{u}S\bar{u}^T = \lambda
$$

- $\lambda$ is called the eigen value found from the equation

$$
|A - \lambda{I}| = 0
$$

- Let $u_1$, $u_2$,... $u_d$ be the eigen vectors, and $\lambda_1$, $\lambda_2$,... $\lambda_d$ be the eigen values, $A$ is a $d$ x $d$ square matrix, we get 

$$
A\gamma = \lambda\gamma
$$

$$
Au_1 = \lambda_1u_1
$$

$$
Au_2 = \lambda_2u_2
$$

- Any of d $\bar{u}$ values are feasible solutions, we need to find optimal solution from the following set of equations

$$
Su_1 = \lambda_1u_1 
$$
$$
Su_2 = \lambda_2u_2 
$$

$$
.
$$

$$
.
$$

$$
.
$$

$$
Su_d = \lambda_du_d 
$$

- The above set of equations simplifies to

$$
u_1Su_1^{T} = \lambda_1
$$

$$
u_2Su_2^{T} = \lambda_2
$$

$$
.
$$

$$
.
$$

$$
.
$$

$$
u_dSu_d^{T} = \lambda_d
$$

- For instance, if we project all points on eigen vector $u_1$ then variance comes out to be $\lambda_1$


- $\lambda_1$, $\lambda_2$, ...., $\lambda_d$ are variances after projecting values/points on eigen vectors $u_1$, $u_2$,...., $u_d$. We need to find that eigen vector which has maximum variance, or simply, maximum $\lambda$.


- For instance, consider the first eigen vector to be of the form 

$$
u_1 = \begin{bmatrix}
  \bar{u_{11}} \\
  \bar{u_{12}} \\
  \bar{u_{13}} \\
  \vdots \\
  \bar{u_{1d}} \\
\end{bmatrix} 
$$

- Transformed point is 

$$
u_{11}x_{11} + u_{12}x_{12}+... + u_{1d}x_{1d}
$$

- Transformation of a point from multidimensional space (d-dimensional in this case) to a uni-dimensional space is a linear transformation (where multiples are componenents of eigen vectors in PCA)

<br>

## ```Conclusion and Interpretation```

- We can see that maximum variance is captured by first two principal components, which helps us in reducing dimension of dataset from 5 features to 2 features while retaining most of the information.


- PCA is a dimensionality reduction technique, and dimensionality reduction is the process of reducing the number of features in a dataset while retaining as much information as possible. This can be done to reduce the complexity of a model, improve the performance of a learning algorithm, or make it easier to visualize the data.


- PCA converts a set of correlated features in the high dimensional space into a series of uncorrelated features in the low dimensional space. These uncorrelated features are also called principal components.


- We can see that points on scatter plot are around the vectors when we take projections of principal components for a standardized dataset, as here we have properly scaled dataset, unlike when we took projection on original dataset.
