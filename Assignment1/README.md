# Linear Models For Regression And Classification

## ```Dataset```

The Diabetes dataset given contains the following features. It is used to predict which patient is diabetic
based on the analysis of the patient’s history and key medical factors.

1. No. of Pregnancies: To express the number of pregnancies
2. Glucose level: To express the Glucose level in blood
3. Blood Pressure: To express the Blood pressure measurement
4. Thickness of Skin: To express the thickness of the skin
5. Insulin level: To express the Insulin level in blood 
6. BMI: To express the Body mass index
7. Diabetes Pedigree Function: To express the Diabetes percentage. DPF calculates diabetes
likelihood depending on the subject's age and their diabetic family history.
8. Age: To express the age
9. Outcome: To express if the person is diabetic; 1 is Yes and 0 is No

## ```Stochastic Gradient Descent and Batch Gradient Descent using Linear Regression```
### ```Gradient Descent Algorithm```

- We will use this equation to update our linear regression model parameters

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
\end{equation}
$$

$$
\begin{equation}
\frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)}, \quad h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1}  + \theta_{2}x_{2}  +  ... +  \theta_{d}x_{d}
\end{equation}
$$

- Repeat until convergence

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)} ,\quad\text {$0 \leq j \leq d$}
\end{equation}
$$

- Such that it minimizes the cost function given by equation

$$
\begin{equation}
J(\theta) = {\frac{1}{2}}\sum_{i=1}^n{(h_{\theta}(x)^{(i)} - y^{(i)})^2}
\end{equation}
$$


- In ```batch gradient descent```, we are updating the model's parameters after we iterate through the entire dataset. Hence, it provides a ```more stable convergence``` towards the ```local minima``` of the cost function.


- In ```stochastic gradient descent```, we are updating the model's parameters after ```each observation```. Hence, it has a higher variance which results in a less stable convergence path.


- In stochastic gradient descent, it takes lesser number of iterations to converge to the local minima, as compared to batch gradient descent. For instance, the graphs plotted above show that stochastic gradient descent takes approximately ```100 iterations```, whereas batch gradient descent takes around ```400 iterations``` to converge to the local minima.


- The graph of the cost function shows ```more fluctuations in stochastic gradient descent```, whereas batch gradient descent has a ```smoother curve```.


- In reference to the model's ability to predict unseen data, the difference in batch and stochastic gradient descent can be ```negligible``` while using appropriate learning rates.


## ```Lasso and Ridge Regression using Polynomial Regression```
### ```Lasso Regression```


- We will use this equation to update our ridge regression model parameters

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
\end{equation}
$$

- Such that it minimizes the cost function given by equation

$$
\begin{equation}
 J(w) = \frac{1}{2} \sum_{n=1}^{N} \left( y(x_n, w) - y^{(i)} \right)^2 + \frac{\lambda}{2} \| \mathbf{w} \|_1
\end{equation}
$$

- Where,
  
$$
\| \mathbf{w} \|_1 \equiv \mathbf{w}^T \mathbf{w} = w_0 + w_1 + \ldots + w_D \quad\text {,for  d  features } 
$$

### ```Ridge Regression```

- We will use this equation to update our ridge regression model parameters

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
\end{equation}
$$

- Such that it minimizes the cost function given by equation

$$
\begin{equation}
 J(w) = \frac{1}{2} \sum_{n=1}^{N} \left( y(x_n, w) - y^{(i)} \right)^2 + \frac{\lambda}{2} \| \mathbf{w} \|_2^2
\end{equation}
$$

- Where,
  
$$
\| \mathbf{w} \|_2^2 \equiv \mathbf{w}^T \mathbf{w} = w_0^2 + w_1^2 + \ldots + w_D^2 \quad\text {,for  d  features } 
$$


- In ```Lasso Regression``` we use ```L1``` norm for the penalty term, whereas in ```Ridge Regression``` we use ```L2``` norm for the penalty term.

- We can see that in both Lasso and Ridge regression ```lower degree polynomials``` are giving lower MSE values for both training as well as testing data, but as we increase the degree of polynomials, we can see that there is significant difference between MSE for train and test data,MSE values for test data is much higher than MSE for training data which shows they are ```overfitting the training data```.

## ```Logistic Regression and Least Squares Classification```

### ```Logistic Regression```

- Logistic Regression is a statistical and machine learning technique for binary classification, i.e., it helps predict one of the two values 0 and 1 based on input features.

- In Logistic Regression, the prabability of an outcome is calculated using the sigmoid function.

### Sigmoid function
$$
\begin{equation}
P(Y = 1 | X) = \sigma(a) = \frac{1}{1 + e^{-a}}
\end{equation}
$$

### Cost Function in Logistic Regression

$$
\begin{equation}
L(y, y_{predicted}) = - [y * log(y_{predicted}) + (1 - y) * log(1 - y_{predicted})]
\end{equation}
$$


*   y: actual value


*   $y_{predicted}$: predicted value


*   Advantage of Logistic Regression is that it is less prone to overfitting, and is easy to read and interpret.


*   Disadvantage of Logistic Regression is that it assumes linearity property between the dependent variable and the independent variables, which narrows down the scope of usage of this technique.

Gradient with respect to weights $dw$:

The gradient of the cost function $J(w)$ with respect to the weights $w_j$ in the context of linear regression is given by:

$$
\begin{equation}
\frac{\partial J(w)}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (h_{w}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
\end{equation}
$$


In vectorized form, this becomes:

$$
\begin{equation}
dw = \frac{1}{n} X^T \cdot (h_{w}(X) - y)
\end{equation}
$$

where:
- $X$ is the matrix of input features where each row is a sample and each column is a feature.


- $y$ is the vector of actual target values.


- $h_{w}(X)$ is the vector of predicted values using the current weights \(w\).

Gradient with respect to bias $db$

The gradient of the cost function $J(w)$ with respect to the bias $b$ is given by:

$$
\begin{equation}
\frac{\partial J(w)}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (h_{w}(x^{(i)}) - y^{(i)})
\end{equation}
$$

In a more compact form:

$$
\begin{equation}
db = \frac{1}{n} \sum (h_{w}(X) - y)
\end{equation}
$$

### ```Least Square Classification```

*   Least Squares Classification is a very important technique used to solve binary classification problems. It directly assigns values 0 or 1 based on linear combination of input features.

* The W matrix mentioned above can be written in the below form

$$
W = \begin{bmatrix}
    w_1^{0} ... w_k^{0} \\
    w_1^{1} ... w_k^{1}\\
    w_1^{1} ... w_k^{2}\\
    \vdots \\
    w_1^{D} ... w_k^{D}
\end{bmatrix}
$$

*   We now determine the parameter matrix W by minimizing a sum-of-squares error function
*   Consider a training set
     {
    $x_n$
     , $t_n$}
    n=1,2,....,N

*   We define a matrix T whose nth row denotes $t_n^T$ vector together with a matrix X whose nth row denotes $x_n^T$

$$
T = \begin{bmatrix}
    t_1^{0} ... t_1^{0} \\
    t_2^{1} ... t_2^{1}\\
    t_3^{1} ... t_3^{2}\\
    \vdots \\
    t_N^{D} ... t_N^{D}
\end{bmatrix}
$$

* The X matrix can be written as

$$
X = \begin{bmatrix}
    x_1^{0} ... x_1^{D} \\
    x_2^{1} ... x_2^{D}\\
    x_3^{1} ... x_3^{D}\\
    \vdots \\
    x_N^{D} ... x_N^{D}
\end{bmatrix}
$$

* The sum of squares error function:

$$
\begin{equation}
E_D(W) = (1/2) Tr[(XW - T)^{T}(XW-T)]
\end{equation}
$$

*   Setting the derivative of the above function with respect to W as 0 and rearranging terms, we get

$$
\begin{equation}
W = (X^{T}X)^{-1}X^{T}T = X^{+}
\end{equation}
$$


*   T : NxK target matrix whose nth row is $t_n$
*   X : Nx(D+1) input matrix whose nth row is $x_n^T$

*   Advantage of Least Square Classification is that it is easy to understand and the decision boundary can be interpreted easily in terms of coefficients of the features


*   Disadvantages of Least Square Classification are that this method is easily influenced by outliers and thus may not give most accurate results. It does not give probabilistic outputs like ridge regression. It does not work for non-linear decision boundaries case as well


These gradients $dw$ and $db$ are then used in gradient descent to update the weights and bias, aiding the model in learning and improving its predictions during the training process.

- In ```Least Square Classification```, we aim to ```minimize the SSE```, wherease in Logisitic Regression we aim to model the probability of the binary outcome.


- In Logistic Regression we model the probability of the binary outcome by using a ```logisitic sigmoid function```.


- As we can see that Logistic Regression is giving ```more accuracy``` for the test data, as compared to the Least Square Classification.
