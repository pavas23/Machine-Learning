# Linear Models For Regression And Classification

## Dataset

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

## Stochastic Gradient Descent and Batch Gradient Descent using Linear Regression
### Gradient Descent Algorithm

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


## Lasso and Ridge Regression using Polynomial Regression
### Lasso Regression


- We will use this equation to update our ridge regression model parameters

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
\end{equation}
$$


$$
\begin{equation}
\frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)} + {\lambda}*{\theta_{j}}, \quad h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1}  + \theta_{2}x_{2}  +  ... +  \theta_{d}x_{d}
\end{equation}
$$


- Repeat until convergence

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}*{({\sum_{i=1}^n(h_{\theta}(x) - y^{(i)})*x_{j}^{(i)}+ {\lambda}*{\theta_{j}}})} ,\quad\text {$0 \leq j \leq d$}
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

