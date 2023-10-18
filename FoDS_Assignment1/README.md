## Assignment 1-A : Regression Without Regularization

- Build ```polynomial regression models``` (with degrees varying from 1 to 9) to predict
the target variable based on the input feature variable. Determine the degree of
the polynomial which best fits the given data

- Apply ```batch gradient descent``` to train the models

- Train each model for 500 iterations.

### `Gradient Descent Algorithm`
# 
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

<br>

### `Training/Testing Error vs Degree of Polynomial`
#

<p align="center">
<img width="501" alt="abcd" src="https://github.com/pavas23/Machine-Learning/assets/97559428/02aaea7c-9639-4b68-a913-fc2fd49fc1a6">
</p>

<br>


### `Conclusion`
#

- Upon plotting the graphs, we can observe that for a constant learning rate (0.0000005 in this case), the Cost v/s Iterations curves become steeper, i.e., the cost function decreases rapidly for lesser number of iterations as the degree of polynomial increases.

- For any degree of polynomial (degree 1 to 9), the cost v/s iteration curve diverges for very high learning rate.

<br>

## Assignment 1-B Polynomial Regression With Regularization

- Develop nine polynomial regression models (with degrees varying from 1 to 9) to
predict the target variable based on the two input feature variables. Determine the
degree of the polynomial which best fits the given data.

- Build polynomial regression models of best fit degree (obtained from above) with the
following generalized regularized error function

- Build different regression models by taking ```q as 0.5,1,2,4```.

- Experiment with λ values between 0 and 1 to obtain the optimal model for each
value of q.

- Use both ```Stochastic and Batch Gradient Descent```. Report best models for each of the
method.

- Plot ```Surface plots``` for the nine polynomial regression models and the four optimal
regularized linear regression models.

<br>

### ```Polynomial Regression```

- We will use this equation to update our regression model parameters

$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}\frac{\partial J(\theta)}{\partial \theta_{j}}  ,\quad\text {$0 \leq j \leq d$} 
\end{equation}
$$
 
$$
\begin{equation}
\frac{\partial J(\theta)}{\partial \theta_{j}} = \sum_{i=1}^n(h_{\theta}(x) - y^{(i)})x_{j}^{(i)} + 
\frac{\lambda}{2}{q}{\theta_{j}^{(q-1)}}, \quad h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1}  + \theta_{2}x_{2}  +  ... +  \theta_{d}x_{d}
\end{equation}
$$
 
- Repeat until convergence
  
$$
\begin{equation}
\theta_{j} = \theta_{j} - {\alpha}{({\sum_{i=1}^n(h_{\theta}(x) - y^{(i)})x_{j}^{(i)}+ \frac{\lambda}{2}{q}{\theta_{j}^{(q-1)}}})} ,\quad\text {$0 \leq j \leq d$}
\end{equation}
$$

- Such that it minimizes the cost function given by equation

$$
\begin{equation}
J(w) = \frac{1}{2} \sum_{n=1}^{N} \left( y(x_n, w) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{M} \left(  \mathbf{w_{j}}^q \right)
\end{equation}
$$

- Where,
  
$$
\| \mathbf{w} \|_1 |= w_0^q + w_1^q + \ldots + w_d^q \quad\text {,for  d  features } 
$$

<br>

### ```Surface Plot for Polynomial with degree 1```

<p align="center">
<img src = "https://github.com/pavas23/Machine-Learning/assets/97559428/9a5c29ae-7cb9-4a74-9953-9b43b0ec850d" height = 250 width = 300>
</p>

### ```Surface Plot for Best Fit Polynomial (degree 7) ```

<p align="center">
<img src = "https://github.com/pavas23/Machine-Learning/assets/97559428/f4facf58-809f-48fd-9161-44738deb7a45" height = 250 width = 300>
</p>

### ```Description Of Model Used```

- We first started off generating nine polynomial regression models using Batch and Stochastic Gradient Descent, in which we varied the value of q among 0.5,1,2,4, and for each case, we plotted the graph for cost function v/s epochs (iterations).
- We also found SSE and MSE for each case
- Among these models developed, we found the best model for both Batch and Stochastic Gradient Descent by plotting a graph between training and testing data v/s degree of the polynomial, for both the gradient descent methods.
- We then plotted surface plots for each of the nine polynomial regression models, and then analyzed the outcome of the same.

### `Training/Testing Error vs Degree of Polynomial`
#

<p align="center">
<img width="535" alt="fgh" src="https://github.com/pavas23/Machine-Learning/assets/97559428/18ad4428-898f-418c-8841-66455c4aa6e1">
</p>
