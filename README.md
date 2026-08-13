# Regression Analysis:
Regression Analysis is a statistical technique for investigating and modeling the relationship between variables  
* It is the most widely used statistical technique  
* Regression Analysis can aid in confirming a cause effect relationship, but it cannot be the sole basis of such a claim
  

Objective of regression analysis is to estimate the unknown parameters in the regression model. This process is called fitting the model to the data  

### What is regression model?  
Regression models (regression equation) is only an approximation to the true functional relationship between the variables of interest  
Functional Relationship --> mechanical models  
Regression models --> Empirical models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y = \beta_0 + \beta_1 x1 + \beta_1 x2 + ..... + \beta_1 x + \epsilon is the regression equation  
where, \beta_0 = intercept, \beta_1,\beta_2,....\beta_n are the slopes of relative x's, x's are the regressor variable or features in machine learning lingo and y is the response variable or target variable  

### What is simple linear regression?  
It is a model with a single regressor that has a relationship with a response y that is a straight line  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y = \beta_0 + \beta_1 x + \epsilon  
&nbsp;&nbsp;&nbsp;&nbsp; and y is a random variable. That is, there is a probability distribution for y at each possible value for x  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The mean and variance of this distribution is  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; E(y|x) = \beta_0 + \beta_1 x  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; var(y|x) = σ²  
The mean of y is a linear function of x  
although the variance of y does not depend on the value of x  
b0 and b1 are usually called regression coefficients  
b1 (slope) = Change in the mean of the distribution of y produced by a unit change of x
if range of x includes x=0 then the intercept b0 is the mean of the distribution of the response y when x = 0  
If the range does not include zero then b0 has no practical information  

There are multiple methods to estimate these parameters but in this project we used ordinary least square estimation method  

### What is ordinary least square estimation method?  
The method of ordinary least squares used to estimate b0 and b1. we estimate b0 and b1 so that the sum of the squares of the difference between the observations yi and the straight line is a minimum  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i.e S(b0,b1) = ∑(yi-b0-b1xi)^2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where y = \beta_0 + \beta_1 x + \epsilon, i = 1,2,3....n  
to find the minima of s(b0,b1) we differentiate with respect to bo and b1 and make it equal to zero  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;∂S(b0,b1)/∂bo = 0, ∂S(b0,b1)/∂b1 = 0  
then we get b1 estimate = cov(x,y)/var(x)  
when var(x) = 1/n-1((∑xi-mean_x)^2)  
The difference between the observed value yi and the corresponding fitted value is called residual  

After obtaining the ordinary least squares fit, a number of interesting questions come to mind  
* How well does this equation fit the data?
* Is the model likely to be useful as a predictor?
* Are any of the basic assumptions violated, and if so, how serious is this?


All of these issues must be investigated before the model is finally adopted for use  

* b0 and b1 estimates are unbiased estimators of the model parameters b0 and b1 i.e,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;E(b0_estimate) = bo, E(b1_estimate) = b1  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;var(b0_estimate) = σ²(1/n + (mean_x)^2/Sxx), var(b1_estimate) = σ²/Sxx  

### Properties of the least-squares fit:  
* The sum of the residuals in any regression model that contains an intercept b0 is always zero
* The sum of the observed values yi equals the sum of the fitted values yi_estimates
* The least squares regression line always pass through the centroid (the point (mean_x, mean_y)) of the data
* The sum of the residuals weighted by the corresponding values of the regressor always equals to zero
* The sum of the residuals weighted by the corresponding fitted value always equals to zero

### Hypothesis Testing on the slope and intercept:  
* Errors ei are Normally and independently distributed with mean 0 and variance σ²
* yi are Normally and independently distributed with mean b0 + b1*xi and variance σ²
* bi_estimate are Normally and independently distributed with mean b1 and variance σ²/sxx

To test the hypothesis that the slope equals to a constant say b10  
The appropriate hypothesis are  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; H0 : b1 = b10   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; H1 : b1! = b10  
If σ² were known we could use z to test the hypothesis  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;z = (b1_estimate - b10)/(σ²/Sxx)^1/2  
typically σ² is unknown,  
so we have mean square error is an unbiased estimator of σ²  
* T-test statistic,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;t0 = (b1_estimate - b10)/(Mean_square_error/Sxx)^1/2  
it follows t(n-2) distribution  
This procedure rejects the null hypothesis if |t0| > t(significance level/2) (n-2)  
Standard error(b1_estimate) = (Mean_square_error/Sxx)^1/2   
A similar procedure can be used to test hypothesis about the intercept  
### Testing significance of regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;H0 : b1 = 0  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;H1 : b1 != 0  
This hypothesis relate to the significance of regression. Failing to reject H0 implies that there is no linear relationship between x and y

## Code Quickstart

### Prerequisites
Only `matplotlib` is required for rendering diagnostic plots.

```bash
pip install matplotlib
