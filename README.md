# Regression Analysis:
Regression Analysis is a statistical technique for investigating and modeling the relationship between variables  
* It is the most widely used statistical technique  
* Regression Analysis can aid in confirming a cause effect relationship, but it cannot be the sole basis of such a claim
  

Objective of regression analysis is to estimate the unknown parameters in the regression model. This process is called fitting the model to the data  

### What is regression model?  
Regression models (regression equation) is only an approximation to the true functional relationship between the variables of interest  
Functional Relationship --> mechanical models  
Regression models --> Empirical models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y = b0 + b1x1 + b2x2 +......bnxn + e is the regression equation  
where, b0 = intercept, b1,b2,....bn are the slopes of relative x's, x's are the regressor variable or features in machine learning lingo and y is the response variable or target variable  

### What is simple linear regression?  
It is a model with a single regressor that has a relationship with a response y that is a straight line  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y = b0 +b1x + e  
&nbsp;&nbsp;&nbsp;&nbsp; and y is a random variable. That is, there is a probability distribution for y at each possible value for x  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The mean and variance of this distribution is  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; E(y|x) = b0 + b1x  
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i.e s(b0,b1) = ∑(yi-b0-b1xi)^2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where yi = b0 + b1xi + ei, i = 1,2,3....n  
to find the minima of s(b0,b1) we differentiate with respect to bo and b1 and make it equal to zero  



