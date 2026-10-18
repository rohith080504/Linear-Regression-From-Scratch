# Simple Linear Regression: Theory, Estimation, and Inference

Regression models (regression equations) are only an approximation to the true functional relationship between the variables of interest:
* **Functional Relationship** $\rightarrow$ Mechanical models
* **Regression Models** $\rightarrow$ Empirical models

The general regression equation with multiple regressors is:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k + \epsilon$$

Where:
* $\beta_0$: Intercept
* $\beta_1, \beta_2, \dots, \beta_k$: Slopes of relative $x$'s (regression coefficients)
* $x_1, x_2, \dots, x_k$: Regressor variables (features in machine learning lingo)
* $y$: Response variable (target variable)
* $\epsilon$: Random error term

---

## What is Simple Linear Regression?

It is a model with a single regressor that has a relationship with a response $y$ that is a straight line:

$$y = \beta_0 + \beta_1 x + \epsilon$$

And $y$ is a random variable. That is, there is a probability distribution for $y$ at each possible value for $x$.

The mean and variance of this distribution are:
$$\mathbb{E}(y \mid x) = \beta_0 + \beta_1 x$$
$$\text{Var}(y \mid x) = \sigma^2$$

* The mean of $y$ is a linear function of $x$.
* The variance of $y$ does not depend on the value of $x$.

### Regression Coefficients ($\beta_0$ and $\beta_1$)
* **$\beta_1$ (Slope):** Change in the mean of the distribution of $y$ produced by a unit change of $x$.
* **$\beta_0$ (Intercept):** If the range of $x$ includes $x = 0$, then the intercept $\beta_0$ is the mean of the distribution of the response $y$ when $x = 0$. If the range does not include zero, then $\beta_0$ has no practical physical interpretation.

There are multiple methods to estimate these parameters, but in this project, we use the Ordinary Least Squares (OLS) estimation method.

---

## What is Ordinary Least Squares (OLS) Estimation?

The method of ordinary least squares is used to estimate $\beta_0$ and $\beta_1$. We estimate $\beta_0$ and $\beta_1$ so that the sum of the squares of the difference between the observations $y_i$ and the straight line is a minimum:

$$S(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

Where $y = \beta_0 + \beta_1 x + \epsilon, \quad i = 1, 2, 3, \dots, n$.

To find the minima of $S(\beta_0, \beta_1)$, we differentiate with respect to $\beta_0$ and $\beta_1$ and set them equal to zero:

$$\frac{\partial S(\beta_0, \beta_1)}{\partial \beta_0} = 0, \quad \frac{\partial S(\beta_0, \beta_1)}{\partial \beta_1} = 0$$

Then we get:

$$\hat{\beta}_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

Where:

$$\text{Var}(x) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

$$\text{With } S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2, \quad S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

---

## Residuals and Model Evaluation

The difference between the observed value $y_i$ and the corresponding fitted value $\hat{y}_i$ is called the residual:

$$e_i = y_i - \hat{y}_i$$

After obtaining the ordinary least squares fit, a number of interesting questions come to mind:
* How well does this equation fit the data?
* Is the model likely to be useful as a predictor?
* Are any of the basic assumptions violated, and if so, how serious is this?

All of these issues must be investigated before the model is finally adopted for use.

---

## Estimator Properties and OLS Fit Properties

### Estimator Properties
$\hat{\beta}_0$ and $\hat{\beta}_1$ estimates are unbiased estimators of the model parameters $\beta_0$ and $\beta_1$:

$$\mathbb{E}(\hat{\beta}_0) = \beta_0, \quad \mathbb{E}(\hat{\beta}_1) = \beta_1$$

Their sampling variances are:

$$\text{Var}(\hat{\beta}_0) = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)$$

$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{S_{xx}}$$

### Properties of the Least-Squares Fit
* The sum of the residuals in any regression model that contains an intercept $\beta_0$ is always zero:
  $$\sum_{i=1}^{n} e_i = 0$$
* The sum of the observed values $y_i$ equals the sum of the fitted values $\hat{y}_i$:
  $$\sum_{i=1}^{n} y_i = \sum_{i=1}^{n} \hat{y}_i$$
* The least squares regression line always passes through the centroid $(\bar{x}, \bar{y})$ of the data.
* The sum of the residuals weighted by the corresponding values of the regressor always equals zero:
  $$\sum_{i=1}^{n} x_i e_i = 0$$
* The sum of the residuals weighted by the corresponding fitted values always equals zero:
  $$\sum_{i=1}^{n} \hat{y}_i e_i = 0$$

---

## Hypothesis Testing on the Slope and Intercept

Under the standard regression assumptions:
* Errors $\epsilon_i$ are normally and independently distributed with mean 0 and variance $\sigma^2$:
  $$\epsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$
* $y_i$ are normally and independently distributed with mean $\beta_0 + \beta_1 x_i$ and variance $\sigma^2$:
  $$y_i \overset{\text{ind}}{\sim} \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2)$$
* $\hat{\beta}_1$ estimates are normally and independently distributed with mean $\beta_1$ and variance $\sigma^2 / S_{xx}$:
  $$\hat{\beta}_1 \sim \mathcal{N}\left(\beta_1, \frac{\sigma^2}{S_{xx}}\right)$$

### Testing the Slope
To test the hypothesis that the slope equals a constant, say $\beta_{10}$:
$$H_0 : \beta_1 = \beta_{10}$$
$$H_1 : \beta_1 \neq \beta_{10}$$

* **If $\sigma^2$ were known**, we could use $Z$ to test the hypothesis:
  $$Z = \frac{\hat{\beta}_1 - \beta_{10}}{\sqrt{\sigma^2 / S_{xx}}}$$

* **Typically, $\sigma^2$ is unknown:**
  Mean Square Error ($MS_{\text{Res}}$) is an unbiased estimator of $\sigma^2$:
  $$\hat{\sigma}^2 = MS_{\text{Res}}$$

* **$t$-Test Statistic:**
  $$t_0 = \frac{\hat{\beta}_1 - \beta_{10}}{\sqrt{MS_{\text{Res}} / S_{xx}}}$$

  It follows a $t_{(n-2)}$ distribution.
* **Decision Rule:**
  Reject the null hypothesis if:
  $$|t_0| > t_{\alpha/2, \, n-2}$$

* **Standard Error:**
  $$\text{se}(\hat{\beta}_1) = \sqrt{\frac{MS_{\text{Res}}}{S_{xx}}}$$

---

### Testing the Intercept
A similar procedure can be used to test hypotheses about the intercept:
$$H_0 : \beta_0 = \beta_{00}$$
$$H_1 : \beta_0 \neq \beta_{00}$$

Where the standard error is:
$$\text{se}(\hat{\beta}_0) = \sqrt{MS_{\text{Res}} \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)}$$

And the test statistic is:
$$t_0 = \frac{\hat{\beta}_0 - \beta_{00}}{\text{se}(\hat{\beta}_0)} \sim t_{n-2}$$

---

### Testing the Significance of Regression
$$H_0 : \beta_1 = 0$$
$$H_1 : \beta_1 \neq 0$$

This hypothesis relates to the significance of regression. Failing to reject $H_0$ implies that there is no linear relationship between $x$ and $y$.