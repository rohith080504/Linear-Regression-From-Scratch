
# Simple Linear Regression: Theory and Estimation

Regression models are approximations to the true functional relationship between variables of interest.

- **Functional relationship:** Mechanical or deterministic models.
- **Regression model:** An empirical model that describes the relationship between variables in the presence of random variation.

The general linear regression model is

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k + \varepsilon
$$

where:

- $\beta_0$ = Intercept.
- $\beta_1, \beta_2, \ldots, \beta_k$ = Regression coefficients.
- $x_1, x_2, \ldots, x_k$ = Regressor variables or features.
- $y$ = Response or target variable.
- $\varepsilon$ = Random error term.

---

## 1. What Is Simple Linear Regression?

Simple linear regression is a regression model that uses a single regressor to describe the linear relationship between a response variable and an explanatory variable.

The model is

$$
y = \beta_0 + \beta_1 x + \varepsilon
$$

For a fixed value of $x$, the response $Y$ is a random variable. Its conditional mean and variance are assumed to satisfy

**Conditional mean:**

$$
E(Y \mid X=x) = \beta_0 + \beta_1 x
$$

**Conditional variance:**

$$
\mathrm{Var}(Y \mid X=x) = \sigma^2
$$

Thus, the conditional mean of $Y$ is a linear function of $x$, while its conditional variance is constant.

Here, $\beta_0$ and $\beta_1$ are called regression coefficients.

- **Slope ($\beta_1$):** The change in the conditional mean of $Y$ associated with a one-unit increase in $x$.
- **Intercept ($\beta_0$):** The conditional mean of $Y$ when $x=0$. Its practical interpretation depends on whether zero is a meaningful value within the range of the data.

The standard regression assumptions include

$$
E(\varepsilon \mid X=x)=0
$$

and

$$
\mathrm{Var}(\varepsilon \mid X=x)=\sigma^2
$$

The errors are also assumed to be independent across observations. Normality is additionally assumed when deriving exact finite-sample inference procedures.

---

## 2. Ordinary Least Squares (OLS) Estimation

The parameters $\beta_0$ and $\beta_1$ are unknown and must be estimated from the observed data.

Suppose we observe $n$ pairs of values:

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)
$$

For the $i$-th observation, the model is

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

The ordinary least squares (OLS) method estimates the regression coefficients by minimizing the sum of squared residuals.

The objective function is

### Sum of Squared Errors

$$
S(\beta_0,\beta_1) =
\sum_{i=1}^{n}(y_i-\beta_0-\beta_1x_i)^2
$$

To find the minimum, differentiate with respect to each parameter and set the derivatives equal to zero.

### 2.1 Normal Equations

Differentiating with respect to $\beta_0$ gives

$$
\frac{\partial S}{\partial \beta_0} =
-2\sum_{i=1}^{n}(y_i-\beta_0-\beta_1 x_i)=0
$$

Therefore,

$$
\sum_{i=1}^{n}y_i =
n\beta_0+\beta_1\sum_{i=1}^{n}x_i
$$

Differentiating with respect to $\beta_1$ gives

$$
\frac{\partial S}{\partial \beta_1} =
-2\sum_{i=1}^{n}x_i(y_i-\beta_0-\beta_1 x_i)=0
$$

Therefore,

$$
\sum_{i=1}^{n}x_i y_i =
\beta_0\sum_{i=1}^{n}x_i
+
\beta_1\sum_{i=1}^{n}x_i^2
$$

Solving these normal equations yields the OLS estimators.

### 2.2 Estimator of the Slope

The OLS estimator of the slope is

$$
\boxed{
\hat{\beta}_1 =
\frac{S_{xy}}{S_{xx}}
}
$$

where

$$
S_{xx} =
\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

and

$$
S_{xy} =
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
$$

Here, the sample means are

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i
$$

Using the sample covariance and sample variance, both defined with denominator $n-1$, we can equivalently write

$$
\boxed{
\hat{\beta}_1 =
\frac{\mathrm{Cov}(X,Y)}
{\mathrm{Var}(X)}
}
$$

This expression requires that the observed regressor values are not all identical, so that $S_{xx}>0$.

### 2.3 Estimator of the Intercept

The OLS estimator of the intercept is

$$
\boxed{
\hat{\beta}_0 =
\bar{y}-\hat{\beta}_1\bar{x}
}
$$

Hence, the fitted regression line is

$$
\boxed{
\hat{y} =
\hat{\beta}_0+\hat{\beta}_1 x
}
$$

---

## 3. Residuals and Model Evaluation

The fitted value corresponding to the $i$-th observation is

$$
\hat{y}_i=\hat{\beta}_0+\hat{\beta}_1 x_i
$$

The residual is the difference between the observed and fitted values:

$$
\boxed{
e_i=y_i-\hat{y}_i
}
$$

Residuals are used to assess the adequacy of the fitted model.

Important questions include:

1. How well does the regression line fit the observed data?
2. Is the model useful for predicting new observations?
3. Are the regression assumptions reasonable?
4. Are there outliers or influential observations?
5. Does the relationship appear linear?

Residual plots, measures of goodness of fit, and prediction errors can help answer these questions.

### 3.1 Residual Sum of Squares

The residual sum of squares (RSS), also called the error sum of squares (SSE), is

$$
\mathrm{SSE} =
\sum_{i=1}^{n}e_i^2 =
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

For simple linear regression with an intercept, the residual degrees of freedom are $n-2$, because two parameters are estimated.

The residual mean square is

$$
\boxed{
MS_{\mathrm{Res}} =
\frac{\mathrm{SSE}}{n-2}
}
$$

Under the standard regression assumptions, $MS_{\mathrm{Res}}$ is an unbiased estimator of the error variance $\sigma^2$.

---

## 4. Properties of the Least-Squares Estimators

Assume that the regressor values $x_1,\ldots,x_n$ are fixed and that the errors have mean zero, constant variance $\sigma^2$, and are independent.

### 4.1 Unbiasedness

The OLS estimators are unbiased:

$$
\boxed{
E(\hat{\beta}_0)=\beta_0
}
$$

$$
\boxed{
E(\hat{\beta}_1)=\beta_1
}
$$

This means that, over repeated samples under the model assumptions, the average value of each estimator equals its corresponding population parameter.

### 4.2 Variance of the Slope Estimator

The variance of the slope estimator is

$$
\boxed{
\mathrm{Var}(\hat{\beta}_1) =
\frac{\sigma^2}{S_{xx}}
}
$$

where

$$
S_{xx}=\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

A larger spread in the regressor values generally leads to a more precise slope estimate, all else being equal.

### 4.3 Variance of the Intercept Estimator

The variance of the intercept estimator is

$$
\boxed{
\mathrm{Var}(\hat{\beta}_0) =
\sigma^2
\left(
\frac{1}{n}
+
\frac{\bar{x}^2}{S_{xx}}
\right)
}
$$

The precision of the intercept estimate depends on the sample size, the spread of the regressor values, and the location of their mean relative to zero.

### 4.4 Algebraic Properties of the Least-Squares Fit

For an OLS regression that includes an intercept, the following identities hold.

**1. The sum of residuals is zero:**

$$
\boxed{
\sum_{i=1}^{n}e_i=0
}
$$

**2. The sum of observed values equals the sum of fitted values:**

$$
\boxed{
\sum_{i=1}^{n}y_i =
\sum_{i=1}^{n}\hat{y}_i
}
$$

**3. The fitted regression line passes through the centroid:**

$$
\boxed{
(\bar{x},\bar{y})
}
$$

Equivalently,

$$
\bar{y}=\hat{\beta}_0+\hat{\beta}_1\bar{x}
$$

**4. The residuals are orthogonal to the regressor values:**

$$
\boxed{
\sum_{i=1}^{n}x_i e_i=0
}
$$

**5. The residuals are orthogonal to the fitted values:**

$$
\boxed{
\sum_{i=1}^{n}\hat{y}_i e_i=0
}
$$

These are algebraic properties of the OLS solution; they do not, by themselves, establish that the model is statistically appropriate.

---

## 5. Hypothesis Testing in Simple Linear Regression

For exact finite-sample hypothesis tests, assume that the errors are independent and normally distributed:

$$
\varepsilon_i
\overset{\mathrm{ind}}{\sim}
N(0,\sigma^2),
\qquad i=1,\ldots,n
$$

Conditional on the regressor values, this implies

$$
Y_i
\overset{\mathrm{ind}}{\sim}
N(\beta_0+\beta_1x_i,\sigma^2)
$$

The OLS estimators have normal sampling distributions:

$$
\hat{\beta}_1
\sim
N\left(
\beta_1,\frac{\sigma^2}{S_{xx}}
\right)
$$

$$
\hat{\beta}_0
\sim
N\left(
\beta_0,
\sigma^2\left(
\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}}
\right)
\right)
$$

The variance $\sigma^2$ is usually unknown, so it is estimated using $MS_{\mathrm{Res}}$.

### 5.1 Testing the Slope

Suppose we want to test whether the slope equals a specified constant $\beta_{10}$.

The hypotheses are

$$
H_0:\beta_1=\beta_{10}
$$

$$
H_1:\beta_1\neq\beta_{10}
$$

If $\sigma^2$ were known, the test statistic would be

$$
Z =
\frac{\hat{\beta}_1-\beta_{10}}
{\sqrt{\sigma^2/S_{xx}}}
$$

Under the null hypothesis, this statistic follows a standard normal distribution.

Since $\sigma^2$ is usually unknown, we replace it with $MS_{\mathrm{Res}}$ and use the $t$-statistic:

$$
\boxed{
t_0 =
\frac{\hat{\beta}_1-\beta_{10}}
{\sqrt{MS_{\mathrm{Res}}/S_{xx}}}
}
$$

Under $H_0$, the statistic follows a Student's $t$-distribution with $n-2$ degrees of freedom:

$$
t_0\sim t_{n-2}
$$

The standard error of the slope estimator is

$$
\boxed{
SE(\hat{\beta}_1) =
\sqrt{\frac{MS_{\mathrm{Res}}}{S_{xx}}}
}
$$

For a two-sided test at significance level $\alpha$, reject $H_0$ if

$$
|t_0|>t_{1-\alpha/2,\,n-2}
$$

where $t_{1-\alpha/2,\,n-2}$ denotes the $(1-\alpha/2)$ quantile of the $t$-distribution with $n-2$ degrees of freedom.

### 5.2 Testing the Intercept

To test a specified value $\beta_{00}$, consider

$$
H_0:\beta_0=\beta_{00}
$$

$$
H_1:\beta_0\neq\beta_{00}
$$

The standard error of the intercept estimator is

$$
SE(\hat{\beta}_0) =
\sqrt{
MS_{\mathrm{Res}}
\left(
\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}}
\right)}
$$

The corresponding test statistic is

$$
t_0 =
\frac{\hat{\beta}_0-\beta_{00}}
{SE(\hat{\beta}_0)}
$$

Under $H_0$ and the normal-error assumptions,

$$
t_0\sim t_{n-2}
$$

For a two-sided test at significance level $\alpha$, reject $H_0$ if

$$
|t_0|>t_{1-\alpha/2,\,n-2}
$$

### 5.3 Testing the Significance of Regression

A common test examines whether there is evidence of a nonzero linear slope.

The hypotheses are

$$
H_0:\beta_1=0
$$

$$
H_1:\beta_1\neq0
$$

Under the null hypothesis, the response has no linear change in its conditional mean as $x$ changes.

The test statistic is

$$
t_0 =
\frac{\hat{\beta}_1}
{SE(\hat{\beta}_1)}
$$

Under $H_0$, it follows a $t$-distribution with $n-2$ degrees of freedom.

Reject $H_0$ for a two-sided test if

$$
|t_0|>t_{1-\alpha/2,\,n-2}
$$

Rejecting $H_0$ provides evidence of a nonzero linear slope. Failing to reject $H_0$ means that the data do not provide sufficient evidence of a nonzero linear slope at the chosen significance level. It does not prove that $x$ and $y$ have no relationship of any kind.

---

## 6. Confidence Intervals for the Regression Coefficients

Under the normal-error assumptions, a $(1-\alpha)100\%$ confidence interval for the slope is

$$
\boxed{
\hat{\beta}_1
\pm
t_{1-\alpha/2,\,n-2}
SE(\hat{\beta}_1)
}
$$

Similarly, the confidence interval for the intercept is

$$
\boxed{
\hat{\beta}_0
\pm
t_{1-\alpha/2,\,n-2}
SE(\hat{\beta}_0)
}
$$

These intervals quantify uncertainty in the estimated regression coefficients.

---

## 7. Summary

- **Slope estimator:** $\hat{\beta}1 = \frac{S_{xy}}{S_{xx}}$
- **Intercept estimator:** $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- **Fitted value:** $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$
- **Residual:** $e_i = y_i - \hat{y}_i$
- **Residual mean square:** $MS_{\mathrm{Res}} = \frac{\sum_{i=1}^{n} e_i^2}{n-2}$
- **Variance of slope estimator:** $\mathrm{Var}(\hat{\beta}1) = \frac{\sigma^2}{S_{xx}}$
- **Variance of intercept estimator:** $\mathrm{Var}(\hat{\beta}0) = \sigma^2\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)$
- **Slope standard error:** $SE(\hat{\beta}1) = \sqrt{\frac{MS_{\mathrm{Res}}}{S_{xx}}}$
- **Intercept standard error:** $SE(\hat{\beta}0) = \sqrt{MS_{\mathrm{Res}}\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)}$



**Key takeaway:** OLS estimates the regression coefficients by minimizing the sum of squared residuals. The sampling distributions of these estimators, together with an estimate of the error variance, allow us to quantify uncertainty and perform hypothesis tests under the appropriate assumptions.
