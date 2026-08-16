
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2.1, 3.8, 6.2, 7.1, 10.5, 11.8, 14.2, 16.0, 17.9, 20.1]

model = LinearRegression()
model.fit(x, y)

print(f"Slope (beta_1): {model.slope:.4f}")
print(f"Intercept (beta_0): {model.intercept:.4f}")

print(f"R-squared: {model.r_squared():.4f}")
print(f"RMSE: {model.root_mean_squared_error():.4f}")

model.verify_properties()

model.visualize()
model.plot_residual_diagnostics()
