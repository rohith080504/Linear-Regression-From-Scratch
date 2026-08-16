import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    """Simple Linear Regression (OLS)
    built completely from scratch 
    using standard Python arithmetic loops
    —without relying on built-in functions 
    like `sum()` or `mean()`."""

    def __init__(self):
        self.slope: float | None = None
        self.intercept: float | None = None
        self.x: list[float] = []
        self.y: list[float] = []

    def _count(self, data: list[float]) -> int:
        """Counts elements without using len()."""
        count = 0
        for _ in data:
            count += 1
        return count

    def _sum(self, data: list[float]) -> float:
        """Sums elements without using standard sum()."""
        total = 0.0
        for val in data:
            total += float(val)
        return total

    def _mean(self, data: list[float]) -> float:
        """Calculates arithmetic mean without using np.mean()."""
        n = self._count(data)
        if n == 0:
            raise ValueError("Cannot calculate mean of an empty list.")
        return self._sum(data) / n

    def fit(self, x: list[float], y: list[float]) -> "LinearRegression":
        """Fits the model using manual least-squares calculations.

        Formula:
            slope = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
            intercept = mean_y - slope * mean_x
        """
        n_x = self._count(x)
        n_y = self._count(y)

        if n_x != n_y:
            raise ValueError("x and y must contain the same number of observations.")
        if n_x < 2:
            raise ValueError("At least 2 data points are required to fit a line.")

        self.x = [float(i) for i in x]
        self.y = [float(i) for i in y]

        # Calculate means manually
        mean_x = self._mean(self.x)
        mean_y = self._mean(self.y)

        # Compute numerator and denominator manually using a loop
        numerator = 0.0
        denominator = 0.0

        for i in range(n_x):
            dev_x = self.x[i] - mean_x
            dev_y = self.y[i] - mean_y

            numerator += dev_x * dev_y
            denominator += dev_x * dev_x  # (x_i - mean_x)^2

        if denominator == 0:
            raise ValueError("Variance of x is zero; line cannot be determined.")

        # OLS Estimates
        self.slope = numerator / denominator
        self.intercept = mean_y - (self.slope * mean_x)

        return self

    def predict(self, x_value: float) -> float:
        """Predicts y_hat for a single x value."""
        if self.slope is None or self.intercept is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.intercept + (self.slope * float(x_value))

    def fitted_values(self) -> list[float]:
        """Calculates fitted values (y_hat) for all training data points."""
        predictions = []
        n = self._count(self.x)
        for i in range(n):
            pred = self.predict(self.x[i])
            predictions.append(pred)
        return predictions

    def residuals(self) -> list[float]:
        """Calculates residual errors (e_i = y_i - y_hat_i)."""
        y_hat = self.fitted_values()
        res = []
        n = self._count(self.y)
        for i in range(n):
            res.append(self.y[i] - y_hat[i])
        return res

    def verify_properties(self, tolerance: float = 1e-5) -> dict[str, bool]:
        """Verifies fundamental linear regression properties:

        1. Sum of y observed == Sum of y predicted
        2. Sum of residuals == 0
        3. Fitted line passes through the sample centroid (mean_x, mean_y)
        """
        if not self.x or not self.y:
            raise RuntimeError("Model must be fitted before verifying properties.")

        mean_x = self._mean(self.x)
        mean_y = self._mean(self.y)

        sum_y = self._sum(self.y)
        sum_y_hat = self._sum(self.fitted_values())
        sum_e = self._sum(self.residuals())
        y_hat_at_mean_x = self.predict(mean_x)

        # Precision check handling floating point representation errors
        prop_sum_equal = abs(sum_y - sum_y_hat) < tolerance
        prop_residuals_zero = abs(sum_e) < tolerance
        prop_passes_centroid = abs(mean_y - y_hat_at_mean_x) < tolerance

        print("=== OLS Property Verification (Pure Calculations) ===")
        print(f"1. Sum(y) == Sum(y_hat): {prop_sum_equal} (Sum y = {sum_y:.4f}, Sum y_hat = {sum_y_hat:.4f})")
        print(f"2. Sum(residuals) == 0:  {prop_residuals_zero} (Sum e = {sum_e:.2e})")
        print(f"3. Line passes centroid: {prop_passes_centroid} (Mean y = {mean_y:.4f}, Pred y at Mean x = {y_hat_at_mean_x:.4f})\n")

        return {
            "sum_y_equals_sum_y_hat": prop_sum_equal,
            "sum_residuals_zero": prop_residuals_zero,
            "passes_centroid": prop_passes_centroid,
        }

    def visualize(self) -> None:
        """Visualizes data and fitted line using Matplotlib."""
        if not self.x or not self.y:
            raise RuntimeError("Model must be fitted before visualizing.")

        plt.figure(figsize=(8, 5))
        plt.scatter(self.x, self.y, color="blue", label="Observed Data", alpha=0.7)

        # Compute continuous line coordinates manually
        min_x = self.x[0]
        max_x = self.x[0]
        n = self._count(self.x)

        for i in range(n):
            if self.x[i] < min_x:
                min_x = self.x[i]
            if self.x[i] > max_x:
                max_x = self.x[i]

        x_line = [min_x, max_x]
        y_line = [self.predict(min_x), self.predict(max_x)]

        plt.plot(x_line, y_line, color="red", linewidth=2, label="OLS Regression Line")

        # Highlight Centroid Point
        mean_x = self._mean(self.x)
        mean_y = self._mean(self.y)
        plt.scatter([mean_x], [mean_y], color="green", s=100, zorder=5, label=r"Centroid $(\bar{x}, \bar{y})$")

        plt.title("Simple Linear Regression (No Built-ins)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def mean_squared_error(self) -> float:
        """Calculates Mean Squared Error (MSE)."""
        residuals = self.residuals()
        sum_sq_err = 0.0
        for e in residuals:
            sum_sq_err += e * e
        return sum_sq_err / self._count(self.x)

    def root_mean_squared_error(self) -> float:
        """Calculates Root Mean Squared Error (RMSE)."""
        return self.mean_squared_error() ** 0.5

    def r_squared(self) -> float:
        """Calculates Coefficient of Determination (R^2)."""
        mean_y = self._mean(self.y)

        # Sum of Squared Residuals (SS_res)
        ss_res = 0.0
        for e in self.residuals():
            ss_res += e * e

        # Total Sum of Squares (SS_tot)
        ss_tot = 0.0
        for y_i in self.y:
            diff = y_i - mean_y
            ss_tot += diff * diff

        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def plot_residual_diagnostics(self) -> None:
        """Generates residual plots to check OLS assumptions."""
        residuals = self.residuals()
        y_hat = self.fitted_values()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 1. Residuals vs Fitted (Checks Linearity & Homoscedasticity)
        ax1.scatter(y_hat, residuals, color="purple", alpha=0.7)
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_title("Residuals vs Fitted")
        ax1.set_xlabel("Fitted Values (y_hat)")
        ax1.set_ylabel("Residuals (e)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Histogram of Residuals (Checks Normality)
        ax2.hist(residuals, bins=10, color="skyblue", edgecolor="black")
        ax2.set_title("Distribution of Residuals")
        ax2.set_xlabel("Residual Error")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

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
