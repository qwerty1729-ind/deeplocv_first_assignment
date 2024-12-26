import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#  Step 1: User Inputs
# User defines the dimensions of the data (features and samples)
num_features = int(input("Enter the number of features: "))  # e.g., 2
num_data_points = int(input("Enter the number of data points: "))  # e.g., 100

# Step 2: Data Generation
np.random.seed(42)  # Ensures reproducibility

# X: Feature matrix (independent variables)
# Each row is a data point, and each column is a feature
X = np.random.randn(num_data_points, num_features)

# Simulated true weights (to create the best-fitting plane)
true_weights = np.random.rand(num_features)

# y: Target variable (dependent variable)
# Linear relationship with added Gaussian noise
y = X @ true_weights + np.random.randn(num_data_points) * 0.1

#  **Mathematical Explanation:**
# The objective of Ridge Regression is to minimize the following:
# Loss = Σ(y_i - (w · x_i + b))² + α Σ(w_j²)
# Where:
#   - y_i: Actual value of the i-th data point
#   - x_i: Feature vector of the i-th data point
#   - w: Coefficients (weights of each feature)
#   - b: Intercept (bias)
#   - α: Regularization parameter (controls penalty strength)
# The term Σ(w_j²) keeps weights small to prevent overfitting.

# Step 3: Train-Test Split
# Splitting data into training and testing sets for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Ridge Regression Model
alpha = 1.0  # Regularization strength (λ)
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(X_train, y_train)  # Training the model

# **Best Plane (Hyperplane) Intuition:**
# Ridge Regression optimizes the hyperplane by:
# 1. Minimizing the prediction error (MSE).
# 2. Penalizing large coefficients (L2 Regularization).
# The result is a hyperplane with a balanced fit to the data.

# Step 5: Model Predictions
y_pred_train = ridge_reg.predict(X_train)
y_pred_test = ridge_reg.predict(X_test)

#  Step 6: Model Evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# L2 Regularized Loss (Manual Calculation)
l2_loss_train = np.sum((y_train - y_pred_train)**2) + alpha * np.sum(ridge_reg.coef_**2)
l2_loss_test = np.sum((y_test - y_pred_test)**2) + alpha * np.sum(ridge_reg.coef_**2)

#  Step 7: Display Results
print("\n **Model Results:**")
print("Model Coefficients (weights):", ridge_reg.coef_)  # Best-fit hyperplane coefficients
print("Model Intercept:", ridge_reg.intercept_)  # Intercept of the hyperplane

print("\n**Loss Metrics:**")
print("Mean Squared Error (Train):", mse_train)  # Measures prediction error on training set
print("Mean Squared Error (Test):", mse_test)  # Measures prediction error on test set
print("L2 Regularized Loss (Train):", l2_loss_train)  # Total loss including regularization
print("L2 Regularized Loss (Test):", l2_loss_test)
