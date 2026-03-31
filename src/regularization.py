import numpy as np

np.random.seed(0)
n = 50
distance = np.random.uniform(5, 40, n)  # meters
load = np.random.uniform(10, 100, n)  # kg
congestion = np.random.randint(0, 5, n)  # number of nearby robots
# True relationship (unknown to the model)
time = 1.8 * distance + 0.3 * load + 5.0 * congestion + 10 + np.random.normal(0, 5, n)
X = np.column_stack([distance, load, congestion])
y = time


def normalize_features(X):
    """
    Normalize features using standardization (z-score normalization).

    Parameters:
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)

    Returns:
    X_normalized : np.ndarray
        Normalized feature matrix
    mean : np.ndarray
        Mean of each feature
    std : np.ndarray
        Standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / (
        std + 1e-8
    )  # Add small epsilon to avoid division by zero
    return X_normalized, mean, std


def compute_mse(y_pred, y_true):
    """Compute Mean Squared Error (MSE)."""
    return np.mean((y_pred - y_true) ** 2)


def compute_regularized_loss(y_pred, y_true, weights, lambda_):
    """Compute MSE + lambda * ||W||^2."""
    mse = compute_mse(y_pred, y_true)
    l2_penalty = lambda_ * np.sum(weights**2)
    return mse + l2_penalty


def gradient_descent_l2(X, y, learning_rate=0.1, iterations=1000, lambda_=0.1):
    """
    Perform gradient descent for linear regression with L2 regularization.

    Parameters:
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    learning_rate : float
        Learning rate for gradient descent (alpha)
    iterations : int
        Number of iterations to run
    lambda_ : float
        L2 regularization strength

    Returns:
    weights : np.ndarray
        Learned weights for each feature
    bias : float
        Learned bias term
    loss_history : list
        Regularized loss at each iteration
    """
    # Normalize features
    X_normalized, mean, std = normalize_features(X)

    m, n = X_normalized.shape

    # Initialize weights and bias
    weights = np.zeros(n)
    bias = 0

    loss_history = []

    # Gradient descent loop
    for iteration in range(iterations):
        # Predictions
        y_pred = np.dot(X_normalized, weights) + bias

        # Compute loss
        loss = compute_regularized_loss(y_pred, y, weights, lambda_)
        loss_history.append(loss)

        # Compute gradients
        dw = (2 / m) * np.dot(X_normalized.T, (y_pred - y)) + 2 * lambda_ * weights
        db = (2 / m) * np.sum(y_pred - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss:.4e}")

    return weights, bias, loss_history, mean, std


if __name__ == "__main__":
    lambdas = [0.01, 0.1, 1.0, 10.0]
    alpha = 0.1
    iterations = 1000

    print("\nL2-Regularized Gradient Descent Results")
    print("=" * 60)
    print(f"alpha = {alpha}")
    print(f"iterations = {iterations}")

    for lambda_ in lambdas:
        weights, bias, _, mean, std = gradient_descent_l2(
            X, y, learning_rate=alpha, iterations=iterations, lambda_=lambda_
        )

        X_norm = (X - mean) / (std + 1e-8)
        y_pred = X_norm @ weights + bias
        mse = compute_mse(y_pred, y)

        print("\n" + "-" * 60)
        print(f"Lambda = {lambda_}")
        print(f"Final weights = {weights}")
        print(f"Final bias = {bias:.6f}")
        print(f"Training MSE = {mse:.4e}")

# As lambda increases, the weights decrease and at the value of 10.0,
# the weights go negative.
