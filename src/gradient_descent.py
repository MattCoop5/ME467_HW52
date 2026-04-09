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


def compute_loss(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE) loss.

    Parameters:
    y_pred : np.ndarray
        Predicted values
    y_true : np.ndarray
        True values

    Returns:
    loss : float
        MSE loss
    """
    loss = np.mean((y_pred - y_true) ** 2)
    return loss


def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent for linear regression.

    Parameters:
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    learning_rate : float
        Learning rate for gradient descent
    iterations : int
        Number of iterations to run

    Returns:
    weights : np.ndarray
        Learned weights for each feature
    bias : float
        Learned bias term
    loss_history : list
        Loss at each iteration
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
        loss = compute_loss(y_pred, y)
        loss_history.append(loss)

        # Compute gradients
        dw = (2 / m) * np.dot(X_normalized.T, (y_pred - y))
        db = (2 / m) * np.sum(y_pred - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss:.4f}")

    return weights, bias, loss_history, mean, std


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Run gradient descent
    weights, bias, loss_history, mean, std = gradient_descent(
        X, y, learning_rate=0.01, iterations=1000
    )

    print("\n--- Final Results ---")
    print(f"Learned weights: {weights}")
    print(f"Learned bias: {bias:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")

    # Check if loss is monotonically decreasing
    is_monotonic = all(
        loss_history[i] >= loss_history[i + 1] for i in range(len(loss_history) - 1)
    )
    print(f"Loss is monotonically decreasing: {is_monotonic}")

    # Plot MSE vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("MSE vs. Iteration Number", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mse_vs_iteration.png", dpi=150)
    print("\nPlot saved as 'mse_vs_iteration.png'")
    plt.show()

    # Test multiple learning rates
    print("\n" + "=" * 60)
    print("Testing Multiple Learning Rates")
    print("=" * 60)

    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = {}

    plt.figure(figsize=(12, 7))

    for lr in learning_rates:
        print(f"\n--- Running with learning rate: {lr} ---")
        w, b, loss_hist, _, _ = gradient_descent(
            X, y, learning_rate=lr, iterations=1000
        )
        results[lr] = {
            "weights": w,
            "bias": b,
            "loss_history": loss_hist,
        }

        print(f"Final loss: {loss_hist[-1]:.4f}")

        # Plot loss curve
        plt.plot(loss_hist, linewidth=2, label=f"LR = {lr}")

    # Configure plot
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("MSE vs. Iteration for Different Learning Rates", fontsize=14)
    plt.yscale("log")  # Use log scale for y-axis
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("mse_vs_iteration_multiple_lr.png", dpi=150)
    print("\nPlot saved as 'mse_vs_iteration_multiple_lr.png'")
    plt.show()

    # Summary
    print("\n--- Summary of Learning Rates ---")
    for lr in learning_rates:
        final_loss = results[lr]["loss_history"][-1]
        print(f"Learning rate {lr}: Final loss = {final_loss:.4f}")

# We can see from the results that the smaller learning rates
#converge slower, while the larger learning rates converge faster.
#this can be seen by the linear decrease in the MSE for LR = 0.001
# The learning rate of 0.1 seems to provide a non oscillating and quick convergance, 
#which makes it an excellant choice for this problem.
#Too large of a learning rate is a poor choice because it will respond quickly, 
#however it may overshoot the optimal parameters and diverge, or may cause oscillation.