import numpy as np

np.random.seed(0)
n = 50
distance = np.random.uniform(5, 40, n)  # meters
load = np.random.uniform(10, 100, n)  # kg
congestion = np.random.randint(0, 5, n)  # number of nearby robots
# True relationship (unknown to the model)
time = 1.8 * distance + 0.3 * load + 5.0 * congestion + 10 + np.random.normal(0, 5, n)
X = np.column_stack([distance, load, congestion])

# Convert continuous target to binary classes for logistic regression
# 1 = high completion time, 0 = low completion time
y = (time > np.median(time)).astype(int)


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


def sigmoid(z):
    """Compute sigmoid activation."""
    z = np.clip(z, -500, 500)  # numerical stability
    return 1 / (1 + np.exp(-z))


def compute_loss(y_pred_proba, y_true):
    """Compute binary cross-entropy loss."""
    eps = 1e-12
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)
    )


def compute_accuracy(y_pred_proba, y_true):
    """Compute training accuracy using threshold 0.5."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return np.mean(y_pred == y_true)


def gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    """
    Perform gradient descent for logistic regression.

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
        Cross-entropy loss at each iteration
    accuracy_history : list
        Training accuracy at each iteration
    """
    # Normalize features
    X_normalized, mean, std = normalize_features(X)

    m, n = X_normalized.shape

    # Initialize weights and bias
    weights = np.zeros(n)
    bias = 0

    loss_history = []
    accuracy_history = []

    # Gradient descent loop
    for iteration in range(iterations):
        # Predicted probabilities
        logits = np.dot(X_normalized, weights) + bias
        y_pred_proba = sigmoid(logits)

        # Compute loss
        loss = compute_loss(y_pred_proba, y)
        loss_history.append(loss)

        # Compute and store accuracy
        acc = compute_accuracy(y_pred_proba, y)
        accuracy_history.append(acc)

        # Compute gradients
        error = y_pred_proba - y
        dw = (1 / m) * np.dot(X_normalized.T, error)
        db = (1 / m) * np.sum(error)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    return weights, bias, loss_history, accuracy_history, mean, std


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Run gradient descent
    weights, bias, loss_history, accuracy_history, mean, std = gradient_descent(
        X, y, learning_rate=0.1, iterations=1000
    )

    # Final metrics
    X_norm = (X - mean) / (std + 1e-8)
    final_probs = sigmoid(X_norm @ weights + bias)
    final_acc = compute_accuracy(final_probs, y)

    print("\n--- Final Results ---")
    print(f"Learned weights: {weights}")
    print(f"Learned bias: {bias:.4f}")
    print(f"Final cross-entropy loss: {loss_history[-1]:.4f}")
    print(f"Final training accuracy: {final_acc:.4f}")

    # Check if loss is monotonically decreasing
    is_monotonic = all(
        loss_history[i] >= loss_history[i + 1] for i in range(len(loss_history) - 1)
    )
    print(f"Loss is monotonically decreasing: {is_monotonic}")

    # Plot cross-entropy loss vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Cross-Entropy Loss vs. Iteration", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cross_entropy_vs_iteration.png", dpi=150)
    print("\nPlot saved as 'cross_entropy_vs_iteration.png'")
    plt.show()

    # Plot training accuracy vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_history, linewidth=2, color="green")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Training Accuracy", fontsize=12)
    plt.title("Training Accuracy vs. Iteration", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_accuracy_vs_iteration.png", dpi=150)
    print("Plot saved as 'training_accuracy_vs_iteration.png'")
    plt.show()
