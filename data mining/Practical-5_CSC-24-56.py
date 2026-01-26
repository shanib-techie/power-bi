import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load any dataset (using WINE)

data = load_wine()
X = data.data

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Simple K-Means Implementation
def simple_kmeans(X, k=3, max_iter=20):
    np.random.seed(42)
    n_samples, n_features = X.shape
    
    # Randomly initialize centroids
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    mse_list = []  # store MSE after each iteration

    for i in range(max_iter):

        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Compute MSE (mean squared error)
        mse = np.mean([np.linalg.norm(X[j] - centroids[labels[j]])**2 for j in range(n_samples)])
        mse_list.append(mse)

        # Update centroids
        new_centroids = np.array([X[labels == c].mean(axis=0) for c in range(k)])

        # Stop if centroids do not change
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids

    return labels, centroids, mse_list

# Run K-Means with user parameters
k = int(input("Enter number of clusters (e.g., 3): "))
max_iter = int(input("Enter maximum iterations (e.g., 20): "))

labels, centroids, mse_list = simple_kmeans(X, k=k, max_iter=max_iter)

# Plot MSE vs Iterations
plt.plot(range(1, len(mse_list) + 1), mse_list, marker='o')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title(f"K-Means MSE vs Iterations (k={k}, max_iter={max_iter})")
plt.grid(True)
plt.show()

print("\nFinal MSE:", mse_list[-1])
print("Centroids shape:", centroids.shape)
