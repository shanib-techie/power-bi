import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Load Dataset

INPUT_FILE = r"C:\Users\anshu\OneDrive\Desktop\College\DSE (Discipline Specific Elective Courses)\Semester 3 - Data Mining I\Practicals\Datasets\Mall_Customers.csv"
df = pd.read_csv(INPUT_FILE)

# Use only numeric clustering features
data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]


# 2. Scale Data

scaler = StandardScaler()
X = scaler.fit_transform(data)


# 3. Try different DBSCAN parameters

eps_values = [0.2, 0.5, 0.8, 1.0]
min_samples_values = [3, 5, 8]

results = []

for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X)

        # Number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate silhouette only if ≥2 clusters exist
        if n_clusters >= 2:
            score = silhouette_score(X, labels)
        else:
            score = -1  # invalid

        results.append([eps, ms, n_clusters, score])
        print(f"eps={eps}, min_samples={ms}, clusters={n_clusters}, silhouette={score}")


# 4. Convert results to DataFrame

results_df = pd.DataFrame(results, columns=["eps", "min_samples", "clusters", "silhouette"])
print("\nParameter Testing Results:\n")
print(results_df)

# 5. Plot clusters for BEST parameter choice

best = results_df.loc[results_df["silhouette"].idxmax()]

best_eps = float(best["eps"])
best_ms = int(best["min_samples"])

print("\nBest Parameters:")
print(best)

# Train DBSCAN with best parameters
db_best = DBSCAN(eps=best_eps, min_samples=best_ms)
labels_best = db_best.fit_predict(X)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'],
            c=labels_best, cmap='viridis')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"DBSCAN Clusters (eps={best_eps}, min_samples={best_ms})")
plt.show()
