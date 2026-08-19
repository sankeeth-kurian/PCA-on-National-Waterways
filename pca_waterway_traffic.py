import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the provided dataset
data = pd.read_csv("Dataset.csv")

# Extract the relevant columns
X = data['YEAR']
Y = data[['National Waterways- I', 'National Waterways- II', 'National Waterways-III']]

# Standardize the data
Y_mean = Y.mean()
Y_std = Y.std()
Y_stdized = (Y - Y_mean) / Y_std

print("Standardized Data:")
print(Y_stdized)

# Calculate the covariance matrix
cov_matrix = np.cov(Y_stdized, rowvar=False)
print("\nCovariance Matrix:")
print(cov_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Select the top k eigenvectors (Here reducing to 2 dimensions)
k = 2
selected_eigenvectors = eigenvectors[:, :k]
print(f"\nSelected {k} Eigenvectors:")
print(selected_eigenvectors)

# Project the data onto the new subspace
reduced_data = np.dot(Y_stdized, selected_eigenvectors)
print("\nReduced Dataset:")
print(reduced_data)

# Create a scatter plot for visualization
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.grid()
plt.show()