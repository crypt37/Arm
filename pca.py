import numpy as np
import matplotlib.pyplot as plt





def calculate_covariance_matrix(data):
    # Calculate the covariance matrix
    num_samples = data.shape[0]
    covariance_matrix = np.dot(data.T, data) / (num_samples - 1)
    return covariance_matrix


def perform_pca(data, num_components):
    # Preprocess the data
    standardized_data = data

    # Calculate the covariance matrix
    covariance_matrix = calculate_covariance_matrix(standardized_data)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the desired number of principal components
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Transform the data
    transformed_data = np.dot(standardized_data, selected_eigenvectors)

    return transformed_data


# Example usage
data = np.random.randn(20, 2)

# Perform PCA with 2 components

# plt.scatter(data[:,0],data[:,1])
transformed_data = perform_pca(data, num_components=2)
plt.scatter(transformed_data[:, 0], transformed_data[:,1])
print(transformed_data)
plt.show()
