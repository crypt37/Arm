import numpy as np
import matplotlib.pyplot as plt

matrix = 5 * np.random.randn(20, 2)
print(matrix)
y = np.linspace(1, 30, 40)
uniform_matrix = np.random.uniform(low=0, high=1, size=(2, 2))
matrix = matrix.reshape(20, 2)
multiplication = np.matmul(matrix, uniform_matrix)
plt.subplot(5, 1, 1)
plt.title("matrix values")
plt.scatter(matrix, y)
plt.subplot(5, 1, 2)
plt.scatter(y[0:4], uniform_matrix)
plt.subplot(5, 1, 3)
plt.title("normal multi")
plt.scatter(multiplication[:, 0], multiplication[:, 1])


covariance_matrix1 = np.cov(multiplication,multiplication)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix1)
print("the eigens are " , eigenvalues, eigenvectors)


plt.subplot(5, 1, 4)
Y=multiplication


Y = np.dot(matrix,eigenvectors)
plt.plot(5, 1, 5)

plt.scatter(Y[:, 0], Y[:, 1])

plt.tight_layout()
plt.show()
