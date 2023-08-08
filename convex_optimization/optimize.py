from typing import Optional

import os
import sys
import csv
import numpy
import cvxpy as cp

import matplotlib.pyplot as plt


def read_matrix_from_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        return numpy.loadtxt(csv_file, delimiter=",")
    

def read_omega_from_csv(csv_path: str, sparsity: float = 0.0):
    omega = read_matrix_from_csv(csv_path)
    for y in range(omega.shape[0]):
        for x in range(omega.shape[1]):
            if numpy.random.rand() < sparsity:
                omega[x, y] = 0

    return omega


def read_names_from_csv(csv_path: str):
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        names = next(reader)

    return names


def read_d_from_dist_csv(csv_path: str):
    dist = read_matrix_from_csv(csv_path)
    return dist * dist


def optimize_gram_matrix(D: numpy.ndarray, omega: numpy.ndarray, margin: float = 0.0):
    # variables
    X = cp.Variable(D.shape, symmetric=True)

    # construct problem
    objective = cp.Minimize(cp.trace(X))
    constraints = [X >> 0]
    for (i, j), is_distance_known in numpy.ndenumerate(omega):
        if is_distance_known:
            if margin == 0.0:
                constraints.append(X[i, i] + X[j, j] - 2*X[i, j] == D[i, j])
            else:
                constraints.append(X[i, i] + X[j, j] - 2*X[i, j] >= D[i, j] * (1 - margin))
                constraints.append(X[i, i] + X[j, j] - 2*X[i, j] <= D[i, j] * (1 + margin))

    # solve and print
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    return X.value


def calculate_points(X: numpy.ndarray, num_spatial_dims: int = 2):
    eigen_values, eigen_vectors = numpy.linalg.eig(X)

    D = numpy.zeros((num_spatial_dims, num_spatial_dims))
    numpy.fill_diagonal(D, eigen_values[:num_spatial_dims])
    D_sqrt = numpy.sqrt(D)

    U = eigen_vectors[:, :num_spatial_dims]

    P = D_sqrt @ U.T
    return P


def visualize_points(P: numpy.ndarray, names: Optional[numpy.ndarray]):
    figure = plt.figure(figsize = (5, 5))
    axis = plt.axes()
    
    # Creating plot
    axis.scatter(*P, s=20, color="blue", marker="^")
    for point, name in zip(P.T, names):
        axis.text(*point, name)

    axis.set_title("Optimial positions")
    axis.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    D = read_d_from_dist_csv(os.path.join(dataset_path, "dist.csv"))
    omega = read_omega_from_csv(os.path.join(dataset_path, "omega.csv"), sparsity=0.0)
    names = read_names_from_csv(os.path.join(dataset_path, "names.csv"))
    
    # find gram matrix
    X = optimize_gram_matrix(D, omega, margin=0.05)  # 0.05 for mass, 0.78 for tufts
    if X is None: raise ValueError("Failed to optimize gram matrix")

    # find points from gram matrix
    P = calculate_points(X, num_spatial_dims=2)

    # visualize points
    visualize_points(P, names)
