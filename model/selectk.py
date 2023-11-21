import os
os.environ["GUROBI_LICENSE_KEY"] = "/Users/michael_khalfin/gurobi.lic"

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def get_k_largest(capitalizations, k):
    """
    Get the indices and corresponding elements of the k largest values in an array.

    Parameters:
    - capitalizations (numpy.ndarray): The array containing capitalization values.
    - k (int): The number of largest values to retrieve.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the indices and corresponding elements
      of the k largest values.
    """
    largest_elems = sorted(capitalizations, reverse=False)[:30]
    indices = []
    for elem in largest_elems:
        i = np.where(capitalizations == elem)
        indices.append(i[0][0])
    indices = sorted(indices, reverse=False)
    return indices, largest_elems

def make_dist_array(correlation_matrix):
    """
    Transform a correlation matrix to a distance matrix.

    Parameters:
    - correlation_matrix (numpy.ndarray): The input correlation matrix.

    Returns:
    - numpy.ndarray: The resulting distance matrix.
    """
    distance_matrix = np.zeros_like(correlation_matrix)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            distance_matrix[i, j] = distance_bw_vectors(correlation_matrix[i], correlation_matrix[j])
    return distance_matrix

def distance_bw_vectors(v1, v2):
    """
    Compute the distance between two vectors.

    Parameters:
    - v1 (numpy.ndarray): The first vector.
    - v2 (numpy.ndarray): The second vector.

    Returns:
    - float: The Euclidean distance between the two vectors.
    """
    dist = 0
    for x in v1:
        for y in v2:
            dist += (x-y) ** 2
    dist = dist**(1/2)
    return dist

def create_clusters(distance_matrix, n, num_groups):
    """
    Create clusters using an integer programming model.

    Parameters:
    - distance_matrix (numpy.ndarray): The distance matrix.
    - n (int): The number of elements.
    - num_groups (int): The desired number of clusters.

    Returns:
    - List[List[int]]: A list of clusters, where each cluster is represented by a list of indices.
    """
    model = gp.Model("IP_Model")

    # Initializes binary variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Minimize the distance within the clusters
    obj_expr = gp.LinExpr()
    for j in range(n):
        for i in range(n):
            obj_expr += distance_matrix[i][j] * x[i,j]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Choose num_groups centroids
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == num_groups)

    # Each object must be represented by one centroid
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

    # i is represented by j only if j is a centroid
    for i in range(n):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j])

    # Each centroid has n/num_groups elements associated with it
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) <= n/num_groups)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        cluster_points = [j for j in range(n) if y[j].x > 0.5]
        clusters = []
        for point in cluster_points:
            cluster = [i for i in range(n) for j in range(n) if (x[i,j].x > 0.5 and j==point)]
            clusters.append(cluster)

        return clusters
    else:
        print("No feasible solution found.")
        return []

if __name__ == '__main__':
    period = 0

    capitalizations = pd.read_csv("data/capitalizations.csv")
    largest_indices, largest_elems = get_k_largest(capitalizations[str(period)].to_numpy(), 30)

    covariance_matrix = pd.read_csv("data/prob_correct_covariance_matrix.csv", header=None, encoding='utf-8')
    covariance_matrix = covariance_matrix.iloc[1:, 1:]
    covariance_matrix = covariance_matrix.astype(float)
    covariance_matrix = covariance_matrix.to_numpy()
    covariance_matrix = covariance_matrix[largest_indices]
    covariance_matrix = covariance_matrix[:, largest_indices]

    # Convert covariance matrix to correlation matrix
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)

    distance_matrix = make_dist_array(correlation_matrix)
    n = distance_matrix.shape[0]
    num_groups = 5

    env = gp.Env()
    clusters = create_clusters(distance_matrix, n, num_groups)

    kstocks = []
    for cluster in clusters:
        caps = []
        caps = [largest_elems[i] for i in cluster]
        i = np.where(largest_elems == max(caps))
        kstocks.append(i[0][0])
    print(kstocks)