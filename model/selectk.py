import os
os.environ["GUROBI_LICENSE_KEY"] = "/Users/michael_khalfin/gurobi.lic"

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os

def get_k_largest(capitalizations, k):
    largest_elems = sorted(capitalizations, reverse=False)[:30]
    indices = []
    for elem in largest_elems:
        i = np.where(capitalizations == elem)
        indices.append(i[0][0])
    indices = sorted(indices, reverse=False)
    return indices, largest_elems

def make_dist_array(correlation_matrix):
    # Transform correlation matrix to distances
    distance_matrix = np.zeros_like(correlation_matrix)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            distance_matrix[i, j] = distance_bw_vectors(correlation_matrix[i], correlation_matrix[j])
    return distance_matrix

def distance_bw_vectors(v1, v2):
    dist = 0
    for x in v1:
        for y in v2:
            dist += (x-y) ** 2
    dist = dist**(1/2)
    return dist

def create_clusters(distance_matrix, n, num_groups):

    model = gp.Model("IP_Model")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    obj_expr = gp.LinExpr()
    for j in range(n):
        for i in range(n):
            obj_expr += distance_matrix[i][j] * x[i,j]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Choose num_groups centroids
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == num_groups)

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    for i in range(n):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print([(i,j) for i in range(n) for j in range(n) if x[i,j].x > 0.5])
        #assignment = {(i, k): x[i, k].x for i in range(n) for k in range(6) if x[i, k].x > 0.5}
        #print("Element assignments to clusters:")
        #for i, k in assignment:
        #    print(f"Element {i} is assigned to cluster {k}")
    else:
        print("No feasible solution found.")

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
    create_clusters(distance_matrix, n, num_groups)