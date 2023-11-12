import os
os.environ["GUROBI_LICENSE_KEY"] = "/Users/michael_khalfin/gurobi.lic"

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os

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

    model = gp.Model("MIP_Model")
    x = model.addVars(n, num_groups, vtype=GRB.BINARY, name="x")

    obj_expr = gp.LinExpr()
    for i in range(n):
        for j in range(n):
            for k in range(6):
                obj_expr += x[i, k] * x[j, k] * distance_matrix[i][j]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Each element is assigned to exactly one cluster
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(6)) == 1)

    # Add constraints: Each cluster has exactly n/6 elements
    for k in range(6):
        model.addConstr(gp.quicksum(x[i, k] for i in range(n)) == n // 6)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        assignment = {(i, k): x[i, k].x for i in range(n) for k in range(6) if x[i, k].x > 0.5}
        print("Element assignments to clusters:")
        for i, k in assignment:
            print(f"Element {i} is assigned to cluster {k}")
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    covariance_matrix = pd.read_csv("data/prob_correct_covariance_matrix.csv", header=None, encoding='utf-8')
    covariance_matrix = covariance_matrix.iloc[1:, 1:]
    covariance_matrix = covariance_matrix.astype(float)
    covariance_matrix = covariance_matrix.to_numpy()

    # Convert covariance matrix to correlation matrix
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    print(correlation_matrix)
    print(correlation_matrix.shape)

    distance_matrix = make_dist_array(correlation_matrix)
    n = distance_matrix.shape[0]
    num_groups = 5

    env = gp.Env()
    #clusters = create_clusters(distance_matrix, n, num_groups)