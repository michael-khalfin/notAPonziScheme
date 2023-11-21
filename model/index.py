import os
os.environ["GUROBI_LICENSE_KEY"] = "/Users/michael_khalfin/gurobi.lic"

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class index:

    def __init__(self, period = 0, distance_metric = 0, selection_metric = 0, weight_metric = 0):
        self.period = period
        self.distance_metric = distance_metric
        self.selection_metric = selection_metric
        self.weight_metric = weight_metric

        self.capitalizations = pd.read_csv("data/capitalizations.csv")
        self.largest_indices, self.largest_elems = self.get_k_largest(self.capitalizations[str(self.period)].to_numpy(), 30)
        self.covariance_matrix = self.get_covariance_matrix()
        self.correlation_matrix = self.make_correlation_matrix()
    
    def get_covariance_matrix(self):
        self.covariance_matrix = pd.read_csv(f"data/covariance_matrix{self.period+60}.csv", header=None, encoding='utf-8')
        self.covariance_matrix = self.covariance_matrix.astype(float)
        self.covariance_matrix = self.covariance_matrix.to_numpy()
        self.covariance_matrix = self.covariance_matrix[self.largest_indices]
        self.covariance_matrix = self.covariance_matrix[:, self.largest_indices]

    def make_correlation_matrix(self):
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        self.correlation_matrix = self.covariance_matrix / np.outer(std_devs, std_devs)
        
    def get_k_largest(self, capitalizations, k):
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
    
    def make_dist_array(self, correlation_matrix):
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
                distance_matrix[i, j] = self.distance_bw_vectors(correlation_matrix[i], correlation_matrix[j])
        return distance_matrix

    def distance_bw_vectors(self, v1, v2):
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