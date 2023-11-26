import os
os.environ["GUROBI_LICENSE_KEY"] = "/Users/michael_khalfin/gurobi.lic"

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

class index:
    """
    Represents an index and provides methods for creating clusters, 
    selecting stocks, and valuing the index based on various metrics.

    Parameters:
    - period (int): The time period for financial data.

    - distance_metric (int): The distance metric used for clustering.
        - 0: Distance between correlation vectors.
        - 1: ...

    - num_groups (int): The desired number of clusters.
        - Defualt: 5

    - selection_metric (int): The metric used for stock selection.
        - 0: Capitalization.

    - weight_metric (int): The metric used for weighting the stocks.
        - 0: Capitalization.
    """

    def __init__(self, period = 59, distance_metric = 0, num_groups = 5, selection_metric = 0, weight_metric = 0):
        self.period = period

        self.capitalizations = pd.read_csv("data/capitalizations_all_periods.csv")
        self.largest_indices, self.largest_elems = self.get_k_largest(self.capitalizations[str(self.period)].to_numpy(), 30)
        self.covariance_matrix = self.get_covariance_matrix()
        self.correlation_matrix = self.make_correlation_matrix()
        self.distance_matrix = self.make_dist_array(self.correlation_matrix, distance_metric = distance_metric)
        
        n = self.distance_matrix.shape[0]
        self.num_groups = num_groups

        self.env = gp.Env()
        self.clusters = self.create_clusters(self.distance_matrix, n, num_groups)

        self.stocks = self.select_stocks(selection_metric = selection_metric)
        self.value = self.determine_value(weight_metric = weight_metric)

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
    
    def get_covariance_matrix(self):
        """
        Retrieve and process the covariance matrix based on the selected period.

        Returns:
        - numpy.ndarray: The covariance matrix.
        """
        covariance_matrix = pd.read_csv(f"data/covariance_matrices/covariance_matrix_{self.period+1}.csv", header=None, encoding='utf-8')
        covariance_matrix = covariance_matrix.astype(float)
        covariance_matrix = covariance_matrix.to_numpy()
        covariance_matrix = covariance_matrix[self.largest_indices]
        covariance_matrix = covariance_matrix[:, self.largest_indices]
        return covariance_matrix

    def make_correlation_matrix(self):
        """
        Calculate the correlation matrix from the covariance matrix.

        Returns:
        - numpy.ndarray: The correlation matrix.
        """
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        return (self.covariance_matrix / np.outer(std_devs, std_devs))
    
    def make_dist_array(self, correlation_matrix, distance_metric):
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
                distance_matrix[i, j] = self.distance_bw_vectors(correlation_matrix[i], correlation_matrix[j], distance_metric)
        return distance_matrix
    
    def distance_bw_vectors(self, v1, v2, distance_metric):
        """
        Compute the distance between two vectors.

        Parameters:
        - v1 (numpy.ndarray): The first vector.
        - v2 (numpy.ndarray): The second vector.
        - distance_metric: An integer denoting how the vectors similarity is computed

        Returns:
        - float: The distance between the two vectors.
        """
        dist = 0
        if distance_metric == 0:
            for i in range(len(v1)):
                dist += (v1[i]-v2[i]) ** 2
            dist = dist**(1/2)
        elif distance_metric == 1:
            for i in range(len(v1)):
                dist += math.abs(v1[i]-v2[i])
        else:
            dot_product = 0
            x_norm = 0 
            y_norm = 0
            for i in range(len(v1)):
                x_norm += v1[i] ** 2
                y_norm += v2[i] ** 2
                dot_product += v1[i] * v2[i]
            dist = dot_product / (math.sqrt(x_norm) * math.sqrt(x_norm))
        return dist
    
    def create_clusters(self, distance_matrix, n, num_groups):
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
    
    def select_stocks(self, selection_metric):
        """
        Select stocks based on the chosen metric within the created clusters.

        Parameters:
        - selection_metric (int): The metric used for stock selection.

        Returns:
        - List[int]: A list of selected stocks.
        """
        kstocks = []
        if selection_metric == 0:
            for cluster in self.clusters:
                caps = []
                caps = [self.largest_elems[i] for i in cluster]
                i = np.where(self.largest_elems == max(caps))
                kstocks.append(i[0][0])
        elif selection_metric == 1:
            for cluster in self.clusters:
                pass
        return kstocks
    
    def determine_value(self, weight_metric):
        """
        Calculates the value of the index for a given weight metric.

        Parameters:
        - weight_metric (int): The metric used to calculate the value.

        Returns:
        - float: The value of the index for the given weight metric.
        """
        value = 0

        if weight_metric == 0:
            total = 0
            for i in self.stocks:
                total += self.largest_elems[i]
            
            for i in self.stocks:
                value += ((self.largest_elems[i] / total) * self.largest_elems[i])

        if weight_metric == 1:
            amt = len(self.stocks)
            for i in self.stocks:
                value += ((1 / amt) * self.largest_elems[i])

        return value
    
x = index()
