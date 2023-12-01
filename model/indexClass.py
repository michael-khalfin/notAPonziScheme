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

    def __init__(self, period = 59, lags = 60, distance_metric = 0, num_groups = 5, selection_metric = 0, weight_metric = 0):
        self.period = period
        self.lags = lags

        self.capitalizations = pd.read_csv("data/capitalizations_all_periods.csv")
        self.largest_indices, self.largest_elems = self.get_k_largest(self.capitalizations[str(self.period)].to_numpy(), 30)
        self.covariance_matrix = self.get_covariance_matrix()
        self.expected_returns = self.get_expected_returns()
        self.correlation_matrix = self.make_correlation_matrix()
        self.distance_matrix = self.make_dist_array(self.correlation_matrix, distance_metric = distance_metric)
        
        n = self.distance_matrix.shape[0]
        self.num_groups = num_groups

        self.env = gp.Env()
        self.clusters = self.create_clusters(self.distance_matrix, n, num_groups)

        self.stocks = self.select_stocks(selection_metric = selection_metric)
        self.small_covariance = self.index_covariance()
        self.index_weights = self.compute_weights(weight_metric = weight_metric)
        self.value_weights = self.compute_benchmark_weights_value()
        self.uniform_weights = self.compute_benchmark_weights_uniform()
        self.next_performance = self.capitalizations[str(self.period + 1)].to_numpy()[self.largest_indices]
        self.uniform_beta, self.value_beta, self.index_beta = self.compute_beta()
        self.uniform_value, self.weighted_value, self.index_value = self.determine_value()
        self.uniform_volatility, self.weighted_volatility, self.index_volatility = self.get_volatility()

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
        largest_elems = sorted(capitalizations, reverse=True)[:k]
        indices = []
        for elem in largest_elems:
            i = np.where(capitalizations == elem)
            indices.append(i[0][0])
        indices = sorted(indices, reverse=False)
        return indices, largest_elems
    
    def get_expected_returns(self):
        """
        Retrieve and process the expected returns based on the selected period.

        Returns:
        - numpy.ndarray: The expected returns 
        """
        expected_returns = pd.read_csv(f"data/expected_returns/expected_return_{self.period+1}.csv", header=None, encoding='utf-8')
        expected_returns = expected_returns.astype(float)
        expected_returns = expected_returns.to_numpy()
        return expected_returns[self.largest_indices]

    def get_covariance_matrix(self):
        """
        Retrieve and process the covariance matrix based on the selected period.

        Returns:
        - numpy.ndarray: The covariance matrix.
        """
        if self.lags == 60:
            covariance_matrix = pd.read_csv(f"data/covariance_matrices/60_periods/covariance_matrix_{self.period+1}_1.csv", header=None, encoding='utf-8')
        elif self.lags == 30:
            covariance_matrix = pd.read_csv(f"data/covariance_matrices/30_periods/covariance_matrix_{self.period+1}_30.csv", header=None, encoding='utf-8')
        elif self.lags == 12:
            covariance_matrix = pd.read_csv(f"data/covariance_matrices/12_periods/covariance_matrix_{self.period+1}_48.csv", header=None, encoding='utf-8')

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
                dist += abs(v1[i]-v2[i])
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
        model.setParam("LogToConsole", 0)

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
                caps = []
                caps = [self.largest_elems[i] for i in cluster]
                i = np.where(self.largest_elems == min(caps))
                kstocks.append(i[0][0])
        else:
            for cluster in self.clusters:
                caps = []
                caps = [self.largest_elems[i] for i in cluster]
                caps = np.sort(caps)
                i = np.where(self.largest_elems == caps[int(len(caps) / 2)])
                kstocks.append(i[0][0])
        return kstocks
    
    def index_covariance(self):
        """
        Method that removes the unselected stocks from the index's covariance matrix

        Parameters:

        Returns:
        - numpy.ndarray: The covariance matrix with k rows and columns
        """
        covariance_matrix = self.covariance_matrix
        covariance_matrix = covariance_matrix[self.stocks]
        covariance_matrix = covariance_matrix[:, self.stocks]
        return covariance_matrix
    
    def compute_weights(self, weight_metric):
        """
        Calculates the portfolio weights of the representative stocks dependent on the portfolio we are tracking

        Parameters:
        - weight_metric (int): The metric used to calculate the value.

        Returns:
        - List[float]: The portfolio weights for our representative stocks
        - numpy.ndarray: The portfolio weights for the benchmark
        """
        # Use the expected returns for the predictions, not the capitalizations
        if weight_metric == 0:
            cluster_weights = []
            total = 0
            for stock in self.expected_returns:
                total += stock
            for cluster in self.clusters:
                cluster_total = 0
                for i in cluster:
                    cluster_total += self.expected_returns[i]
                cluster_weights.append(cluster_total / total)
            cluster_weights = np.array(cluster_weights)
        if weight_metric == 1:
            cluster_weights = [len(self.clusters[i]) / 30 for i in range(len(self.clusters))]
            cluster_weights = np.reshape(np.array(cluster_weights), (self.num_groups, 1))
        return cluster_weights
    
    
    def compute_benchmark_weights_value(self):
        """
        Calculate the value benchmark portfolio weights 

        Parameters:

        Returns:
        - numpy.ndarray: The portfolio weights for the value weighted benchmark
        """
        benchmark_weights = (1 / (np.transpose(np.ones(30)) @ np.linalg.inv(self.covariance_matrix) @ self.expected_returns)) * np.linalg.inv(self.covariance_matrix) @ self.expected_returns
        return benchmark_weights

    def compute_benchmark_weights_uniform(self):
        """
        Calculate the uniform benchmark portfolio weights 

        Parameters:

        Returns:
        -numpy.ndarry: The portfolio weights for the uniform weighted benchmark
        """
        return np.reshape(np.array([(1 / 30) for i in range(30)]), (30, 1))
    
    def determine_value(self):
        """
        Calculates the value of the index for a given weight metric.

        Parameters:

        Returns:
        - float: The value of the uniform benchmark.
        - float: The value of the weighted benchmark.
        - float: The value of the index
        """
        index_value = 0
        uniform_value = 0
        weighted_value = 0
        i = 0
        for weight in self.uniform_weights:
            uniform_value += weight * self.next_performance[i]
            i += 1
        i = 0
        for weight in self.value_weights:
            weighted_value += weight * self.next_performance[i]
            i += 1
        i = 0
        for stock in self.stocks:
            index_value += self.index_weights[i] * self.next_performance[stock]
            i += 1
        return uniform_value.item(), weighted_value.item(), index_value.item()

    def compute_beta(self):
        """
        Calculates the betas relative to the benchmarks

        Parameters:

        Returns:
        - numpy.ndarray: beta relative to our uniform benchmark
        - numpy.ndarray: beta relative to our value benchmark
        - numpy.ndarray: beta for our index
        """
        value_beta = (self.covariance_matrix @ self.value_weights) / (np.transpose(self.value_weights) @ self.covariance_matrix @ self.value_weights)
        uniform_beta = (self.covariance_matrix @ self.uniform_weights) / (np.transpose(self.uniform_weights) @ self.covariance_matrix @ self.uniform_weights)
        # I'm not sure about how we should get the betas when we only have the 5 stocks
        index_beta = (self.small_covariance @ self.index_weights) / (np.transpose(self.index_weights) @ self.small_covariance @ self.index_weights)
        return uniform_beta, value_beta, index_beta
 
    def get_volatility(self):
        """
        Calculate the expected volatility of each portfolio

        Parameters:

        Returns:
        - numpy.ndarray: volatility for the uniform benchmark
        - numpy.ndarray: volatility for the weighted benchmark
        - numpy.ndarray: volatility for the index
        """
        value_volatility = np.transpose(self.value_weights) @ self.covariance_matrix @ self.value_weights
        uniform_volatility =  np.transpose(self.uniform_weights) @ self.covariance_matrix @ self.uniform_weights
        # Again, I'm not sure how we want to do this with the index
        index_volatility = np.transpose(self.index_weights) @ self.small_covariance @ self.index_weights
        return uniform_volatility.item(), value_volatility.item(), index_volatility.item()

