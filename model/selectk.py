import pandas as pd
import numpy as np
import gurobipy as gp

def make_dist_array(data):
    arr = [[]*len(data)]
    for i in range(len(data)):
        for j in range(len(data)):
            arr[i].append(distance_bw_vectors(data[i], data[j]))
    return arr

def distance_bw_vectors(v1, v2):
    dist = 0
    for x in v1:
        for y in v2:
            dist += (x-y) ** 2
    dist = dist**(1/2)
    return dist

if __name__ == '__main__':
    covariance_matrix = pd.read_csv("data/new_covariance_matrix.txt", header=None, encoding='utf-8')
    print(covariance_matrix)
    # covariance_matrix = covariance_matrix.to_numpy()

    # Convert covariance matrix to correlation matrix
    #std_devs = np.sqrt(np.diag(covariance_matrix))
    #correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    #print(correlation_matrix)

    #dist = make_dist_array(correlation_matrix)
    #print(dist)

    #model = gp.Model("IP_Model")