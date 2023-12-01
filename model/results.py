import indexClass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def plot_lags(indices):
    # 0, 0, 0 - 5 groups, replicating weighted performance
    # 2, 0, 0 - 10 groups, replicating weighted performance
    # 1, 2, 0 - 5 groups, replicating uniform performance
    # 1, 0, 1 - 10 groups, furthest from weighted and uniform

    dates = pd.read_csv("data/market_return.csv")['Date'].tolist()
    
    # 0: uniform benchmark
    # 1: weighted benchmark
    lags = {0: [], 1: [], 12: [], 30: [], 60: []}
    for j in {12, 30, 60}:
        for i in range(11):
            lags[j].append(float(indexClass.index(period = 59 + i, lags = j, distance_metric = 1, 
                                                  num_groups = 10, selection_metric = 0, 
                                                  weight_metric = 1).determine_value()[2]))
    for j in {0, 1}:
        for i in range(11):
            lags[j].append(float(indexClass.index(period = 59 + i, lags = 60, num_groups = 5).determine_value()[j]))

    plt.figure(figsize=(13, 6))

    if 0 in indices:
        plt.plot(dates[60:71], lags[0], marker='o', linestyle='--', color='#D8BFD8', label='Uniform Benchmark')

    if 1 in indices:
        plt.plot(dates[60:71], lags[1], marker='o', linestyle='--', color='#E6E6FA', label='Weighted Benchmark')

    if 12 in indices:
        plt.plot(dates[60:71], lags[12], marker='o', linestyle='-', color='blue', label='Lag 12')
    
    if 30 in indices:
        plt.plot(dates[60:71], lags[30], marker='o', linestyle='-', color='green', label='Lag 30')
    
    if 60 in indices:
        plt.plot(dates[60:71], lags[60], marker='o', linestyle='-', color='red', label='Lag 60')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Index Portfolio Values Over Time (10 Groups, Furthest From Weighted and Uniform)')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_num_groups():
    dates = pd.read_csv("data/market_return.csv")['Date'].tolist()
    num_groups = {5: [], 10: [], 15: []}
    for j in {5, 10, 15}:
        for i in range(11):
            num_groups[j].append(float(indexClass.index(period = 59 + i, lags = 30, distance_metric = 0, 
                                                        num_groups = j, selection_metric = 0,
                                                        weight_metric = 0).determine_value()[2]))
            
    bar_width = 0.2  # Adjust as needed
    bar_positions = np.arange(len(dates[60:71]))

    plt.figure(figsize=(15, 6))

    # Plot bars
    for i, j in enumerate([5, 10, 15]):
        plt.bar(bar_positions + i * bar_width, num_groups[j], width=bar_width, label=f'{j} Groups')

    # Plot lines
    plt.plot(bar_positions + 0.5 * bar_width, num_groups[5], linestyle='dashed', dashes=[8, 4], marker='o', color='#66B2FF')
    plt.plot(bar_positions + 1.5 * bar_width, num_groups[10], linestyle='dashed', dashes=[8, 4], marker='o', color='#FFB366')
    plt.plot(bar_positions + 2.5 * bar_width, num_groups[15], linestyle='dashed', dashes=[8, 4], marker='o', color='#8FED8F')

    plt.xticks(bar_positions + bar_width, dates[60:71])

    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Index Portfolio Values Over Time Varying Num of Groups')
    plt.legend()
    plt.show()

def plot_matrices():
    matrices = {}
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    i = 0
    for j in {12, 30, 60}:
        matrices[j] = indexClass.index(period = 59, lags = j, num_groups = 5).get_covariance_matrix()
        np.fill_diagonal(matrices[j], np.nan)

        sns.heatmap(matrices[j], annot=False, cmap='inferno', cbar=False, xticklabels=False, 
                    yticklabels=False, ax=axes[i])
        axes[i].set_title(f'Covariance Matrix Heatmap Lags = {j}')
        i += 1

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_lags([0, 1, 12, 30, 60])
    #plot_matrices()
    #plot_num_groups()