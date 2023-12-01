import indexClass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def plot_lags():
    dates = pd.read_csv("data/market_return.csv")['Date'].tolist()
    
    lags = {12: [], 30: [], 60: []}
    for j in {12, 30, 60}:
        for i in range(11):
            lags[j].append(float(indexClass.index(period = 59 + i, lags = j, num_groups = 5).determine_value()[2][0]))
    plt.figure(figsize=(10, 6))
    plt.plot(dates[60:71], lags[12], marker='o', linestyle='-', color='blue', label='Lag 12')
    plt.plot(dates[60:71], lags[30], marker='o', linestyle='-', color='green', label='Lag 30')
    plt.plot(dates[60:71], lags[60], marker='o', linestyle='-', color='red', label='Lag 60')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Index Portfolio Values Over Time')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
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
    plot_lags()
    #plot_matrices()