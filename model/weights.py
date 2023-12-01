import indexClass
import matplotlib.pyplot as plt
import numpy as np
import math


def first_period(num_groups, period):
    """
    
    """
    betas = []
    values = []
    volatilities = []
    for i in range(3):
        for j in range(3): 
            for k in range(2):
                testCase = indexClass.index(period = 59 + period, distance_metric = i, num_groups = num_groups, selection_metric = j, weight_metric = k)
                betas.append(testCase.compute_beta()[2])
                values.append(testCase.determine_value()[2])
                volatilities.append(testCase.get_volatility()[2])
    benchmarks = indexClass.index(period = 59, distance_metric = i, num_groups = 5, selection_metric = j, weight_metric = k)
    benchmark_values = benchmarks.determine_value()[:2]
    benchmark_volatilities = benchmarks.get_volatility()[:2]
    benchmark_betas =  benchmarks.compute_beta()[:2]
    return benchmark_betas, benchmark_values, benchmark_volatilities, betas, values, volatilities

def variance_value_plot(benchmark_values, benchmark_volatilities, values, volatilities, name1, name2):
    """
    
    """
    volatilities.append(benchmark_volatilities[0])
    volatilities.append(benchmark_volatilities[1])
    values.append(benchmark_values[0])
    values.append(benchmark_values[1])
    volatilities_normalized, values_normalized, closest = find_closest_points(volatilities, values)
    colors = ["blue" for i in range(18)]
    # Green is uniform
    colors.append("green")
    # Red is value
    colors.append("red")
    colors[closest] = ("purple")
    plt.scatter(volatilities, values, c=colors, s = 75, label = ["blue", "green", "red", "purple"])
    plt.title("Different Possible Inputs")
    plt.xlabel("Volatilities")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(name1)
    plt.clf()
    plt.scatter(volatilities_normalized, values_normalized, c=colors, s = 75, label = ["blue", "green", "red", "purple"])
    plt.title("Different Possible Inputs Normalized")
    plt.xlabel("Volatilities")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(name2)
    plt.clf()

def find_closest_points(volatilities, values):
    """
    """
    volatilities -= np.mean(volatilities)
    volatilities /= np.std(volatilities)
    values -= np.mean(values)
    values /= np.std(values)
    weighted_point = (volatilities[-1],values[-1])
    uniform_point = (volatilities[-2], values[-2])
    closest = [None, float('inf')]
    for i in range(len(values) - 2):
        distance = math.sqrt((volatilities[i] - weighted_point[0]) ** 2 + (values[i] - weighted_point[1]) ** 2) + math.sqrt((volatilities[i] - uniform_point[0]) ** 2 + (values[i] - uniform_point[1]) ** 2)
        if distance < closest[1]:
            closest = [i, distance]
    return volatilities, values, closest[0]



if __name__ == '__main__':
    benchmark_betas_1, benchmark_values_1, benchmark_volatilities_1, betas_1, values_1, volatilities_1 = first_period(5, 0)
    variance_value_plot(benchmark_values_1, benchmark_volatilities_1, values_1, volatilities_1, "period1group5","period1group5Normalized")
    benchmark_betas_2, benchmark_values_2, benchmark_volatilities_2, betas_2, values_2, volatilities_2 = first_period(10, 0)
    variance_value_plot(benchmark_values_2, benchmark_volatilities_2, values_2, volatilities_2, "period1group10", "period1group10Normalized")
    benchmark_betas_3, benchmark_values_3, benchmark_volatilities_3, betas_3, values_3, volatilities_3 = first_period(15, 0)
    variance_value_plot(benchmark_values_3, benchmark_volatilities_3, values_3, volatilities_3, "period1group15", "period1group15Normalized")

