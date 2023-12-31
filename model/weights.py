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

def variance_value_plot(benchmark_values, benchmark_volatilities, values, volatilities, name1):
    """
    
    """
    volatilities.append(benchmark_volatilities[0])
    volatilities.append(benchmark_volatilities[1])
    values.append(benchmark_values[0])
    values.append(benchmark_values[1])
    volatilities_normalized, values_normalized, closest_weighted, furthest_weighted, closest_uniform, furthest_uniform = find_closest_points(volatilities, values)
    # Green is uniform red is value
    # closest = 7 = [1, 0, 0], furthest = 14 = [2, 0, 1]
    plt.style.use('dark_background')
    plt.scatter(volatilities_normalized, values_normalized, c = "blue", s = 75, label = "Other Input Combinations")
    plt.scatter(volatilities_normalized[closest_weighted], values_normalized[closest_weighted], c="salmon", s = 75, label = "Closest to Weighted")
    plt.scatter(volatilities_normalized[furthest_weighted], values_normalized[furthest_weighted], c="orange", s = 75, label = "Furthest From Weighted")
    plt.scatter(volatilities_normalized[closest_uniform], values_normalized[closest_uniform], c="yellowgreen", s = 75, label = "Closest to Uniform")
    plt.scatter(volatilities_normalized[furthest_uniform], values_normalized[furthest_uniform], c="yellow", s = 75, label = "Furthest From Uniform")
    plt.scatter(volatilities_normalized[-2], values_normalized[-2], c= "green", s = 75, label = "Uniform Benchmark")
    plt.scatter(volatilities_normalized[-1], values_normalized[-1], c="red", s = 75, label = "Weighted Benchmark")
    plt.title("Different Possible Inputs")
    plt.xlabel("Volatilities")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(name1)
    plt.clf()
    return closest_weighted, furthest_weighted, closest_uniform, furthest_uniform


def find_closest_points(volatilities, values):
    """
    """
    volatilities -= np.mean(volatilities)
    volatilities /= np.std(volatilities)
    values -= np.mean(values)
    values /= np.std(values)
    weighted_point = (volatilities[-1],values[-1])
    uniform_point = (volatilities[-2], values[-2])
    uniform_closest = [None, float('inf')]
    uniform_furthest = [None, float('-inf')]
    weighted_closest = [None, float('inf')]
    weighted_furthest = [None, float('-inf')]
    for i in range(len(values) - 2):
        uniform_distance = math.sqrt((volatilities[i] - uniform_point[0]) ** 2 + (values[i] - uniform_point[1]) ** 2)
        weighted_distance = math.sqrt((volatilities[i] - weighted_point[0]) ** 2 + (values[i] - weighted_point[1]) ** 2)
        if uniform_distance <= uniform_closest[1]:
            uniform_closest = [i, uniform_distance]
        if uniform_distance >= uniform_furthest[1]:
            uniform_furthest = [i, uniform_distance]
        if weighted_distance <= weighted_closest[1]:
            weighted_closest = [i, weighted_distance]
        if weighted_distance >= weighted_furthest[1]:
            weighted_furthest = [i, weighted_distance]
    return volatilities, values, weighted_closest[0], weighted_furthest[0], uniform_closest[0], uniform_furthest[0]


def find_highest_sharpe_inputs(num_groups):
    sharpes = [[] for i in range(11)]
    for i in range(3):
        for j in range(3): 
            for k in range(2):
                for h in range(11):
                    testCase = indexClass.index(period = 59 + h, distance_metric = i, num_groups = num_groups, selection_metric = j, weight_metric = k)
                    sharpes[h].append(testCase.get_sharpe()[2])
                    # sharpes11.append(testCase11.get_sharpe()[2])
                    # testCase12 = indexClass.index(period = 59, distance_metric = i, num_groups = 10, selection_metric = j, weight_metric = k)
                    # sharpes12.append(testCase12.get_sharpe()[2])
                    # testCase13 = indexClass.index(period = 59, distance_metric = i, num_groups = 15, selection_metric = j, weight_metric = k)
                    # sharpes13.append(testCase13.get_sharpe()[2])
                    # testCase21 = indexClass.index(period = 60, distance_metric = i, num_groups = 5, selection_metric = j, weight_metric = k)
                    # sharpes21.append(testCase21.get_sharpe()[2])
                    # testCase22 = indexClass.index(period = 60, distance_metric = i, num_groups = 10, selection_metric = j, weight_metric = k)
                    # sharpes22.append(testCase22.get_sharpe()[2])
                    # testCase23 = indexClass.index(period = 60, distance_metric = i, num_groups = 15, selection_metric = j, weight_metric = k)
                    # sharpes23.append(testCase23.get_sharpe()[2])
    max = float("-inf")
    for sharpe in range(len(sharpes[0])):
        total = 0
        for period in sharpes:
            total += (period[sharpe])
        total /= 12
        if total > max:
            max = total
            max_index = sharpe
        # if (sharpes[sharpe] + sharpes21[sharpe]) / 2 > max5:
        #     max5_index = sharpe
        #     max5 = (sharpes11[sharpe] + sharpes21[sharpe]) / 2
        # if (sharpes12[sharpe] + sharpes22[sharpe]) / 2 > max10:
        #     max10_index = sharpe
        #     max10 = (sharpes12[sharpe] + sharpes22[sharpe]) / 2
        # if (sharpes13[sharpe] + sharpes23[sharpe]) / 2 > max15:
        #     max15_index = sharpe
        #     max15 = (sharpes13[sharpe] + sharpes23[sharpe]) / 2
    return max_index, max

def index_to_inputs(index):
    distance_metric = math.floor(index / 6)
    index = index % 6
    selection_metric = math.floor(index / 3)
    index = index % 3
    weight_metric = index
    return distance_metric, selection_metric, weight_metric

def decide_inputs(index):
    return index_to_inputs(index)



if __name__ == '__main__':
    # inputs = []
    # benchmark_betas_1, benchmark_values_1, benchmark_volatilities_1, betas_1, values_1, volatilities_1 = first_period(5, 0)
    # inputs.append(variance_value_plot(benchmark_values_1, benchmark_volatilities_1, values_1, volatilities_1, "period1group5Normalized"))
    # benchmark_betas_2, benchmark_values_2, benchmark_volatilities_2, betas_2, values_2, volatilities_2 = first_period(10, 0)
    # inputs.append(variance_value_plot(benchmark_values_2, benchmark_volatilities_2, values_2, volatilities_2, "period1group10Normalized"))
    # benchmark_betas_3, benchmark_values_3, benchmark_volatilities_3, betas_3, values_3, volatilities_3 = first_period(15, 0)
    # inputs.append(variance_value_plot(benchmark_values_3, benchmark_volatilities_3, values_3, volatilities_3, "period1group15Normalized"))
    # print(inputs)
    # [0, 0, 0]
    # [2, 0, 0]
    # [1, 0, 1] This one will be bad
    # 
    max_index_5, max_5 = find_highest_sharpe_inputs(5)
    inputs_5 = decide_inputs(max_index_5)
    max_index_10, max_10 = find_highest_sharpe_inputs(10)
    inputs_10 = decide_inputs(max_index_10)
    max_index_15, max_15 = find_highest_sharpe_inputs(15)
    inputs_15 = decide_inputs(max_index_15)
    print(max_5, inputs_5)
    print(max_10, inputs_10)
    print(max_15, inputs_15)
    sharpes_value = 0
    sharpes_uniform = 0
    for h in range(11):
        testCase = indexClass.index(period = 59 + h)
        sharpes_uniform += testCase.get_sharpe()[0]
        sharpes_value += testCase.get_sharpe()[1]
    print(sharpes_uniform / 11)
    print(sharpes_value / 11)
    



