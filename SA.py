import math
import random

import pandas as pd
import numpy as np
import os

def find_a_neighbour(data, current_solution, config_columns, performance_column, worst_value):

    config_columns = list(config_columns)

    random_column = random.choice(config_columns)
    index = config_columns.index(random_column)

    random_value = int(np.random.choice(data[random_column].unique()))

    current_solution[index] = random_value


    matched_row = data.loc[(data[config_columns] == pd.Series(current_solution, index=config_columns)).all(axis=1)]

    if not matched_row.empty:
        performance = matched_row[performance_column].iloc[0]
    else:
        performance = worst_value

    return current_solution, performance


def random_solution_generator(data, config_columns, performance_column, worst_value):
    sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]

    matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

    if not matched_row.empty:
        # Existing configuration
        performance = matched_row[performance_column].iloc[0]
    else:
        performance = worst_value

    return sampled_config, performance

def simulated_annealing(file_path, budget, temperature, output_file, cooling_rate):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Identify the columns for configurations and performance
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]


    # Determine if this is a maximization or minimization problem
    # maximize throughput and minimize runtime
    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True
    else:
        maximization = False

    # Extract the best and worst performance values
    if maximization:
        worst_value = data[performance_column].min() / 2  # For missing configurations
    else:
        worst_value = data[performance_column].max() * 2  # For minssing configrations

    random_solution, random_performance = random_solution_generator(data, config_columns, performance_column, worst_value)

    budget -= 1


    # Initialize the best solution and performance
    current_performance = random_performance
    current_solution = random_solution

    best_performance = current_performance
    best_solution = current_solution

    # Store all search results
    search_results = [current_solution + [current_performance]]

    for i in range(budget):

        temperature *= cooling_rate

        neighbour_solution, neighbour_performance = find_a_neighbour(data, current_solution, config_columns, performance_column, worst_value)

        delta = neighbour_performance - current_performance

        try:
            prob = math.exp(-delta / temperature)
        except OverflowError:
            prob = float('inf')

        if random.random() < prob:
            current_solution = neighbour_solution[:]
            current_performance = neighbour_performance


        if maximization:
            if neighbour_performance > best_performance:
                best_solution = neighbour_solution[:]
                best_performance = neighbour_performance
        else:
            if neighbour_performance < best_performance:
                best_solution = neighbour_solution[:]
                best_performance = neighbour_performance

        search_results.append(neighbour_solution + [neighbour_performance])

    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance


# Main function to test on multiple datasets
def main():
    datasets_folder = "datasets"
    output_folder = "search_results"
    os.makedirs(output_folder, exist_ok=True)
    budget = 100
    runs = 1

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            if runs == 1:
                file_path = os.path.join(datasets_folder, file_name)
                output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
                best_solution, best_performance = simulated_annealing(file_path, budget, 1, output_file, 0.963)
                results[file_name] = {
                    "Best Solution": best_solution,
                    "Best Performance": best_performance
                }
            else:
                file_path = os.path.join(datasets_folder, file_name)
                output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
                system_name = file_name.split('.')[0]
                best_performances = []
                for run in range(1, runs + 1):
                    best_solution, best_performance = simulated_annealing(file_path, budget, 1, output_file, 0.963)
                    best_performances.append(best_performance)
                    print(f"Run {run} for {system_name} completed. Best Performance: {best_performance}")
                results[system_name] = best_performances

    if runs == 1:
        # Print the results
        for system, result in results.items():
            print(f"System: {system}")
            print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
            print(f"  Best Performance: {result['Best Performance']}")
    else:

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_folder, "best_performances.csv")

        # Save to a single CSV file
        results_df.to_csv(results_csv_path, index_label="Run")
        print(f"CSV file with best performances over {runs} runs was returned to {results_csv_path}")
if __name__ == "__main__":
    main()
