import math
import random

import pandas as pd
import numpy as np
import os

def find_a_neighbour(data, current_solution, config_columns, performance_column, past_solutions, worst_value, loose_attempts = 100, max_attempts = 200):
    new_solution = []
    attempts = 0

    while not new_solution:
        temp = current_solution[:]
        config_columns = list(config_columns)

        if attempts < loose_attempts:
            random_column = random.choice(config_columns)
            index = config_columns.index(random_column)

            random_value = int(np.random.choice(data[random_column].unique()))

            temp[index] = random_value

        else:
            random_column_one, random_column_two = random.sample(config_columns, k=2)
            one_index = config_columns.index(random_column_one)
            two_index = config_columns.index(random_column_two)

            one_random_value = int(np.random.choice(data[random_column_one].unique()))
            two_random_value = int(np.random.choice(data[random_column_two].unique()))

            temp[one_index] = one_random_value
            temp[two_index] = two_random_value

        matched_row = data.loc[(data[config_columns] == pd.Series(temp, index=config_columns)).all(axis=1)]

        if not temp in past_solutions:
            if not matched_row.empty:
                performance = matched_row[performance_column].iloc[0]

            else:
                performance = worst_value

            new_solution = temp[:]

        else:
            attempts += 1

        if attempts == max_attempts:
            matched_row = data.loc[(data[config_columns] == pd.Series(current_solution, index=config_columns)).all(axis=1)]

            performance = matched_row[performance_column].iloc[0]
            new_solution = current_solution[:]



    return new_solution, performance


def random_solution_generator(data, config_columns, performance_column, worst_value):
    sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]

    matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

    if not matched_row.empty:
            # Existing configuration
        performance = matched_row[performance_column].iloc[0]
    else:
        performance = worst_value



    return sampled_config, performance

def improved_simulated_annealing(file_path, budget, temperature, output_file, cooling_rate, initialisation_ratio):
    # Load the dataset
    data = pd.read_csv(file_path)
    past_checks = []

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

    random_budget = int(budget * initialisation_ratio)
    budget = budget - random_budget
    initial_solutions = []
    initial_performances = []
    search_results = []

    for i in range(random_budget):
        random_solution, random_performance = random_solution_generator(data, config_columns, performance_column, worst_value)
        initial_solutions.append(random_solution)
        initial_performances.append(random_performance)
        past_checks.append(random_solution)
        search_results.append(random_solution + [random_performance])


    if maximization:
        initial_index = initial_performances.index(max(initial_performances))

        initial_solution = initial_solutions[initial_index]
        initial_performance = initial_performances[initial_index]
    else:
        initial_index = initial_performances.index(min(initial_performances))

        initial_solution = initial_solutions[initial_index]
        initial_performance = initial_performances[initial_index]



    # Initialize the best solution and performance
    current_performance = initial_performance
    current_solution = initial_solution

    best_performance = current_performance
    best_solution = current_solution


    # Store all search results
    search_results.append(current_solution + [current_performance])

    for i in range(budget):

        temperature *= cooling_rate

        neighbour_solution, neighbour_performance = find_a_neighbour(data, current_solution, config_columns, performance_column, past_checks, worst_value)
        past_checks.append(neighbour_solution)

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
    temperature = 1
    cooling_rate = 0.963
    initialisation_ratio = 0.3

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            if runs == 1:
                file_path = os.path.join(datasets_folder, file_name)
                output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
                best_solution, best_performance = improved_simulated_annealing(file_path, budget, temperature, output_file, cooling_rate, initialisation_ratio)
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
                    best_solution, best_performance = improved_simulated_annealing(file_path, budget, temperature, output_file, cooling_rate, initialisation_ratio)
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
