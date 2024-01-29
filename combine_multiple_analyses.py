"""
This file was created to combine multiple runs of the analyses into a single result
"""

import os
import sys

import pandas as pd

folder_path = sys.argv[1]
batches = [directory for directory in os.listdir(folder_path) if 'repetition' in directory]
models = ['FastML', 'GaussianCopula', 'CTGAN', 'DP-CGAN']
experiments = ['n_input', 'n_output']

# initialize an empty DataFrame to store the combined data
average_results = pd.DataFrame()

for model in models:
    # ensure a path for the average results of model
    if os.path.exists(os.path.join(folder_path, model)) is False:
        os.mkdir(os.path.join(folder_path, model))

    for experiment in experiments:
        # ensure that within the average result directory there is a directory for the analysis
        if os.path.exists(os.path.join(folder_path, model, experiment)) is False:
            os.mkdir(os.path.join(folder_path, model, experiment))

        # use the first batch to retrieve all csv files
        _dummy_path = os.path.join(folder_path, batches[0], model, 'evaluation', experiment)
        # retrieve all csv files for a given experiment and model; assuming all folders contain similar files
        csv_files = [f for f in os.listdir(_dummy_path) if f.endswith('.csv')]

        # loop through every csv filename and average them out over the included batches
        for csv_file in csv_files:
            datasets = []
            for batch in batches:
                file_path = os.path.join(folder_path, batch, model, 'evaluation', experiment, csv_file)
                datasets.append(pd.read_csv(file_path))

            stacked_csv_files = pd.concat(datasets)

            average_result = stacked_csv_files.groupby(level=0).mean()

            # save the result in the appropriate folder
            average_result.to_csv(os.path.join(folder_path, model, experiment, csv_file))
