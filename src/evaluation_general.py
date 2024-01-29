import copy
import logging
import os.path
import time
import warnings

import numpy as np
import pandas as pd

# private modules
from src import evaluation_visualisation, data_processing_general, evaluation_metrics, file_handling, generator_sdv

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataEvaluation:
    """
    Aims to evaluate a dictionary of synthetic data with the specified original data
    """

    def __init__(self, post_hoc=False, sample_to_generate=None, directory=None):
        """
        Initialises the DataEvaluation class by finding and loading the original dataset and its metadata

        :param bool post_hoc: specify whether to generate data or analyse existing data i.e., post-hoc analysis
        :param int sample_to_generate: specify the number of samples to generate if not post-hoc
        :param str directory: allows to manually specify the directory that will be searched
        """
        self.PostHoc = post_hoc

        self.BaseNumberOfSamples = sample_to_generate
        # ensure that there is a certain base number to generate when necessary
        if isinstance(self.BaseNumberOfSamples, int) is False:
            self.BaseNumberOfSamples = 1000

        self.Directory = directory
        self.OriginalDataPath = None
        self.MetaDataPath = None

        self.OriginalData = None
        self.MetaData = None
        self.SyntheticData = {}

        # dictionaries for evaluation output
        self.EvaluationSettings = {}
        self.EvaluationOutput = {}
        self.Timer = {}
        self.OriginalDataReproduced = {}
        self.GeneratorModels = []

        # variables to evaluate
        self.VariableValues = []

        if self.PostHoc:
            # retrieve existing data
            self._initialise_post_hoc()
        else:
            self.DataGenerator = None

            # initiate the data generator
            self._initiate_generator()

        # ensure path is present
        if os.path.exists(os.path.join(self.Directory, 'evaluation')) is False:
            os.mkdir(os.path.join(self.Directory, 'evaluation'))
        self.OutputPath = os.path.join(self.Directory, 'evaluation')

        if os.path.exists(os.path.join(self.OutputPath, 'dump')) is False:
            os.mkdir(os.path.join(self.OutputPath, 'dump'))
        self.AnalysesDumpPath = os.path.join(self.OutputPath, 'dump')

        # noinspection PyUnresolvedReferences
        self.OriginalName = self.OriginalDataPath[self.OriginalDataPath.rfind(os.path.sep):
                                                  self.OriginalDataPath.find('.csv')]

    def _initialise_post_hoc(self):
        """
        Initialises post-hoc analysis by retrieving the path and data, proceeds to retrieve existing synthetic data
        """
        logging.info(f'Initialising post-hoc analysis')
        # retrieve data, model and the directory
        self.OriginalDataPath, self.MetaDataPath, self.Directory, self.SettingsPath \
            = file_handling.find_items(data_directory=self.Directory)

        # metadata is required to determine what variables are categorical and boolean to determine hamming distance
        self.OriginalData = file_handling.read_csv(self.OriginalDataPath, 'float', ' ', ',')
        self.MetaData = file_handling.read_json(self.MetaDataPath)
        self.Settings = file_handling.read_json(self.SettingsPath)

        # retrieve (synthetic) data files
        self.data_load_files()

        # process data
        self.OriginalData = data_processing_general.impute_mice(self.OriginalData)

    def _initiate_generator(self, deep_learning_models=None, machine_learning_models=None):
        """
        Initialises an SDV generator object and generates data using the specified sample size and model

        :param numpy.ndarray deep_learning_models: array of the deep learning models that are available
        :param numpy.ndarray machine_learning_models: array of the machine learning models that are available
        """
        # defined separately as the generator module does not discriminate on deep or machine learning
        if deep_learning_models is None:
            deep_learning_models = np.array(['CTGAN', 'CopulaGAN', 'TVAE', 'DP-CGAN'])

        # CopulaGAN has both deep as machine learning parameters
        if machine_learning_models is None:
            machine_learning_models = np.array(['GaussianCopula', 'CopulaGAN'])

        logging.info(f'Initialising data generator')
        # start with the entire dataset and initialisation
        self.DataGenerator = generator_sdv.SDVSynthetic(directory=self.Directory)

        # retrieve paths and data via initiated generator
        self.OriginalDataPath = self.DataGenerator.OriginalDataPath
        self.MetaDataPath = self.DataGenerator.MetaDataPath
        self.Directory = self.DataGenerator.Directory
        self.OriginalData = self.DataGenerator.OriginalDataImputed
        self.MetaData = self.DataGenerator.MetaData
        self.Settings = self.DataGenerator.Settings

        # generative models are selected and retrieved on a function level

        # set available model names per type
        self.DeepLearningModels = deep_learning_models
        self.MachineLearningModels = machine_learning_models

    def data_load_files(self, data_identifier='^.csv', analysis_name=None):
        """
        Load existing synthetic data files i.e., saved in the specified directory, to the evaluation class

        :param str data_identifier: specify the substring that is to be used for data file searchin
        :param str analysis_name: specify a name for the analysis you would like to perform
        """
        if isinstance(analysis_name, str) is False:
            analysis_name = str(input(f'Specify a name for your analysis.\nNote that: this name will be used as column '
                                      f'heading and identification key.')).strip().replace("'", "")

        # logging is present in data reading functions

        # allow selection of multiple synthetic data sets
        continue_loading = True
        post_hoc = 'no'

        while continue_loading is True:
            # allow to add existing data to generated data
            if self.PostHoc is False:
                warnings.warn(
                    f'Data has already been loaded in the generator module as post-hoc analysis was not specified.\n')
                post_hoc = input(f'Would you still like to add existing data to the generated data? Enter yes or no.')
                if 'yes' in post_hoc:
                    self.PostHoc = True
                    break

            synthetic_path = file_handling.find_file(os.path.join(self.Directory, 'synthetic'), data_identifier)
            synthetic_name = synthetic_path[synthetic_path.rfind(os.path.sep): synthetic_path.find('.')]

            # check whether a duplicate is desired and prevent key errors when there is no data in the dictionary yet
            if analysis_name in self.SyntheticData.keys():
                if synthetic_name in self.SyntheticData[analysis_name].keys():
                    duplicate = input(f'You have already selected this dataset, would you want to load it once more? '
                                      f'Enter yes or no\n')
                    if duplicate in 'no':
                        # determine whether to load more datasets
                        continue_loading = input(f'Would you like to load another dataset? Enter yes or no\n')
                        if continue_loading in 'yes':
                            continue_loading = True
                        elif continue_loading in 'no':
                            continue_loading = False
                        continue

            # load data
            synthetic_dataset = file_handling.read_csv(synthetic_path, 'float64')

            # place data in dictionary
            if analysis_name in self.SyntheticData.keys():
                self.SyntheticData[analysis_name].update({synthetic_name: synthetic_dataset})
            else:
                self.SyntheticData.update({analysis_name: {synthetic_name: synthetic_dataset}})

            # determine whether to load more datasets
            continue_loading = input(f'Would you like to load another dataset? Enter yes or no\n')
            if continue_loading in 'yes':
                continue_loading = True
            elif continue_loading in 'no':
                continue_loading = False

        if post_hoc in 'yes':
            self._initialise_post_hoc()

    def evaluate_an_analysis(self, analysis_name, statistical_similarity=True,
                             identity_disclosure=True, attribute_disclosure=True,
                             replication=True, timing=True, line_width=None,
                             save_to_csv=True, save_to_plot=True, impute=False):
        """
        Perform the specified evaluations on the given analysis e.g., determine the statistical similarity for an
        analysis of different input sample sizes.

        :param str analysis_name: specify the name of the analysis, i.e.,
        a name associated with a built-in evaluation such as input sample size or the post-hoc specified name.
        This name will also serve as column heading.
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param bool timing: specify whether to collect data from the timing variable and add it to the dataframe.
        Only possible when synthetic data generation has been performed i.e., not possible post-hoc
        :param int/float line_width: specify the line width in the plot when opting to plot the box and line plot;
        main purpose is to mute the lines when datapoints should not be connected
        :param bool save_to_csv: specify whether to save the evaluation output to a .csv file
        :param bool save_to_plot: specify whether to plot the data in a graph
        :param bool impute: specify whether to impute missing values in the data
        :return: DataFrame of the evaluation output
        """
        logging.info(f'Evaluating analysis {analysis_name}')
        _attribute_disclosure = None
        _identity_disclosure = None
        _parameter_values = []
        _statistical_similarity = None
        _reproducibility = None

        attribute_known = None
        attribute_to_guess = None
        effect_to_plot = None
        output_values = {}
        parameter_values = None
        rename_evaluated_dataset = None

        # retrieve certain settings
        if isinstance(self.Settings, dict):
            self.EvaluationSettings = self.Settings['Evaluation']
            if 'variables_to_plot' in self.EvaluationSettings:
                if isinstance(self.EvaluationSettings['variables_to_plot'], list):
                    effect_to_plot = self.EvaluationSettings['variables_to_plot']
            if 'parameter_values' in self.EvaluationSettings:
                if isinstance(self.EvaluationSettings['parameter_values'], list):
                    parameter_values = self.EvaluationSettings['parameter_values']
            if 'known_variables' in self.EvaluationSettings:
                if isinstance(self.EvaluationSettings['known_variables'], list):
                    attribute_known = self.EvaluationSettings['known_variables']
            if 'sensitive_variables' in self.EvaluationSettings:
                if isinstance(self.EvaluationSettings['sensitive_variables'], list):
                    attribute_to_guess = self.EvaluationSettings['sensitive_variables']

        # impute missing data when desired
        if impute:
            self.OriginalData = data_processing_general.impute_mice(self.OriginalData)

            # loop through all loaded synthetic data to already impute missing values
            for synthetic_key in self.SyntheticData[analysis_name].keys():
                self.SyntheticData[analysis_name][synthetic_key] = \
                    data_processing_general.impute_mice(self.SyntheticData[analysis_name][synthetic_key])

        # allow specification of variable values via function
        if parameter_values is None or isinstance(parameter_values, (list, set, tuple)) is False:
            logging.debug(f'Attempting to fetch the values of the performed evaluation')
            if self.VariableValues:
                _parameter_values = self.VariableValues
            else:
                # attempt to generate appropriate entries for the analysed variable
                for identification in self.SyntheticData[analysis_name].keys():
                    try:
                        value = float(identification[identification.find(analysis_name): identification.rfind('(')])
                        _parameter_values.append(value)
                    except ValueError:
                        try:
                            # in case a file id was not used a period should provide the correct value
                            value = float(
                                identification[identification.find(f'{analysis_name}_'): identification.rfind('.')])
                            _parameter_values.append(value)
                        except ValueError:
                            logging.debug(f'No identifiable value found in dataset identification')
                            if rename_evaluated_dataset is None:
                                rename_evaluated_dataset = input(
                                    f'Current dataset names are similar to {identification}.\n'
                                    f'Would you like to alter the name that is used in the plot to e.g., '
                                    f'a changed variable such as the input sample size?\n'
                                    f'Enter yes or no.')
                            if rename_evaluated_dataset in 'yes':
                                _parameter_values.append(
                                    input(f'For dataset with given name: {identification}.\n'
                                          f'Please provide the name you would like to give this dataset:'))
                            if rename_evaluated_dataset in 'no':
                                # copy the full name and break to stop renaming
                                _parameter_values = np.copy(self.SyntheticData[analysis_name].keys())
                                break
        else:
            _parameter_values = parameter_values

        # perform desired evaluations
        if statistical_similarity:
            _statistical_similarity = self.evaluate_statistical_similarity(analysis_name)
            # update the dict that contains the evaluation output
            output_values.update(_statistical_similarity)

        if identity_disclosure:
            _identity_disclosure = self.evaluate_privacy_identity(analysis_name)
            # update the dict that contains the evaluation output and prevent replacing existing ones
            if bool(output_values) is True:
                for sample in output_values.keys():
                    output_values[sample].update(_identity_disclosure[sample])
            else:
                output_values.update(_identity_disclosure)

        if attribute_disclosure:
            if attribute_to_guess is None or attribute_known is None:
                warnings.warn(f'Attribute disclosure cannot be determined without providing '
                              f'known and sensitive variables in the evaluation configuration JSON file.')
            else:
                _attribute_disclosure = self.evaluate_privacy_attribute(analysis_name,
                                                                        attribute_known, attribute_to_guess)
                # update the dict that contains the evaluation output and prevent replacing existing ones
                if bool(output_values) is True:
                    for sample in output_values.keys():
                        output_values[sample].update(_attribute_disclosure[sample])
                else:
                    output_values.update(_attribute_disclosure)

        if replication:
            _reproducibility = self.evaluate_utility(analysis_name)
            # update the dict that contains the evaluation output and prevent replacing existing ones
            if bool(output_values) is True:
                for sample in output_values.keys():
                    output_values[sample].update(_reproducibility[sample])
            else:
                output_values.update(_reproducibility)
            pass

        if timing:
            if analysis_name in self.SyntheticData.keys():
                # update the dict that contains the evaluation output and prevent replacing existing ones
                if bool(output_values) is True:
                    for sample in output_values.keys():
                        output_values[sample].update(self.Timer[analysis_name][sample])
                else:
                    output_values.update(_reproducibility)
            else:
                warnings.warn(f'Analysis {analysis_name} was not timed and therefore timing cannot be retrieved')
                pass

        # create a base that consists of file identifications and the chosen values
        output_frame = pd.DataFrame({'file_identification': list(self.SyntheticData[analysis_name].keys()),
                                     analysis_name: _parameter_values})

        # add the necessary columns to the dataset and avoid column with file identification as name
        output_frame = output_frame.reindex(columns=output_frame.columns.tolist() +
                                                    list(output_values[list(output_values)[0]].keys()))

        # iterate over the existing rows of the DataFrame so that their values can be retrieved
        for evaluated_row in output_frame.itertuples():
            logging.debug(f'Gathering output values for {evaluated_row.file_identification}')
            # select the variables per sample to ensure that it is always specific to that dataset
            for variable in output_values[evaluated_row.file_identification].keys():
                # update the specific column with the value corresponding to the specific file identification
                output_frame.at[evaluated_row.Index, variable] = \
                    output_values[evaluated_row.file_identification][variable]

        # compose filename and save file
        original_name = self.OriginalDataPath[self.OriginalDataPath.rfind(os.path.sep):
                                              self.OriginalDataPath.find('.csv')]

        self.EvaluationOutput = output_frame

        if save_to_csv:
            file_handling.save_csv(output_frame, f'{self.OutputPath}{original_name}_{analysis_name}.csv')

        if save_to_plot:
            evaluation_visualisation.visualise_line_and_box(output_frame, analysis_name, y_line_width=line_width,
                                                            filename=f'{self.OutputPath}{original_name}_'
                                                                     f'{analysis_name}'
                                                                     f'_data_quality'
                                                                     f'{self.EvaluationSettings["graph_file_extension"]}')
            if replication:
                for variable in effect_to_plot:
                    if isinstance(variable, str):
                        evaluation_visualisation.visualise_effect_single_measure(output_frame, analysis_name, variable,
                                                                                 settings=self.EvaluationSettings,
                                                                                 reference_data=self.OriginalDataReproduced,
                                                                                 filename=f'{self.OutputPath}{original_name}_'
                                                                                          f'{analysis_name}'
                                                                                          f'_effect_measure_{variable}'
                                                                                          f'{self.EvaluationSettings["graph_file_extension"]}')

    def evaluate_statistical_similarity(self, analysis_name, timer=True, save_timer=False):
        """
        Evaluate the statistical similarity of every loaded synthetic dataset with the original dataset

        :param str analysis_name: specify the name of the analysis you would like to evaluate statistical similarity for
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :return: dictionary containing statistical similarity metrics per loaded synthetic dataset
        """
        logging.info(f'Evaluating statistical similarity')
        statistical_similarity = {}

        # evaluate every synthetic dataset
        for synthetic_data in self.SyntheticData[analysis_name].keys():
            logging.debug(f'Evaluating statistical similarity of {synthetic_data}')

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, synthetic_data, 1)

            statistical_similarity.update({synthetic_data: evaluation_metrics.compute_stat_similarity(
                self.OriginalData, self.SyntheticData[analysis_name][synthetic_data])})

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, synthetic_data, 1)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        return statistical_similarity

    def evaluate_privacy_identity(self, analysis_name, original_key='original', synthetic_key='synthetic',
                                  timer=True, save_timer=False, save_distances=False):
        """
        Evaluate the identity disclosure by separating the original and synthetic data in
        categorical/boolean and continuous subsets to then pass these to the identity disclosure function.
        Will compare all datasets loaded in the synthetic data dictionary to the original dataset

        :param str analysis_name: specify a name of the analysis you would like to evaluate identity disclosure for
        :param any original_key: specify the key value that will be used to denote the original data
        :param any synthetic_key: specify the key value that will be used to denote the synthetic data
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_distances: specify whether to save the distance arrays, can consume substantial disk space
        :return: dictionary containing the identity disclosure stats for every dataset in the synthetic data dictionary
        """
        logging.info(f'Evaluating identity disclosure.')
        identity_disclosure = {}
        variables_categorical = []
        categorical_data = None
        variables_continuous = []
        continuous_data = None

        # retrieve column types
        for variable_name in self.MetaData['fields']:

            # find categorical and boolean variable names
            if self.MetaData['fields'][variable_name]['type'] in 'categorical' or \
                    self.MetaData['fields'][variable_name]['type'] in 'bool':
                variables_categorical.append(variable_name)

            # find continuous variable names
            if self.MetaData['fields'][variable_name]['type'] in 'numerical':
                variables_continuous.append(variable_name)

        # retrieve data of specific type by variable name if they are present; if statement avoids using pandas index's
        if variables_categorical:
            # convert categorical data to boolean for hamming distance
            original_data_dummified, metadata, constraints = \
                data_processing_general.dummify(self.OriginalData[variables_categorical], self.MetaData)
            categorical_data = {original_key: original_data_dummified}

        if variables_continuous:
            continuous_data = {original_key: self.OriginalData[variables_continuous]}

        # evaluate every synthetic dataset
        for synthetic_data in self.SyntheticData[analysis_name].keys():
            logging.debug(f'Evaluating identity disclosure of {synthetic_data}')

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, synthetic_data, 2)

            if variables_categorical:
                # dummify the data for hamming distance
                synthetic_data_dummified, metadata, constraints = \
                    data_processing_general.dummify(
                        self.SyntheticData[analysis_name][synthetic_data][variables_categorical],
                        self.MetaData)

                # compensate column mismatch due to dummification of categories that are not present in synthetic sample
                # noinspection PyUnboundLocalVariable
                synthetic_data_dummified = synthetic_data_dummified.reindex(
                    columns=original_data_dummified.columns.tolist())

                # retrieve data of the specific type by variable name of the selected synthetic dataset
                categorical_data.update(
                    {synthetic_key: synthetic_data_dummified})
            if variables_continuous:
                continuous_data.update(
                    {synthetic_key: self.SyntheticData[analysis_name][synthetic_data][variables_continuous]})

            # compute statistics
            statistics, categorical_distances, continuous_distances = evaluation_metrics.compute_privacy_identity(
                original_key, synthetic_key, categorical_data, continuous_data)

            if save_distances:
                # noinspection DuplicatedCode
                if categorical_distances is not None:
                    # store in separate folder for cleanliness
                    if os.path.exists(os.path.join(self.AnalysesDumpPath, analysis_name)) is False:
                        os.mkdir(os.path.join(self.AnalysesDumpPath, analysis_name))
                    _analyses_dump_path = os.path.join(self.AnalysesDumpPath, analysis_name)

                    np.save(f"{_analyses_dump_path}"
                            f"{synthetic_data[synthetic_data.rfind(os.path.sep):synthetic_data.rfind('.csv')]}"
                            f"^_categorical_distance_array", categorical_distances)

                # noinspection DuplicatedCode
                if continuous_distances is not None:
                    # store in separate folder for cleanliness
                    if os.path.exists(os.path.join(self.AnalysesDumpPath, analysis_name)) is False:
                        os.mkdir(os.path.join(self.AnalysesDumpPath, analysis_name))
                    _analyses_dump_path = os.path.join(self.AnalysesDumpPath, analysis_name)

                    np.save(f"{_analyses_dump_path}"
                            f"{synthetic_data[synthetic_data.rfind(os.path.sep):synthetic_data.rfind('.csv')]}"
                            f"^_continuous_distance_array", continuous_distances)

            # store score
            identity_disclosure.update({synthetic_data: statistics})

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, synthetic_data, 2)

            # save the timer when desired
            if save_timer:
                if timer:
                    file_handling.save_json(self.Timer[analysis_name],
                                            f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
                else:
                    warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        return identity_disclosure

    def evaluate_privacy_attribute(self, analysis_name, known_variables, sensitive_variables, number_of_bins=None,
                                   timer=True, save_timer=False):
        """
        Evaluate the attribute disclosure by guessing all combinations of sensitive and known variables specified
        in the analysis JSON file

        :param str analysis_name: specify a name of the analysis you would like to evaluate identity disclosure for
        :param list known_variables: the variables names known to the guessing party
        :param list sensitive_variables: the variables names that are to be guessed
        :param int number_of_bins: define the number of bins that non-categorical values will be put in
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :return: dictionary containing the attribute disclosure stats for every dataset in the synthetic data dictionary
        """
        logging.info(f'Evaluating attribute disclosure.')
        attribute_disclosure = {}
        variables_to_bin = []

        # create a copy to work on
        _original_data = self.OriginalData.copy()

        if isinstance(number_of_bins, int) and number_of_bins != 0:
            # generate a list of variables that are to be binned as result of not being categorical
            variables_to_bin = [variable for variable in known_variables + sensitive_variables
                                if self.MetaData['fields'][variable]['type'] != 'categorical' and
                                self.MetaData['fields'][variable]['type'] != 'boolean']

            # bin the non-categorical variables
            for variable in variables_to_bin:
                _original_data[variable] = pd.qcut(_original_data[variable].rank(method='first'), q=number_of_bins)

        # evaluate every synthetic dataset
        for synthetic_data in self.SyntheticData[analysis_name].keys():
            logging.debug(f'Evaluating statistical similarity of {synthetic_data}')

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, synthetic_data, 1)

            # create a copy to work on
            _synthetic_data = self.SyntheticData[analysis_name][synthetic_data].copy()

            if isinstance(number_of_bins, int) and number_of_bins != 0:
                # bin the non-categorical variables
                for variable in variables_to_bin:
                    _synthetic_data[variable] = pd.qcut(_synthetic_data[variable].rank(method='first'),
                                                        q=number_of_bins)

            attribute_disclosure.update({synthetic_data: evaluation_metrics.compute_privacy_attribute(
                _original_data, _synthetic_data, known_variables, sensitive_variables)})

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, synthetic_data, 1)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        return attribute_disclosure

    def evaluate_utility(self, analysis_name, univariate=True, timer=True, save_timer=False):
        """
        Evaluate the data quality by reproducing an analysis performed on the original data

        :param str analysis_name: specify a name of the analysis you would like to evaluate utility for
        :param bool univariate: specify whether to also perform univariate analyses
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :return: dictionary containing the odds ratio and confidence intervals for every dataset in
        the synthetic data dictionary
        """
        reproducibility = {}

        if 'formula_logistic_regression' in self.EvaluationSettings:
            # retrieve the formula to replicate
            model_formula = self.EvaluationSettings['formula_logistic_regression']

            logging.info(f'Replicating logistic regression with formula\n{model_formula}')

            # perform logistic regression on the original data so that all columns are established and for comparison
            self.OriginalDataReproduced = evaluation_metrics.logistic_regression(self.OriginalData, model_formula,
                                                                                 validation_data=self.OriginalData)

            # evaluate every synthetic dataset
            for synthetic_data in self.SyntheticData[analysis_name].keys():
                logging.debug(f'Reproducing logistic regression for {synthetic_data}')

                if timer:
                    # call timer to store start time
                    self.evaluate_time(analysis_name, synthetic_data, 3)

                reproducibility.update({synthetic_data: evaluation_metrics.logistic_regression(
                    self.SyntheticData[analysis_name][synthetic_data], r_formula=model_formula,
                    original_data_outcome=self.OriginalDataReproduced, validation_data=self.OriginalData,
                    univariate=univariate)})

                if timer:
                    # call timer to store difference
                    self.evaluate_time(analysis_name, synthetic_data, 3)

            # save the timer when desired
            if save_timer:
                if timer:
                    file_handling.save_json(self.Timer[analysis_name],
                                            f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
                else:
                    warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        else:
            logging.warning(f'Utility evaluation not supported, returning an empty dictionary.')

        return reproducibility

    def evaluate_time(self, analysis_name, evaluation_name, evaluation_type, new_evaluation_name=None):
        """
        Records the time for specified evaluation, is to be called twice (i.e., at time 1 and time 2).
        Stores the time in class's Timer dictionary. Both analysis and evaluation name are converted to string

        :param any analysis_name: specify the name of the analysis
        :param any evaluation_name: name of the evaluation in need for time evaluation
        :param int evaluation_type: specify the type of evaluation that is performed: 0 data generation;
        1 evaluation statistical similarity; 2 evaluation identity disclosure; 3 evaluation reproducibility
        :param any new_evaluation_name: specify a name that is to replace the evaluation name
        """
        # convert to string
        analysis_name = str(analysis_name)
        evaluation_name = str(evaluation_name)

        if evaluation_type == 0:
            _evaluation_type = 'time generation'
        elif evaluation_type == 1:
            _evaluation_type = 'time evaluation statistical similarity'
        elif evaluation_type == 2:
            _evaluation_type = 'time evaluation identity disclosure'
        elif evaluation_type == 3:
            _evaluation_type = 'time evaluation reproducibility'
        else:
            _evaluation_type = 'time'

        # ensure that the analysis is accounted for in the Timer dictionary
        if analysis_name not in self.Timer.keys():
            self.Timer.update({analysis_name: {}})

        # ensure that the evaluation is accounted for in the Timer dictionary
        if evaluation_name not in self.Timer[analysis_name].keys():
            self.Timer[analysis_name].update({evaluation_name: {}})

        try:
            # if a timer has been started compute the time difference
            if isinstance(self.Timer[analysis_name][evaluation_name][_evaluation_type], (int, float)) is True:
                # overwrite the start time with time passed
                self.Timer[analysis_name][evaluation_name][_evaluation_type] = \
                    time.perf_counter() - self.Timer[analysis_name][evaluation_name][_evaluation_type]
            if new_evaluation_name is not None:
                # replace the name with a filename if desired
                self.Timer[analysis_name][new_evaluation_name] = self.Timer[analysis_name][evaluation_name]
                del self.Timer[analysis_name][evaluation_name]

        except KeyError:
            # start a timer if none has been set
            self.Timer[analysis_name][evaluation_name].update({_evaluation_type: time.perf_counter()})

    def generator_evaluate_n_input_random(self, stop=600, step=250, output_start=None, output_stop=None,
                                          output_step=None,
                                          timer=True, save_timer=False, save_models=False, save_data=False,
                                          statistical_similarity=True,
                                          identity_disclosure=True, attribute_disclosure=True,
                                          replication=True,
                                          models_to_evaluate=None, default_model=False, analysis_name='n_input'):
        """
        Generate synthetic data that is using a synthetic data generation model that is receiving less and
        less input (or training) data. More precisely, the function randomly resamples the original dataset,
        initiates re-training and generates a sample from this new model.
        This process continues until the specified limit is reached.

        :param int stop: specify the lowest number of input samples (i.e., the lower boundary to step towards)
        :param int step: specify the number of samples to decrease the input sample size with
        :param int output_start: specify the lowest number of output samples (i.e., the lower boundary to start stepping from)
        :param int output_stop: specify the highest number of output samples (i.e., the upper boundary to step towards)
        :param int output_step: specify the number of samples to increase the start sample with
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the input sample size by random sub-sampling')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of the input sample size by random sub-sampling')

        # make sure the loop breaks if the generator still is not present
        if self.DataGenerator is None:
            return

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()
            # store a copy of the loaded models to ensure that the base models are available
            self.GeneratorModels = self.DataGenerator.LoadedModels

        # set up the base input sample size
        n_input = len(self.OriginalData)
        if timer:
            # call timer to store start time
            self.evaluate_time(analysis_name, n_input, 0)

        # model data
        # noinspection DuplicatedCode
        self.DataGenerator.model_data(save=save_models, name_appendices=f'_n_{n_input}')

        # retrieve synthetic data
        self.generator_evaluate_n_output(start=output_start, stop=output_stop, step=output_step, save_data=save_data,
                                         analysis_name=f'{analysis_name}_{n_input}_n_output',
                                         called_by=True,
                                         statistical_similarity=False, identity_disclosure=False,
                                         attribute_disclosure=False, replication=False, default_model=default_model)

        if timer:
            # call timer to store difference
            self.evaluate_time(analysis_name, n_input, 0)

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}_{n_input}_n_output',
                                  statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure,
                                  attribute_disclosure=attribute_disclosure,
                                  replication=replication)

        self.DataGenerator.SyntheticData = {}
        del self.SyntheticData[f'{analysis_name}_{n_input}_n_output']
        self.VariableValues = []
        self.EvaluationOutput = {}

        # determine samples to generate
        samples_to_generate = np.arange(start=stop, stop=len(self.OriginalData), step=step)
        # flip to change the order of samples generated (i.e., ascending or descending order)
        samples_to_generate = np.flip(samples_to_generate)

        for n_input in samples_to_generate:
            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, n_input, 0)

            # generate a subset and place in class's original data
            logging.debug(f'Adjusting input sample size to {n_input}')

            original_data_copy = copy.deepcopy(self.OriginalData)
            self.OriginalData = self.OriginalData.sample(n=n_input, random_state=n_input)
            self.DataGenerator.OriginalData = self.OriginalData

            if save_data:
                file_handling.save_csv(self.DataGenerator.OriginalData,
                                       f"{self.OriginalDataPath[:self.OriginalDataPath.rfind('.csv')]}"
                                       f"_subsample_{n_input}.csv")
            # remodel data
            # noinspection DuplicatedCode
            self.DataGenerator.model_data(save=save_models, name_appendices=f'_n_{n_input}')

            # retrieve synthetic data
            self.generator_evaluate_n_output(start=output_start, stop=output_stop, step=output_step,
                                             save_data=save_data, analysis_name=f'{analysis_name}_{n_input}_n_output',
                                             called_by=True,
                                             statistical_similarity=False, identity_disclosure=False,
                                             attribute_disclosure=False, replication=False, default_model=default_model)

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, n_input, 0)

            # perform evaluation
            self.evaluate_an_analysis(f'{analysis_name}_{n_input}_n_output',
                                      statistical_similarity=statistical_similarity,
                                      identity_disclosure=identity_disclosure,
                                      attribute_disclosure=attribute_disclosure,
                                      replication=replication)

            # return original data to original state and remove synthetic sets to prevent memory stress
            self.OriginalData = copy.deepcopy(original_data_copy)
            self.DataGenerator.SyntheticData = {}
            del self.SyntheticData[f'{analysis_name}_{n_input}_n_output']
            self.VariableValues = []
            self.EvaluationOutput = {}

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

    def generator_evaluate_n_output(self, start=100, stop=40000, step=1000, called_by=False,
                                    timer=True, save_timer=False, save_data=False, save_model=False,
                                    statistical_similarity=True,
                                    identity_disclosure=True, attribute_disclosure=True,
                                    replication=True,
                                    models_to_evaluate=None, default_model=False, analysis_name='n_output'):
        """
        Generates synthetic data using the base model with increasing output sample sizes.
        More precisely, a model trained on the complete dataset will be sampled using the initial sample generation size
        plus the specified proportion until the specified limit is reached.

        :param int start: specify the lowest number of output samples (i.e., the lower boundary to start stepping from)
        :param int stop: specify the highest number of output samples (i.e., the upper boundary to step towards)
        :param int step: specify the number of samples to increase the start sample with
        :param bool called_by: specify whether the function was called by another function
        relevant when calling this function within another evaluation, e.g., n_input evaluation
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool save_model: specify whether to save the created generator model when applicable
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the output sample size')

        if called_by is False:
            # ensure that a generator is present
            if self.DataGenerator is None:
                self._generator_ensure_presence('Evaluation of the output sample size')

            if self.DataGenerator is None:
                return

            # specify whether to use SDV's defaults or those set as default in script
            if isinstance(default_model, bool):
                self.DataGenerator.SDVDefaults = default_model

            # ensure models are selected if not specified correctly
            if isinstance(models_to_evaluate, list):
                self.DataGenerator.model_selection(models_to_evaluate)
            else:
                self.DataGenerator.data_find_model()

            # ensure the model can be sampled
            try:
                self.DataGenerator.model_sample(1, save=False, return_name=False)
            except AttributeError:
                logging.info(f'Data was not modelled using selected model yet, modelling data')
                self.DataGenerator.model_data(save=save_model, name_appendices=f'{analysis_name}')

            # store a copy of the loaded models to ensure that the base models are available
            self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # determine samples to generate
        samples_to_generate = np.arange(start=start, stop=stop, step=step)

        for n_output in samples_to_generate:
            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, n_output, 0)

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(n_output, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{n_output}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, n_output, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(n_output)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        # store in dictionary so that it can be evaluated sequentially
        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        if called_by is False:
            # perform evaluation
            self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                      identity_disclosure=identity_disclosure,
                                      attribute_disclosure=attribute_disclosure,
                                      replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_epochs(self, start=100, end=2000, step=100,
                                  timer=True, save_timer=False, save_models=False, save_data=False,
                                  statistical_similarity=True,
                                  identity_disclosure=True, attribute_disclosure=True,
                                  replication=True,
                                  models_to_evaluate=None, default_model=True, analysis_name='n_epochs'):
        """
        Generate synthetic data that is using a synthetic data generation model that is trained with a
        varying number of epochs. More precisely, the function changes the number of epochs
        initiates re-training and generates a sample from this new model.
        This process continues until the specified end is reached.

        :param int start: smallest value for number of epochs
        :param int end: largest value for number of epochs
        :param int step: specify the step size to change the number of epochs with
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the number of epochs')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of model parameter epochs')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # epochs can only be evaluated for deep learning models
        # noinspection PyUnresolvedReferences
        model_names = [name for name in self.DataGenerator.LoadedModels.keys() if name in self.DeepLearningModels]

        # remove models that are not te be evaluated to prevent unnecessary operations
        self._generator_remove_non_evaluated_models(model_names)

        for n_epochs in np.arange(start, end, step):

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, n_epochs, 0)

            # adjust the models that are still loaded
            self.DataGenerator.model_adjust(epochs=n_epochs)

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'_{analysis_name}_{n_epochs}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{n_epochs}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, n_epochs, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(n_epochs)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_batch_size(self, start=100, end=1500, step=100,
                                      timer=True, save_timer=False, save_models=False, save_data=False,
                                      statistical_similarity=True,
                                      identity_disclosure=True, attribute_disclosure=True,
                                      replication=True,
                                      models_to_evaluate=None, default_model=True, analysis_name='batch_size'):
        """
        Generate synthetic data that is using a synthetic data generation model that is trained with a
        varying batch size. More precisely, the function changes the batch size
        initiates re-training and generates a sample from this new model.
        This process continues until the specified end is reached.

        :param int start: smallest value batch size
        :param int end: largest value batch size
        :param int step: specify the step size to change the batch size with
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the batch size')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of model parameter batch size')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # batch_size can only be evaluated for deep learning models
        # noinspection PyUnresolvedReferences
        model_names = [name for name in self.DataGenerator.LoadedModels.keys() if name in self.DeepLearningModels]

        # remove models that are not te be evaluated to prevent unnecessary operations
        self._generator_remove_non_evaluated_models(model_names)

        for n_batch_size in np.arange(start, end, step):

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, 0, n_batch_size)

            # adjust the models that are still loaded
            self.DataGenerator.model_adjust(batch_size=int(n_batch_size))

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'{analysis_name}_{n_batch_size}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{n_batch_size}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, n_batch_size, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(n_batch_size)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_embedding_dim(self, start=20, end=560, step=40,
                                         timer=True, save_timer=False, save_models=False, save_data=False,
                                         statistical_similarity=True,
                                         identity_disclosure=True, attribute_disclosure=True,
                                         replication=True,
                                         models_to_evaluate=None, default_model=True, analysis_name='embedding_dim'):
        """
        Generate synthetic data that is using a synthetic data generation model that is trained with a
        varying number of epochs. More precisely, the function changes the number of epochs
        initiates re-training and generates a sample from this new model.
        This process continues until the specified end is reached.

        :param int start: smallest value for embedding dimension
        :param int end: the largest value for embedding dimension
        :param int step: specify the step size to change the embedding dimension with
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the embedding dimension')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of model parameter embedding dimension')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # epochs can only be evaluated for deep learning models
        # noinspection PyUnresolvedReferences
        model_names = [name for name in self.DataGenerator.LoadedModels.keys() if name in self.DeepLearningModels]

        # remove models that are not te be evaluated to prevent unnecessary operations
        self._generator_remove_non_evaluated_models(model_names)

        for embedding_dim in np.arange(start, end, step):

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, embedding_dim, 0)

            # adjust the models that are still loaded
            self.DataGenerator.model_adjust(embedding_dim=embedding_dim)

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'_{analysis_name}_{embedding_dim}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{embedding_dim}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, embedding_dim, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(embedding_dim)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_log_frequency(self, timer=True, save_timer=False, save_models=False, save_data=False,
                                         statistical_similarity=True,
                                         identity_disclosure=True, attribute_disclosure=True,
                                         replication=True,
                                         models_to_evaluate=None, default_model=True, analysis_name='log_frequency'):
        """
        Generate synthetic data that is using a synthetic data generation model that is trained with
        log frequency enabled or not.

        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the effect of log frequency')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of model parameter log frequency')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # epochs can only be evaluated for deep learning models
        # noinspection PyUnresolvedReferences
        model_names = [name for name in self.DataGenerator.LoadedModels.keys() if name in self.DeepLearningModels]

        # remove models that are not te be evaluated to prevent unnecessary operations
        self._generator_remove_non_evaluated_models(model_names)

        # vary between false and true; true is SDV default but for sake of completeness included
        _log_frequency = [False, True]

        for log_frequency in _log_frequency:

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, f'{log_frequency}', 0)

            # adjust the models that are still loaded
            self.DataGenerator.model_adjust(log_frequency=log_frequency)

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'_{analysis_name}_{log_frequency}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{log_frequency}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, f'{log_frequency}', 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(log_frequency)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_distribution(self, default_distribution=None, field_distribution=None,
                                        timer=True, save_timer=False, save_models=False, save_data=False,
                                        statistical_similarity=True,
                                        identity_disclosure=True, attribute_disclosure=True,
                                        replication=True,
                                        models_to_evaluate=None, default_model=True, analysis_name='distribution'):
        """
        Generate synthetic data that is using a synthetic data generation model that is trained with a
        varying batch size. More precisely, the function changes the default
        initiates re-training and generates a sample from this new model.
        This process continues until the specified end is reached.

        :param str or list default_distribution: smallest value batch size
        :param dict field_distribution: specify exceptions to the default distribution as
        {variable_name_x: distribution_x, variable_name_y: distribution_y, et cetera}
        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating the default distributions')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of the default distribution batch size')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        # default distribution can only be evaluated for machine learning models or models with aspects thereof
        # noinspection PyUnresolvedReferences
        model_names = [name for name in self.DataGenerator.LoadedModels.keys() if name in self.MachineLearningModels]

        # remove models that are not te be evaluated to prevent unnecessary operations
        self._generator_remove_non_evaluated_models(model_names)

        if default_distribution is None:
            default_distribution = ['gaussian', 'gamma', 'beta', 'student_t', 'gaussian_kde', 'truncated_gaussian']

        for distribution in default_distribution:

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, distribution, 0)

            if isinstance(field_distribution, dict) is True:
                # adjust the models that are still loaded
                self.DataGenerator.model_adjust(default_distribution=distribution,
                                                field_distributions=field_distribution)
            else:
                # adjust the models that are still loaded
                self.DataGenerator.model_adjust(default_distribution=distribution)

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'{analysis_name}_{distribution}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{distribution}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, distribution, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(distribution)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation
        self.evaluate_an_analysis(f'{analysis_name}', statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    # noinspection DuplicatedCode
    def generator_evaluate_models(self, timer=True, save_timer=False, save_models=False, save_data=False,
                                  statistical_similarity=True,
                                  identity_disclosure=True, attribute_disclosure=True,
                                  replication=True,
                                  models_to_evaluate=None, default_model=True, analysis_name='base_models'):
        """
        Generate synthetic data with various models without performing any sub-sampling or optimisation steps

        :param bool timer: specify whether to time the evaluation
        :param bool save_timer: specify whether to save the timer dictionary to JSON
        :param bool save_models: specify whether to save the generator models created with each step
        :param bool save_data: specify whether to save the synthetic data created with each step
        :param bool statistical_similarity: specify whether to compute the statistical similarity
        :param bool identity_disclosure: specify whether to compute the identity disclosure
        :param bool attribute_disclosure: specify whether to compute the attribute disclosure
        :param bool replication: specify whether to replicate a study as defined in the evaluation settings
        :param list models_to_evaluate: specify the models to evaluate
        :param bool default_model: specify whether to use SDV default parameters
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Evaluating different base models')

        # ensure that a generator is present
        if self.DataGenerator is None:
            self._generator_ensure_presence('Evaluation of the default distribution batch size')

        if self.DataGenerator is None:
            return

        # replace existing models to ensure that previously evaluated models are not used e.g., models with little input
        self.DataGenerator.LoadedModels = self.GeneratorModels

        # specify whether to use SDV's defaults or those set as default in script
        if isinstance(default_model, bool):
            self.DataGenerator.SDVDefaults = default_model

        # ensure models are selected if not specified correctly
        if isinstance(models_to_evaluate, list):
            self.DataGenerator.model_selection(models_to_evaluate)
        else:
            self.DataGenerator.data_find_model()

        # store a copy of the loaded models to ensure that the base models are available
        self.GeneratorModels = copy.deepcopy(self.DataGenerator.LoadedModels)

        for model in self.GeneratorModels:

            # evaluate generative models manually to ensure timing functionality
            self.DataGenerator.LoadedModels = {model: self.GeneratorModels[model]}

            if timer:
                # call timer to store start time
                self.evaluate_time(analysis_name, model, 0)

            # re-model the data
            self.DataGenerator.model_data(save=save_models, name_appendices=f'{analysis_name}_{model}')

            # retrieve synthetic data
            sample_name = self.DataGenerator.model_sample(self.BaseNumberOfSamples, save=save_data, return_name=True,
                                                          name_appendices=f'_evaluation_{analysis_name}_{model}')

            if timer:
                # call timer to store difference
                self.evaluate_time(analysis_name, model, 0, sample_name)

            # store variable for later evaluation
            self.VariableValues.append(model)

        # save the timer when desired
        if save_timer:
            if timer:
                file_handling.save_json(self.Timer[analysis_name],
                                        f'{self.OutputPath}{self.OriginalName}_{analysis_name}.json')
            else:
                warnings.warn(f'The analysis was not timed, not generating an empty json file.')

        self.SyntheticData.update({analysis_name: self.DataGenerator.SyntheticData})

        # perform evaluation; line-width set to zero to ensure different models' performance is not associated
        self.evaluate_an_analysis(f'{analysis_name}', line_width=0, statistical_similarity=statistical_similarity,
                                  identity_disclosure=identity_disclosure, attribute_disclosure=attribute_disclosure,
                                  replication=replication)

    def _generator_ensure_presence(self, evaluation_name):
        """
        Ensures that there is a generator as it is necessary for certain evaluations

        :param str evaluation_name: name of the evaluation in need for a generator
        """
        warnings.warn(f'{evaluation_name} can only performed with an initialised generator.\n')
        generate = input(f'Would you like to initialise the generator? Enter yes or no')

        if generate in 'no':
            # answering no will result in break as the evaluation is illogical to perform
            return None

        elif generate in 'yes':
            clear_synthetic = input(f'Would you like to sustain the already read synthetic data?'
                                    f'Please be cautious as mismatches might occur.\nEnter yes or no')

            # clear the read synthetic data to prevent mismatches unless specified
            if clear_synthetic not in 'yes':
                self.SyntheticData = {}

            # initiate the generator
            self._initiate_generator()

        else:
            # force the initialisation of a generator if a clear no is not provided
            warnings.warn(f'Input not recognised, initialising generator')
            clear_synthetic = input(f'Would you like to sustain the already read synthetic data?'
                                    f'Please be cautious as mismatches might occur.\nEnter yes or no')

            # clear the read synthetic data to prevent mismatches unless specified
            if 'yes' not in clear_synthetic:
                self.SyntheticData = {}

            self._initiate_generator()

    def _generator_remove_non_evaluated_models(self, model_names):
        """
        Remove models that are not loaded or existent based on name

        :param list model_names: specify the model name(s) in a list of strings
        """
        # remove models that are not to be evaluated from the loaded models
        for model_name in self.DataGenerator.LoadedModels.keys():
            if model_name not in model_names:
                logging.debug(f'Removing model {model_name}, from models loaded in the generator')
                del self.DataGenerator.LoadedModels[model_name]
            else:
                continue

    def _test_metrics(self, variation=0.10, limit=600, analysis_name='sample_of_original'):
        """
        Evaluate the behaviour of the metrics when comparing original data with original data

        :param float variation: specify the proportion that the sample size should vary per step e.g., deciles
        :param int limit: specify the lowest number of input samples (i.e., the lower boundary to step towards)
        :param str analysis_name: specify a name for the analysis
        """
        logging.info(f'Drawing samples from the original data')

        # set up the base input sample size
        n_input = len(self.OriginalData)

        self.SyntheticData.update({analysis_name: {str(n_input): self.OriginalData}})

        # store variable for later evaluation
        self.VariableValues.append(n_input)

        while n_input >= limit:
            # determine sample size of subset
            n_input = int(n_input * (1 - variation))

            # generate a subset and place in class's original data
            logging.debug(f'Adjusting sample size to {n_input}')

            self.SyntheticData[analysis_name].update({str(n_input): self.OriginalData.sample(n_input)})

            # store variable for later evaluation
            self.VariableValues.append(n_input)
