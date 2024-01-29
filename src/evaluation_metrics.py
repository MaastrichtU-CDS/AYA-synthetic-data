"""
Aims to provide numerous functions that can be used to evaluate the quality of synthetic data
"""
import copy
import logging
import os
import patsy
import statsmodels.tools.sm_exceptions
import warnings

import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from itertools import chain, combinations
from prdc import compute_prdc
from scipy.spatial.distance import cdist
from sdmetrics.single_table import CategoricalCAP
from sklearn import metrics as skl
from statsmodels.formula.api import logit as logisticregression

# private modules
import src.file_handling


def compute_stat_similarity(original_data, synthetic_data, nearest_k=5):
    """
    Compute the statistical similarity in terms of precision, recall, diversity, and coverage of the synthetic data
    to the original data

    :param pandas.DataFrame original_data: original data that is to be compared with the synthetic dataset
    :param pandas.DataFrame synthetic_data: synthetic dataset that is to be compared with the original dataset
    :param int nearest_k: specify the distance for nearest neighbour; as in PMLR 119:7176-7185,2020
    :return: dict containing the statistical similarity metrics
    """
    logging.info(f'Computing statistical similarity using precision, recall, diversity, and coverage')

    # convert DataFrame to numpy array
    original_data = original_data.to_numpy(dtype='float64')
    synthetic_data = synthetic_data.to_numpy(dtype='float64')

    # determine precision, recall, diversity, and coverage by Mohammed F. Naeem in PMLR 119:7176-7185,2020
    return compute_prdc(original_data, synthetic_data, nearest_k)


def compute_privacy_identity(original_key, synthetic_key, categorical_data=None, continuous_data=None,
                             minimum_variables=3):
    """
    Compute the identity disclosure score using Euclidean and Hamming distance for continuous and categorical/boolean
    data respectively. Can cope if only a single data type is provided, e.g., no categorical but only continuous data.

    :param any original_key: specify the key that will be used to denote the original data
    :param any synthetic_key: specify the key that will be used to denote the synthetic data
    :param dict categorical_data: dictionary that contains original and synthetic categorical/boolean data
    :param dict continuous_data: dictionary that contains original and synthetic continuous data
    :param int minimum_variables: define the minimum number of variables necessary to compute identity disclosure
    i.e., prevent calculating identity disclosure when e.g., age, is the only continuous variable
    :return: scores on identity disclosure
    """
    logging.info(f'Computing identity disclosure for {original_key} and {synthetic_key}')

    # set up score, necessary as both are not always calculated
    identity_disclosure_stats = None
    categorical_distance_array = None
    continuous_distance_array = None

    categorical_duplicates = np.nan
    categorical_disclosure_min = np.nan
    categorical_disclosure_q_one = np.nan
    categorical_disclosure_median = np.nan
    categorical_disclosure_q_three = np.nan
    categorical_disclosure_max = np.nan
    continuous_duplicates = np.nan
    continuous_disclosure_min = np.nan
    continuous_disclosure_q_one = np.nan
    continuous_disclosure_median = np.nan
    continuous_disclosure_q_three = np.nan
    continuous_disclosure_max = np.nan

    # ensure data is provided
    if isinstance(categorical_data, dict) is False and isinstance(continuous_data, dict) is False:
        warnings.warn(f'Please provide original and synthetic data in a dictionary.\n'
                      f'Provide two separate dictionaries if you would like to compute '
                      f'the identity disclosure for both categorical and continuous variable')
        return identity_disclosure_stats

    # allow to only compute identity disclosure for a single data type
    if categorical_data is not None:
        if isinstance(categorical_data, dict) is False:
            warnings.warn(
                f'Please provide the data in a dictionary with a corresponding dictionary key for the two datasets')
            return identity_disclosure_stats

        # prevent computation of a certain number of variables, e.g., only computing the distance of biological sex
        if len(categorical_data[original_key].columns) < minimum_variables or \
                len(categorical_data[synthetic_key].columns) < minimum_variables:
            logging.info(
                f'Provided categorical data consists of less than {minimum_variables} variables, '
                f'categorical identity disclosure will not be computed')
        else:
            # retrieve categorical data
            original_categorical = categorical_data[original_key]
            synthetic_categorical = categorical_data[synthetic_key]

            logging.debug(f'Computing Hamming distance')
            categorical_distance_array = cdist(original_categorical.iloc[:, 1:], synthetic_categorical.iloc[:, 1:],
                                               metric='hamming')

            # compute statistics
            categorical_duplicates, categorical_disclosure_min, categorical_disclosure_q_one, \
                categorical_disclosure_median, categorical_disclosure_q_three, categorical_disclosure_max \
                = compute_privacy_disclosure_stats(categorical_distance_array)

    # allow to only compute identity disclosure for a single data type
    if continuous_data is not None:
        if isinstance(continuous_data, dict) is False:
            warnings.warn(
                f'Please provide the data in a dictionary with a corresponding dictionary key for the two datasets')
            return identity_disclosure_stats

        # prevent computation of a certain number of variables, e.g., only computing the distance of age
        if len(continuous_data[original_key].columns) < minimum_variables or \
                len(continuous_data[synthetic_key].columns) < minimum_variables:
            logging.info(
                f'Provided continuous data consists of less than {minimum_variables} variables, '
                f'continuous identity disclosure will not be computed')
        else:
            # retrieve continuous data
            original_continuous = continuous_data[original_key]
            synthetic_continuous = continuous_data[synthetic_key]

            logging.debug(f'Computing Euclidean distance')
            continuous_distance_array = cdist(original_continuous.iloc[:, 1:], synthetic_continuous.iloc[:, 1:],
                                              metric='euclidean')

            # compute statistics
            continuous_duplicates, continuous_disclosure_min, continuous_disclosure_q_one, \
                continuous_disclosure_median, continuous_disclosure_q_three, continuous_disclosure_max \
                = compute_privacy_disclosure_stats(continuous_distance_array)

    logging.debug(f'Returning identity disclosure scores as overall score for the dataset as float')

    return {'categorical duplicates': categorical_duplicates,
            'categorical disclosure min': categorical_disclosure_min,
            'categorical disclosure first quartile': categorical_disclosure_q_one,
            'categorical disclosure median': categorical_disclosure_median,
            'categorical disclosure third quartile': categorical_disclosure_q_three,
            'categorical disclosure max': categorical_disclosure_max,
            'continuous duplicates': continuous_duplicates,
            'continuous disclosure min': continuous_disclosure_min,
            'continuous disclosure first quartile': continuous_disclosure_q_one,
            'continuous disclosure median': continuous_disclosure_median,
            'continuous disclosure third quartile': continuous_disclosure_q_three,
            'continuous disclosure max': continuous_disclosure_max}, \
        categorical_distance_array, continuous_distance_array


def compute_privacy_attribute(original_data, synthetic_data, known_variables, sensitive_variables):
    """
    Compute the attribute disclosure score using the Correct Attribution Probability (CAP) by SDMetrics
    Detailed information can be found at https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/categoricalcap

    :param pandas.DataFrame original_data: the dataset containing the original data
    :param pandas.DataFrame synthetic_data: the dataset containing the synthesised data
    :param list known_variables: names of variables known to the guessing party
    :param list sensitive_variables: names of variables that are to be guessed
    :return: scores on attribute disclosure
    """
    # ensure that pandas sees variables of choice as categorical
    categorical_variables = {variable: 'category' for variable in known_variables}
    categorical_variables.update({variable: 'category' for variable in sensitive_variables})

    original_data = original_data.astype(categorical_variables)
    synthetic_data = synthetic_data.astype(categorical_variables)

    # retrieve all possible combinations of exposed variables, e.g., knowing only age, sex, and age and sex
    known_combinations = list(chain(*(list(combinations(known_variables, i + 1))
                                      for i in range(len(known_variables)))))
    known_combinations = list(list(x) for x in known_combinations if x != ())

    # retrieve all possible combinations of sensitive variables, e.g., emotional, and physical functioning, and both
    sensitive_combinations = list(chain(*(list(combinations(sensitive_variables, i + 1))
                                          for i in range(len(sensitive_variables)))))
    sensitive_combinations = list(list(x) for x in sensitive_combinations if x != ())

    # store every attribute disclosure score per combination in a numpy array
    disclosure_array = np.empty((len(known_combinations), len(sensitive_combinations)), dtype=float)
    for index, known_variable in enumerate(known_combinations):
        # first create a list of possible combinations of sensitive variables, then store in numpy array
        disclosure_array[index] = np.asarray(
            [CategoricalCAP.compute(real_data=original_data, synthetic_data=synthetic_data,
                                    key_fields=known_variable,
                                    sensitive_fields=sensitive_variable)
             for sensitive_variable in sensitive_combinations])

    # compute statistics
    attribute_disclosure_min, attribute_disclosure_q_one, attribute_disclosure_median, \
        attribute_disclosure_q_three, attribute_disclosure_max = compute_privacy_disclosure_stats(disclosure_array,
                                                                                                  count_duplicates=False)

    logging.debug(f'Returning attribute disclosure scores as overall score for the dataset as float')

    return {'attribute disclosure min': attribute_disclosure_min,
            'attribute disclosure first quartile': attribute_disclosure_q_one,
            'attribute disclosure median': attribute_disclosure_median,
            'attribute disclosure third quartile': attribute_disclosure_q_three,
            'attribute disclosure max': attribute_disclosure_max}


# noinspection PyArgumentList
def compute_privacy_disclosure_stats(score_array, count_duplicates=True):
    """
    Compute the min, first quartile, median, third quartile, and max of the given array

    :param numpy.nd_array score_array: an array representing the distance of each row in dataset x to
     each row in dataset y
    :param bool count_duplicates: specify whether to count the incidence of zero in the array
    :return: min, first quartile, median, third quartile, and max of the given array
    """
    logging.debug(f'Calculating privacy disclosure statistics')

    # take the median distance
    lowest = np.min(score_array)
    first_quartile = np.percentile(score_array, 25, interpolation='midpoint')
    median = np.median(score_array)
    third_quartile = np.percentile(score_array, 75, interpolation='midpoint')
    highest = np.max(score_array)

    if count_duplicates:
        number_of_duplicates = np.count_nonzero(score_array == 0)
        return number_of_duplicates, lowest, first_quartile, median, third_quartile, highest
    else:
        return lowest, first_quartile, median, third_quartile, highest


def logistic_regression(dataset, r_formula, odds_ratio=True, alpha=0.05, original_data_outcome=None,
                        validation_data=None, univariate=True):
    """
    Perform logistic regression analysis and retrieve the coefficient/odds ratio and the associated confidence interval

    :param pandas.DataFrame dataset: the dataset to fit the logistic regression model on
    :param str r_formula: R-style formula to perform the logistic regression with
    :param bool odds_ratio: specify whether to exponentiate the coefficient; to directly obtain an odds ratio
    :param float alpha: determine the alpha that is to be used to calculate the confidence interval
    :param dict original_data_outcome: logistic regression outcome of the original data
    :param pandas.DataFrame validation_data: the dataset that will be used to calculate the concordance statistic for
    :param bool univariate: specify whether the analysis is univariate
    :return: sorted dictionary containing the odds' ratio, lower and upper confidence interval for every parameter
    """
    model_output = {}
    formulas = [r_formula]
    concordance_statistic = np.nan
    concordance_statistic_adjusted = np.nan

    # specify whether to convert the coefficient to odds ratio
    if odds_ratio:
        coefficient_name = 'odds ratio'
    else:
        coefficient_name = 'beta'

    logging.info(f'Performing logistic regression for formula {r_formula} with an alpha of {alpha}')

    if univariate:
        _outcome = r_formula[:r_formula.rfind('~')].strip()
        _formula = r_formula[r_formula.find('~') + 1:].strip()

        for variables in range(0, _formula.count('+') + 1):
            if '+' in _formula:
                # retrieve a variable
                _variable = _formula[:_formula.find('+')].strip()
                # remove variable that has been used
                _formula = _formula[_formula.find('+') + 1:]
            else:
                _variable = _formula.strip()

            # add the formula to the list of formulas to run
            __formula = f'{_outcome} ~ {_variable}'
            formulas.append(__formula)

    for formula in formulas:
        try:
            if '+' in formula:
                name_prefix = ''
            else:
                name_prefix = 'univariate - '

            # initiate and fit model
            logistic_model = logisticregression(formula=formula, data=dataset).fit()

            # exponentiate if odds ratios are desired
            if odds_ratio:
                _coefficients = np.exp(logistic_model.params).to_dict()
                _confidence_intervals = np.exp(logistic_model.conf_int(alpha=alpha)).to_dict()
            else:
                _coefficients = logistic_model.params.to_dict()
                _confidence_intervals = logistic_model.conf_int(alpha=alpha).to_dict()

            if validation_data is not None and '+' in formula:
                outcome = formula[:formula.find('~')].strip()
                # calculate concordance statistic
                try:
                    concordance_statistic = skl.roc_auc_score(validation_data[outcome], logistic_model.predict(
                        validation_data.loc[:, ~validation_data.columns.isin([outcome])]))
                except patsy.PatsyError:
                    pass
                except statsmodels.tools.sm_exceptions.PerfectSeparationError:
                    pass

                validation_data_rus = validation_data.copy(deep=True)
                validation_resampler = RandomUnderSampler(random_state=53)
                validation_data_rus.loc[:, ~validation_data_rus.columns.isin([outcome])], validation_data_rus[outcome] = \
                    validation_resampler.fit_resample(
                        validation_data_rus.loc[:, ~validation_data_rus.columns.isin([outcome])],
                        validation_data_rus[outcome])
                validation_data_rus.dropna(axis=0, inplace=True)

                dataset_rus = dataset.copy(deep=True)

                test_resampler = RandomUnderSampler()
                dataset_rus.loc[:, ~dataset_rus.columns.isin([outcome])], dataset_rus[outcome] = \
                    test_resampler.fit_resample(dataset_rus.loc[:, ~dataset_rus.columns.isin([outcome])],
                                                dataset_rus[outcome])

                dataset_rus.dropna(axis=0, inplace=True)
                # refit logistic regression
                logistic_model_rus = logisticregression(formula, dataset_rus).fit()

                # calculate concordance statistic
                try:
                    concordance_statistic_adjusted = skl.roc_auc_score(validation_data_rus[outcome],
                                                                       logistic_model_rus.predict(
                                                                           validation_data_rus.loc[:,
                                                                           ~validation_data_rus.columns.isin(
                                                                               [outcome])]))
                except patsy.PatsyError:
                    pass
                except statsmodels.tools.sm_exceptions.PerfectSeparationError:
                    pass

            # reshape to cater for output table
            _model_output = {f'{name_prefix}{variable_name} {coefficient_name}': variable_info
                             for variable_name, variable_info in _coefficients.items()}
            _model_output.update(
                {f'{name_prefix}{variable_name} {1 - alpha}% confidence interval (lower)': variable_info
                 for variable_name, variable_info in _confidence_intervals[0].items()})
            _model_output.update(
                {f'{name_prefix}{variable_name} {1 - alpha}% confidence interval (upper)': variable_info
                 for variable_name, variable_info in _confidence_intervals[1].items()})

            # sort the variables to their original order for cleanliness
            for variable in logistic_model.params.keys().to_list():
                model_output.update(
                    {variable_name: variable_info for variable_name, variable_info in _model_output.items()
                     if variable in variable_name})

            # include variables that might not be present
            if original_data_outcome is not None and '+' in formula:
                model_output.update({f'{variable}': np.nan for variable in original_data_outcome.keys()
                                     if variable not in _model_output.keys()})

            # add the concordance statistics last to prevent that they are set to nan in previous actions
            if validation_data is not None:
                model_output.update({f'c-statistic': concordance_statistic,
                                     f'adjusted c-statistic': concordance_statistic_adjusted})

        # in small samples variables might be constant
        except np.linalg.LinAlgError:

            # return np.nan in case logistic regression failed
            if original_data_outcome is not None:
                _model_output = {f'{variable}': np.nan for variable in original_data_outcome.keys()}
            else:
                _model_output = {f'no variables were modelled': np.nan}

        except statsmodels.tools.sm_exceptions.PerfectSeparationError:

            # return np.nan in case logistic regression failed
            if original_data_outcome is not None:
                _model_output = {f'{variable}': np.nan for variable in original_data_outcome.keys()}
            else:
                _model_output = {f'no variables were modelled': np.nan}

        except patsy.PatsyError:

            # return np.nan in case logistic regression failed
            if original_data_outcome is not None:
                _model_output = {f'{variable}': np.nan for variable in original_data_outcome.keys()}
            else:
                _model_output = {f'no variables were modelled': np.nan}

    return model_output


def compute_number_needed_to_print(evaluation_folders,
                                   statistical_similarity_metric_to_plot=None, statistical_similarity_limit=0.40,
                                   privacy_disclosure_metric=None, privacy_disclosure_limit=None,
                                   file_key_word=('n_input_', '_n_output.csv')):
    """
    Retrieve the largest number of samples that can be generated without exceeding the defined privacy limit

    :param dict evaluation_folders: model name and folder evaluations to include i.e., {model: folder}
    More specifically, folders should contain the evaluation output of synthetic datasets of increasing size,
    generated from original samples of decreasing size, as is produced by the evaluation_general.py module
    :param list statistical_similarity_metric_to_plot: metrics that are averaged and visualised as marker size,
    therefore should be included in evaluation scheme, defaults to an average of coverage and density
    :param float statistical_similarity_limit: value that can be used to exclude certain samples
    with statistical similarity scores lower than the defined limit
    :param str privacy_disclosure_metric: metric that is used to determine the maximum generated number of samples
    :param float privacy_disclosure_limit: value that can be used to exclude a certain closeness
    :param tuple file_key_word: keywords used to identify the files,
    e.g., n_input_1000_n_output.csv as is produced by the evaluation_general.py module
    :return: output of logistic regression as pandas.DataFrame
    """
    if statistical_similarity_metric_to_plot is None:
        statistical_similarity_metric_to_plot = ['coverage', 'density']

    if privacy_disclosure_metric is None:
        privacy_disclosure_metric = 'categorical disclosure min'

    _data_frame_columns = ['input', 'output',
                           f'{statistical_similarity_metric_to_plot[0]}/{statistical_similarity_metric_to_plot[1]}',
                           'model']
    nnt_data = pd.DataFrame(columns=_data_frame_columns)

    for name, data_directory in evaluation_folders.items():
        datasets = [file for file in os.listdir(data_directory)
                    if file_key_word[0] in file and file_key_word[1] in file]
        for dataset in datasets:
            # retrieve input sample size
            n_input = dataset[dataset.find(file_key_word[0]):dataset.rfind(file_key_word[1])]
            n_input = int(n_input[n_input.rfind('_') + 1:])

            # load dataset
            dataset = src.file_handling.read_csv(f'{data_directory}{os.path.sep}{dataset}',
                                                 datatype=float, remove='file_identification')

            column_to_check = [col for col in dataset.columns if file_key_word[0] in col]
            if len(column_to_check) > 1:
                logging.critical(f'Unexpected number of columns found containing {file_key_word[0]}.\n'
                                 f'Consider using different keywords.')
                exit()
            else:
                column_to_check = column_to_check[0]

            # filter out those datasets that have below satisfactory statistical similarity scores
            if isinstance(statistical_similarity_limit, (float, int)) is False:
                statistical_similarity_limit = 0.1

            for statistical_similarity_metric in statistical_similarity_metric_to_plot:
                dataset = dataset[(dataset.filter(like=statistical_similarity_metric)
                                   > statistical_similarity_limit).any(axis=1)]

            # filter out those datasets that have below satisfactory privacy disclosure
            if isinstance(privacy_disclosure_limit, float):
                _dataset = dataset[(dataset[privacy_disclosure_metric] / dataset[column_to_check] * 100) >=
                                   privacy_disclosure_limit]

                if len(_dataset) > 0:
                    try:
                        # keep the last one as it will have the largest sample size
                        highest_statistical_similarity_sample = _dataset.iloc[0]
                    except IndexError:
                        __dataset = copy.deepcopy(_dataset)
                        __dataset.loc[0] = 0
                        highest_statistical_similarity_sample = __dataset.loc[0]
                else:
                    # in case of no violations use the largest
                    highest_statistical_similarity_sample = dataset.loc[dataset[column_to_check].idxmax()]

            else:
                # try to find the first row with disclosure breach, use the largest sample size if not present
                try:
                    first_violation = dataset[dataset[privacy_disclosure_metric] == 0].index.values.astype(int)[0]
                    try:
                        highest_statistical_similarity_sample = dataset.loc[first_violation - 1]
                    except KeyError:
                        _dataset = copy.deepcopy(dataset)
                        _dataset.loc[first_violation - 1] = 0
                        highest_statistical_similarity_sample = _dataset.loc[first_violation - 1]
                except IndexError:
                    highest_statistical_similarity_sample = dataset.loc[dataset[column_to_check].idxmax()]

            _nnt_data = pd.DataFrame({_data_frame_columns[0]: n_input,
                                      _data_frame_columns[1]: highest_statistical_similarity_sample[column_to_check],
                                      _data_frame_columns[2]: (highest_statistical_similarity_sample[
                                                                   statistical_similarity_metric_to_plot[0]] +
                                                               highest_statistical_similarity_sample[
                                                                   statistical_similarity_metric_to_plot[1]]) / 2,
                                      _data_frame_columns[3]: name},
                                     index=[len(nnt_data.index) + 1])

            nnt_data = pd.concat([nnt_data, _nnt_data])

    # convert all information to integer
    nnt_data[[_data_frame_columns[0], _data_frame_columns[1]]] \
        = nnt_data[[_data_frame_columns[0], _data_frame_columns[1]]].astype(int)

    return nnt_data


def linear_predictor(coefficients, dataset, convert_odds_ratios):
    """
    Predicts probabilities using logistic regression.

    :param dict coefficients: Coefficients for logistic regression.
    :param pd.DataFrame dataset: Input data with features.
    :param bool convert_odds_ratios: If True, coefficients are assumed to be in odds ratios.
     If False, coefficients are in beta/log-odds.

    :return np.ndarray: Predicted probabilities.
    """
    updated_coefficients = coefficients.copy()  # Create a copy to avoid modifying the original dictionary

    # convert coefficients to beta coefficients if specified
    if convert_odds_ratios:
        for key, beta in coefficients.items():
            stripped_key = key.strip()
            updated_coefficients[stripped_key] = np.log(np.exp(beta))
            if stripped_key != key:
                del updated_coefficients[key]

    linear_preds = np.zeros(len(dataset))

    for feature, beta in updated_coefficients.items():
        feature_name = feature.strip()
        if 'C(' in feature_name:
            # handle categorical variable
            category = feature_name.split('[T.')[-1].split(']')[0]
            dummy_col = f"{feature_name.split('(')[-1].split(',')[0]}_{category}"
            linear_preds += dataset[dummy_col] * beta
        else:
            # handle non-categorical variable
            linear_preds += dataset[feature_name] * beta

    return linear_preds
