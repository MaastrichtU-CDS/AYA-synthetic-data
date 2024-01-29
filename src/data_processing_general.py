"""
Aims to provide general processing functions
"""
import logging
import sdv.constraints

import pandas as pd

# from sdv.constraints import OneHotEncoding
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

dummy_prefix = '_dummy_'


def impute_mice(dataset, zero_threshold=None):
    """
    Impute missing values using sci-kit's implementation of chained equations namely: IterativeImputer
    The contents of variables that contain only missing data are set to zero

    :param pandas.DataFrame dataset: the dataset in which missing values are to be imputed
    :param float zero_threshold: define the proportion of values that can be missing,
    variables that pass the threshold will be imputed with zero
    :return: pandas.DataFrame of the imputed dataset
    """
    # check if an appropriate threshold was given
    if isinstance(zero_threshold, float) is False:
        zero_threshold = 0.66

    logging.debug(f'Using {zero_threshold} as threshold to zero-impute missing values.')

    # retrieve columns with more than threshold proportion of missing values
    high_missing_variables = [variable_name for variable_name in dataset.columns if
                              (dataset[variable_name].isna().sum() / len(dataset[variable_name])) >= zero_threshold]

    # impute a value of 0 for those with more missing considered desirable
    dataset[high_missing_variables] = dataset[high_missing_variables].fillna(0)

    # initialise the imputation
    logging.info(f'Imputing using the MICE / Iterative Imputation procedure')

    # use most frequent to cater for the plethora of categorical variables
    imputation = IterativeImputer(keep_empty_features=True, initial_strategy='most_frequent')

    # impute data and convert back to pandas.DataFrame
    return pd.DataFrame(imputation.fit_transform(dataset), columns=dataset.keys())


def normalise(dataset, metadata):
    """
    Normalise the given dataset columns in case they are defined as numerical in the given metadata

    :param pandas.DataFrame dataset: dataset containing values to normalise
    :param dict metadata: SDV metadata dictionary containing fields of numerical type
    :return: normalised DataFrame and metadata dictionary that only contains float subtypes instead of
    integer and float subtypes
    """
    logging.info(f'Normalising numerical variables')

    # retrieve which variables are numerical
    numerical_variables = [variable_name for variable_name in metadata['fields'].keys() if
                           metadata['fields'][variable_name]['type'] == 'numerical']

    # ensure that numerical_variables are defined as numeric
    dataset[numerical_variables] = dataset[numerical_variables].apply(pd.to_numeric)

    # normalise the values
    dataset[numerical_variables] = (dataset[numerical_variables] - dataset[numerical_variables].min()) / \
                                   (dataset[numerical_variables].max() - dataset[numerical_variables].min())

    # round values to circumvent faulty rounding in SDV RDF module; https://github.com/sdv-dev/SDV/issues/1039
    dataset[numerical_variables] = dataset[numerical_variables].round(14)

    # create new field dictionary for normalised variables
    metadata_int_to_float = {variable_name: {'type': 'numerical', 'subtype': 'float'}
                             for variable_name in numerical_variables
                             if 'integer' in metadata['fields'][variable_name]['subtype']}

    # update the numerical types to ensure float datatype
    metadata['fields'].update(metadata_int_to_float)

    return dataset, metadata


def de_normalise(dataset, original_dataset, original_metadata):
    """
    De-normalise the given dataset columns in case they are defined as numerical in the original metadata.
    That is, this function will convert the normalised values to their original scales

    :param pandas.DataFrame dataset: dataset containing values to de-normalise
    :param pandas.DataFrane original_dataset: dataset containing the original values of the normalised variables
    :param dict original_metadata: SDV metadata dictionary containing fields of numerical type
    including integers that have been converted to float during normalisation
    :return: de-normalised DataFrame in which normalised numerical values are returned to their original scale
    """
    logging.info(f'De-normalising numerical variables')

    # retrieve which variables are numerical
    numerical_variables = [variable_name for variable_name in original_metadata['fields'].keys() if
                           original_metadata['fields'][variable_name]['type'] == 'numerical']

    # ensure that numerical_variables are defined as numeric
    dataset[numerical_variables] = dataset[numerical_variables].apply(pd.to_numeric)
    original_dataset[numerical_variables] = original_dataset[numerical_variables].apply(pd.to_numeric)

    # de-normalise the values
    dataset[numerical_variables] = (dataset[numerical_variables] * (original_dataset[numerical_variables].max()
                                                                    - original_dataset[numerical_variables].min())
                                    + original_dataset[numerical_variables].min())

    integers = [variable_name for variable_name in numerical_variables if
                original_metadata['fields'][variable_name]['subtype'] == 'integer']

    # set integers to integers for cleanliness
    dataset[integers] = dataset[integers].round(0)

    return dataset


def dummify(dataset, metadata, one_hot_prefix=None):
    """
    Create dummy variables i.e., a one-hot encoding scheme, for fields/variable declared as categorical in the metadata

    :param pandas.DataFrame dataset: dataset containing categorical fields/variables to create dummies of
    :param dict metadata: dictionary containing fields declared as categorical
    :param string one_hot_prefix: Specify the prefix to add to the dummy variable name,
    change with caution as this prefix is used for de-dummification i.e., reverse this function. Will default to _dummy_
    :return: pandas.Dataframe of the given dataset in which categorical variables are reshaped to dummies,
    dict of adapted metadata in which the dummies are accounted for as booleans
    dict of OneHot SDV constraints per categorical variable
    """
    # force a prefix to ease addition to metadata
    if one_hot_prefix is None:
        one_hot_prefix = dummy_prefix

    logging.info(f'One-hot encoding categorical variables using prefix {one_hot_prefix}')

    # retrieve which variables are categorical
    categorical_variables = [variable_name for variable_name in metadata['fields'].keys() if
                             metadata['fields'][variable_name]['type'] == 'categorical']

    # create dummies
    dataset = pd.get_dummies(dataset, prefix_sep=one_hot_prefix, columns=categorical_variables)

    # retrieve the names of the dummies
    dummies = [variable_name for variable_name in dataset.columns.to_numpy() if one_hot_prefix in variable_name]

    # create a base metadata dictionary to include additional items that are not part of the field info
    metadata_new = {variable_name: variable_info for variable_name, variable_info
                    in metadata.items() if variable_name not in 'fields'}

    # create a new dictionary that does not contain categorical variables
    metadata_without_categorical = {'fields':
                                        {variable_name: variable_info for variable_name, variable_info
                                         in metadata['fields'].items() if variable_name not in categorical_variables}}

    # add the fields that are not categorical
    metadata_new.update(metadata_without_categorical)

    # create a dictionary with only dummy variables
    metadata_of_dummies = {variable_name: {'type': 'boolean'} for variable_name in dummies}

    # add the new metadata fields to the metadata that was stripped of categorical_variables
    metadata_new['fields'].update(metadata_of_dummies)

    # future implementation
    # add constraint that one-hot can only have one value per original category
    # one_hot_constraints = \
    #     {variable_name: OneHotEncoding(column_names=[dummy_variable for dummy_variable in dummies
    #                                                  if variable_name ==
    #                                                  dummy_variable[0: dummy_variable.rfind(one_hot_prefix)]])
    #                                                  for variable_name in categorical_variables}

    # circumvent SDV bug constraint loading bug as in e.g., https://github.com/sdv-dev/SDV/issues/1115
    one_hot_constraints = [{'constraint': sdv.constraints.OneHotEncoding,
                            'column_names': [dummy_variable for dummy_variable in dummies
                                             if
                                             variable_name == dummy_variable[0: dummy_variable.rfind(one_hot_prefix)]]
                            } for variable_name in categorical_variables]

    # check whether there already were certain constraints
    if 'constraints' not in metadata.keys():
        # create constraints key in metadata
        metadata_new.update({'constraints': []})

    # retrieve the existing constraints
    existing_constraints = metadata_new['constraints']

    # add the new constraints to the existing ones
    new_constraints = existing_constraints + one_hot_constraints

    # place new list in constraints
    metadata_new['constraints'] = new_constraints

    return dataset, metadata_new, one_hot_constraints


def de_dummifiy(dataset, metadata_w_dummies=None, one_hot_prefix=None):
    """
    Remove dummy variables i.e., a one-hot encoding scheme using the one_hot_prefix as identifier of the dummy

    :param pandas.DataFrame dataset: dataset containing dummy fields/variables that are to be converted to categorical
    :param dict metadata_w_dummies: dictionary containing dummy fields/variables
    :param string one_hot_prefix: Specify the prefix to add to the dummy variable name,
    change with caution as this prefix is used for de-dummification i.e., remove dummies. Will default to _dummy_
    :return: pandas.Dataframe of the given dataset in which dummy variables are reshaped to categorical
    """
    # force a prefix to ease addition to metadata
    if one_hot_prefix is None:
        one_hot_prefix = dummy_prefix

    logging.info(f'Reshaping one-hot encoded variables to original categorical state using trigger {one_hot_prefix}')

    # allow to only use dataset if metadata is not present
    if metadata_w_dummies is None:
        variables = dataset.columns.to_numpy()
    else:
        variables = metadata_w_dummies['fields'].keys()

    # retrieve the names of dummy variables by the string trigger
    dummies = [variable_name for variable_name in variables if one_hot_prefix in variable_name]

    # retrieve original categories
    dataset_de_dummified = pd.from_dummies(dataset[dummies], sep=f'{one_hot_prefix}')

    # retrieve dataset without the categorical/dummified variables
    dataset = dataset.drop(dummies, axis=1)

    # concatenate the two dataframes
    dataset = pd.concat([dataset, dataset_de_dummified], axis=1)

    return dataset
