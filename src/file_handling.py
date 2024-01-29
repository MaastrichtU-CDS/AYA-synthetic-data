"""
Aims to provide general file handling functions.
Exception is SDV model searching and loading, which is performed in generator_sdv.py
"""

import json
import logging
import os
import warnings

import pandas as pd

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def find_items(data_directory=None, data_path=None, metadata_path=None, model_directory=None, settings_path=None):
    """
    Initialises searching for a dataset, a metadata file and returns their path and the directory in which
    models can be found. Will only search for a specific item if the associated variable is not a string.

    :param str data_directory: specify the directory in which items can be found, will be requested when not provided
    :param str data_path: specify the path in which the data can be found, if not in the specified directory
    :param str metadata_path: specify the path in which the metadata can be found if not in the specified directory
    :param str model_directory: specify the path in which the models can be found if not in the specified directory
    :param str settings_path: specify the path in which the settings can be found if not in the specified directory
    :return: strings of the data, metadata path and the directory
    """
    # retrieve directory when not provided
    if isinstance(data_directory, str) is False:
        data_directory = input(f'Please specify the path that contains the specified items\n')

    # only change the path when it is not specified for data, metadata, and the models
    if isinstance(data_path, str) is False:
        data_path = find_file(data_directory, '_clean.csv')

    if isinstance(metadata_path, str) is False:
        metadata_path = find_metadata(data_directory)

    if isinstance(model_directory, str) is False:
        model_directory = data_directory

    if isinstance(settings_path, str) is False:
        settings_path = find_file(data_directory, 'settings.json')

    return data_path, metadata_path, model_directory, settings_path


def find_file(file_directory, file_identifier):
    """
    Find data that are located in the specified directory using an identification substring.

    :param str file_directory: specify the directory which should be searched for data
    :param str file_identifier: specify the substring that is to be used for data file searching
    :return: path to selected data file
    """
    logging.info(f'Searching directory {file_directory} for files containing {file_identifier}')
    available_files = {}
    file_number = 0
    file_path = None

    # loop through settings_directory and add name to available data dictionary when it contains the identifier
    for file in os.listdir(file_directory):
        if file_identifier in file:
            file_number += 1
            available_files.update({file_number: os.path.join(file_directory, file)})
        else:
            continue

    # allow to specify what dataset is to be used in case multiple are found
    if file_number > 1:
        selected_data = int(input(f'The following files containing identifier "{file_identifier}" were found'
                                  f'\n{available_files}\n'
                                  f'Please input the number of the file that is to be used.'))
        file_path = available_files[selected_data]

    # assign filename when only one is found
    elif file_number == 1:
        file_path = available_files[1]

    # cannot proceed without any data
    elif file_number == 0:
        logging.critical(f'No files containing {file_identifier} where found in {file_directory}')
        exit(f'Exiting, ensure that all files, i.e., settings and a clean dataset are present.')

    else:
        warnings.warn(f'A file was already specified, searching for it is unnecessary.')

    return file_path


def find_metadata(metadata_directory, sdv_table_format='SDV_single_table', metadata_identifier='_metadata.json'):
    """
    Find metadata that are located in the specified directory using an identification substring and
    specify whether the data path should be retrieved from the metadata if available.

    :param str metadata_directory: specify the directory which should be searched for metadata
    :param str sdv_table_format: specify the sdv table format
    :param str metadata_identifier: specify the substring that is to be used for metadata file searching
    :return: dictionary
    """
    # identifier should not only contain identifier but also the table format as metadata format can differ
    metadata_identifier = f'{sdv_table_format}{metadata_identifier}'

    logging.info(f'Searching directory {metadata_directory} for files containing {metadata_identifier}')
    available_metadata = {}
    metadata_number = 0
    metadata_path = None

    # loop through directory and add name to available metadata dictionary when it contains the identifier
    logging.debug(f'files found in {metadata_directory}:\n {os.listdir(metadata_directory)}')
    for file in os.listdir(metadata_directory):
        if metadata_identifier in file:
            metadata_number += 1
            available_metadata.update({metadata_number: os.path.join(metadata_directory, file)})
        else:
            continue

    # allow to specify what metadata is to be used in case multiple are found
    if metadata_number > 1:
        selected_data = int(input(f'The following files containing identifier "{metadata_identifier}" were '
                                  f'found\n{available_metadata}\n'
                                  f'Please input the number of the file that is to be used. '))
        metadata_path = available_metadata[selected_data]

    # assign filename when only one is found
    elif metadata_number == 1:
        metadata_path = available_metadata[1]

    # cannot proceed without any metadata
    elif metadata_number == 0:
        logging.critical(f'No files containing {metadata_identifier} where found in {metadata_directory}')

    return metadata_path


def read_csv(filepath, datatype, missing=' ', separator=',', remove='hash_id'):
    """
    Reads a csv file and return it as pandas DataFrame object.

    :param str filepath: specify the filepath of the csv file that is to be read
    :param str datatype: specify the data type
    :param str missing: allows to specify how missing data is formatted in read data
    :param str separator: allows to specify how data is separated
    :param str remove: specify the key of a variable that has to be removed e.g. a hash identifier
    :return pandas.DataFrame
    """
    # read dataset using specified delimiter
    logging.info(f'Reading dataset from {filepath}')
    try:
        return pd.read_csv(filepath, sep=separator, dtype=datatype, na_values=missing)
    except ValueError:
        try:
            logging.debug(f'Reading dataset from {filepath} as object datatype,\n'
                          f'removing variable {remove} and converting datatype to {datatype}')
            # read as object to then remove the specified variable
            dataset = pd.read_csv(filepath, sep=separator, dtype=object, na_values=missing)

            # certain variables might have to be removed
            dataset.pop(remove)
            dataset = dataset.astype(datatype)
            return dataset
        except KeyError:
            logging.warning(f'The dataset does not contain the specified variable key and will be read as object.')
            dataset = pd.read_csv(filepath, sep=separator, dtype=object, na_values=missing)
            return dataset


def read_excel(filepath):
    """
    Reads a csv file and return it as pandas DataFrame object.

    :param str filepath: specify the filepath of the Excel file that is to be read
    :return pandas DataFrame
    """
    # read Excel file
    logging.info(f'Reading excel file from {filepath}')
    return pd.read_excel(filepath)


def read_json(filepath):
    """
    Read a JSON file

    :param str filepath: specify the filepath of the json file that is to be read
    :return any
    """
    # read JSON file
    logging.info(f'Reading JSON file from {filepath}')
    with open(f'{filepath}', 'r') as filepath:
        meta_data = json.load(filepath)
    return meta_data


def save_csv(data_frame, filename):
    """
    Save a pandas.DataFrame as csv.

    :param pandasDataFrame data_frame: dataframe that is to be saved
    :param str filename: filename including the path in which it is to be saved
    """
    # save data
    data_frame.to_csv(filename, index=False)
    logging.info(f'Saving processed DataSet as {filename}')


def save_json(data, filename):
    """
    Save data as JSON file

    :param data: data that is to be saved
    :param str filename: filename including the path in which it is to be saved
    """
    # save json
    with open(filename, 'w') as filepath:
        json.dump(data, filepath, indent=4)

    logging.info(f'Saved JSON file as {filename}')
