import copy
import logging
import os
import sdv
import uuid
import warnings

import numpy as np
import pandas as pd

# DP-CGAN was built in a different SDV version and should only be imported when used
# from dp_cgans import DP_CGAN
from sdv.lite import TabularPreset

# private modules
from src import data_processing_general, file_handling

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SDVSynthetic:
    """
    Aims to generate synthetic data using the Synthetic Data Vault package
    """

    def __init__(self, table_format='SDV_single_table', directory=None, data=None, meta_data=None, model_path=None,
                 settings_path=None, sdv_defaults=False):
        """
        Initialises the SDVSynthetic class

        :param str table_format: specify the type of data to synthesize, single table, repeated measures and so forth
        :param str directory: allows to manually specify the directory that will be searched
        :param any data: allows to manually specify the data (path) rather than from metadata or selection
        :param str meta_data: allows to manually specify the JSON metadata (path) rather than by (auto-) selection
        :param str model_path: allows to manually specify a previously fitted model to be used for sampling
        :param any data: allows to manually specify the path to the settings by searching selection
        :param bool sdv_defaults: specify whether to use SDV default parameter settings
        """
        # table format has to be defined so that appropriate modelling can be selected
        self.TableFormat = table_format

        # initialise paths
        self.Directory = directory
        self.ModelDataPath = model_path
        self.SettingsPath = settings_path

        # retrieve data, model and the directory
        self.OriginalDataPath, self.MetaDataPath, self.Directory, self.SettingsPath = \
            file_handling.find_items(self.Directory, data, meta_data, model_path)

        # data is to be read using functions
        if data is None:
            self.OriginalData = file_handling.read_csv(self.OriginalDataPath, 'float', ' ', ',')
        elif isinstance(data, pd.DataFrame):
            self.OriginalData = data
        else:
            logging.info(f'Supplied data set is not of type pandas.DataFrame, reading data from path:\n'
                         f'{self.OriginalDataPath}')
            self.OriginalData = file_handling.read_csv(self.OriginalDataPath, 'float', ' ', ',')
        self.MetaData = file_handling.read_json(self.MetaDataPath)
        self.Settings = file_handling.read_json(self.SettingsPath)

        self.Model = None
        self.ModelName = None
        self.SyntheticData = {}
        self.Constraints = {}

        # ensure paths are present
        if os.path.exists(os.path.join(self.Directory, 'models')) is False:
            os.mkdir(os.path.join(self.Directory, 'models'))
        self.ModelOutputPath = os.path.join(self.Directory, 'models')

        if os.path.exists(os.path.join(self.Directory, 'synthetic')) is False:
            os.mkdir(os.path.join(self.Directory, 'synthetic'))
        self.SyntheticOutputPath = os.path.join(self.Directory, 'synthetic')

        logging.info(f'Initialising SDV variables.')

        # loaded models
        self.LoadedModels = {}

        # modelling techniques available for single table data in November 2022
        # data loading has a strong dependency on this list, alterations should be performed with caution
        self.SingleTableStrategies = np.array(['FAST_ML', 'GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE', 'DP-CGAN'])

        # available table formats
        self.ModellingStrategies = {'SDV_single_table': self.SingleTableStrategies}

        # save a copy of the original data
        self.OriginalDataCopy = self.OriginalData.copy()
        self.MetaDataCopy = copy.deepcopy(self.MetaData)

        # process data
        self.OriginalDataImputed = data_processing_general.impute_mice(self.OriginalData)
        self.OriginalData = self.OriginalDataImputed.copy()

        # specifies whether to use SDV default parameters
        self.SDVDefaults = sdv_defaults

    def data_find_model(self, model_identifier='.pk1'):
        """
        Find models that are located in the class directory using an identification substring or
        pass to modelling when no models are found

        :param str model_identifier: specify string that is used to identify model files by filename
        """
        # only perform search when path was not specified
        if self.ModelDataPath is None:
            logging.info(f'Searching directory {self.ModelOutputPath} for files containing {model_identifier}')
            available_models = {}
            model_number = 0

            # loop through directory and add name to available data dictionary when it contains the identifier
            for file in os.listdir(self.ModelOutputPath):
                if model_identifier in file:
                    model_number += 1
                    available_models.update({model_number: os.path.join(self.ModelOutputPath, file)})
                else:
                    continue

            # allow to specify what dataset is to be used in case multiple are found
            if model_number > 1:
                selected_data = input(f'The following files containing identifier "{model_identifier}" were '
                                      f'found\n{available_models}\n'
                                      f'Please input the number(s) of the file(s) that is or are to be used. '
                                      f'Separate multiple numbers using a comma (,).\n'
                                      f'Input "all" to select all available models\n'
                                      f'Input "train" to select a different model\n')
                if selected_data in 'all':
                    self.ModelDataPath = list(available_models.values())
                elif selected_data in 'train':
                    self.model_selection()
                else:
                    self.ModelDataPath = []

                    for model in selected_data.split(','):
                        self.ModelDataPath.append(available_models[int(model)])

                if selected_data not in 'train':
                    # read models
                    self.LoadedModels = self._data_read_sdv_model(model_identifier)

            # assign filename when only one is found
            elif model_number == 1:
                # allow the user to train a model despite that one was found
                train = input(
                    f'The following model was found {[available_models[1]]}\n'
                    f'Would you like use this model? Enter yes or no\n')

                if train in 'no':
                    self.model_selection()

                else:
                    self.ModelDataPath = [available_models[1]]

                    # read model
                    self.LoadedModels = self._data_read_sdv_model(model_identifier)

            # refer to data modelling when no models are found
            elif model_number == 0:
                warnings.warn(f'No files containing {model_identifier} where found in {self.ModelOutputPath}\n'
                              f'To synthesise data, ensure that it is modelled using the model_data function.')
                train = input(f'Would you like to model the data? Enter yes or no\n')
                if train in 'yes':
                    self.model_selection()
        else:
            warnings.warn(f'A model was already specified (i.e., {self.ModelDataPath}), '
                          f'searching for a model is unnecessary.\n'
                          f'Use the model_sample function to generate synthetic data using the specified model')

    def data_read(self, dataset=None, missing=' ', separator=','):
        """
        Reads a csv file and stores it as pandas DataFrame object.

        :param pandas.DataFrame dataset: directly pass a pandas Dataframe and bypass actual reading
        :param str missing: allows to specify how missing data is formatted in read data
        :param str separator: allows to specify how data is separated
        :return pandas DataFrame: read CSV data
        """
        if self.OriginalDataPath is None:
            try:
                self.OriginalDataPath = self.MetaData['path']
            except KeyError:
                file_handling.find_file(self.Directory, '.csv')
        else:
            # safety check that warns if the specified data path does not match metadata data path
            try:
                if self.OriginalDataPath is not self.MetaData['path']:
                    warnings.warn(f'The specified data path ({self.OriginalDataPath}, does not match with the data path'
                                  f'specified in the selected metadata ({self.MetaData["path"]}); '
                                  f'Please proceed with caution')
            except KeyError:
                pass

        if dataset is None:
            # read dataset using specified delimiter
            original_data = file_handling.read_csv(self.OriginalDataPath, 'float', missing, separator)
        else:
            original_data = dataset

        return original_data

    def _data_read_sdv_model(self, model_identifier, data_extension='.csv', model_identifier_substring='_model_'):
        """
        Read / load SDV models by model name as specified in ModelDataPath, is model name sensitive

        :param str model_identifier: specify string that is used to identify model files by filename
        :param str data_extension: specify the file format of the data
        :param str model_identifier_substring: specify sub string that is used to identify model files by filename
        :return: dict of loaded models
        """
        # create an empty dictionary so that it can be updated with all models that have to be loaded
        loading_strategies = {}

        for model_path in self.ModelDataPath:
            logging.info(f'Reading SDV model from {model_path}')

            # determine specific model name, i.e., remove data name and other descriptions
            original_name = self.OriginalDataPath[self.OriginalDataPath.rfind(os.path.sep):]
            data_model_name = str.replace(f"{self.ModelOutputPath}{original_name}",
                                          data_extension, model_identifier_substring)
            # remove everything but the model name
            model_name = str.replace(model_path, data_model_name, '')
            model_name = str.replace(model_name, model_identifier, '')

            # clunky but functional
            if self.TableFormat == 'SDV_single_table':
                if self.ModellingStrategies[self.TableFormat][0] in model_name:
                    loading_strategies.update({'FAST_ML': TabularPreset.load(model_path)})
                elif self.ModellingStrategies[self.TableFormat][1] in model_name:
                    loading_strategies.update({'GaussianCopula': sdv.tabular.GaussianCopula.load(model_path)})
                elif self.ModellingStrategies[self.TableFormat][2] in model_name:
                    loading_strategies.update({'CTGAN': sdv.tabular.CTGAN.load(model_path)})
                elif self.ModellingStrategies[self.TableFormat][3] in model_name:
                    loading_strategies.update({'CopulaGAN': sdv.tabular.CopulaGAN.load(model_path)})
                elif self.ModellingStrategies[self.TableFormat][4] in model_name:
                    loading_strategies.update({'TVAE': sdv.tabular.TVAE.load(model_path)})
                # elif self.ModellingStrategies[self.TableFormat][5] in model_name:
                # loading_strategies.update({'DP-CGAN': DP_CGAN.load(model_path)})
                else:
                    logging.critical(f'The model name {model_name} was not found for {self.TableFormat}.\n'
                                     f'Please ensure that it is included in the modelling and loading strategies '
                                     f'for it to be loaded.')
            else:
                logging.critical(f'The selected table format of {self.TableFormat}, is included in model loading.\n'
                                 f'Carefully consider the table format you are using or update the loading function')

        logging.debug(f'loaded the following models:\n{loading_strategies}')
        return loading_strategies

    def model_adjust(self, epochs=None, batch_size=None, log_frequency=None, embedding_dim=None,
                     generator_dim=None, discriminator_dim=None, generator_lr=None, generator_decay=None,
                     discriminator_lr=None, discriminator_decay=None, discriminator_steps=None, verbose=None,
                     compress_dims=None, decompress_dims=None, l2scale=None, loss_factor=None,
                     cuda=None, private=None, field_distributions=None, default_distribution=None):
        """
        Adjust a model parameter and initialise the model for the specified class (SDV) table format e.g., single_table

        For a description of the function of every parameter please see SDV documentation at:
        https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers
        """
        logging.info(f'The following models were selected: {self.LoadedModels.keys()}')
        for model_name in self.LoadedModels.keys():
            # ensure that no space is interfering with the process
            model_name = model_name.strip().replace("'", "")

            # clunky but functional
            if self.TableFormat == 'SDV_single_table':
                if model_name == self.ModellingStrategies[self.TableFormat][0]:
                    warnings.warn(f'Model: {self.ModellingStrategies[self.TableFormat][0]} has fixed parameters.')
                elif model_name == self.ModellingStrategies[self.TableFormat][1]:
                    self.LoadedModels.update({'GaussianCopula': _gaussian_copula_initiate(table_metadata=self.MetaData,
                                                                                          field_distributions=
                                                                                          field_distributions,
                                                                                          default_distribution=
                                                                                          default_distribution)})
                elif model_name == self.ModellingStrategies[self.TableFormat][2]:
                    self.LoadedModels.update({'CTGAN': _ctgan_initiate(table_metadata=self.MetaData, epochs=epochs,
                                                                       batch_size=batch_size,
                                                                       log_frequency=log_frequency,
                                                                       embedding_dim=embedding_dim,
                                                                       generator_dim=generator_dim,
                                                                       discriminator_dim=discriminator_dim,
                                                                       generator_lr=generator_lr,
                                                                       generator_decay=generator_decay,
                                                                       discriminator_lr=discriminator_lr,
                                                                       discriminator_decay=discriminator_decay,
                                                                       discriminator_steps=discriminator_steps,
                                                                       verbose=verbose,
                                                                       cuda=cuda)})
                elif model_name == self.ModellingStrategies[self.TableFormat][3]:
                    self.LoadedModels.update({'CopulaGAN': _copula_gan_initiate(table_metadata=self.MetaData,
                                                                                epochs=epochs, batch_size=batch_size,
                                                                                log_frequency=log_frequency,
                                                                                embedding_dim=embedding_dim,
                                                                                generator_dim=generator_dim,
                                                                                discriminator_dim=discriminator_dim,
                                                                                generator_lr=generator_lr,
                                                                                generator_decay=generator_decay,
                                                                                discriminator_lr=discriminator_lr,
                                                                                discriminator_decay=discriminator_decay,
                                                                                discriminator_steps=discriminator_steps,
                                                                                verbose=verbose,
                                                                                cuda=cuda,
                                                                                field_distributions=field_distributions,
                                                                                default_distribution=default_distribution)})
                elif model_name == self.ModellingStrategies[self.TableFormat][4]:
                    self.LoadedModels.update({'TVAE': _tvae_initiate(table_metadata=self.MetaData, epochs=epochs,
                                                                     batch_size=batch_size, embedding_dim=embedding_dim,
                                                                     compress_dims=compress_dims,
                                                                     decompress_dims=decompress_dims, l2scale=l2scale,
                                                                     loss_factor=loss_factor, cuda=cuda)})
                elif model_name == self.ModellingStrategies[self.TableFormat][5]:
                    self.LoadedModels.update({'DP-CGAN': _dp_cgan_initiate(table_metadata=self.MetaData, epochs=epochs,
                                                                           batch_size=batch_size,
                                                                           log_frequency=log_frequency,
                                                                           embedding_dim=embedding_dim,
                                                                           generator_dim=generator_dim,
                                                                           discriminator_dim=discriminator_dim,
                                                                           generator_lr=generator_lr,
                                                                           generator_decay=generator_decay,
                                                                           discriminator_lr=discriminator_lr,
                                                                           discriminator_decay=discriminator_decay,
                                                                           discriminator_steps=discriminator_steps,
                                                                           verbose=verbose,
                                                                           cuda=cuda, private=private)})
                else:
                    logging.critical(f'The model name {model_name} was not found for {self.TableFormat}.\n'
                                     f'Please ensure that it is included in the modelling and loading strategies '
                                     f'for it to be loaded.')
            else:
                logging.critical(
                    f'The selected table format of {self.TableFormat}, is included in model loading.\n'
                    f'Carefully consider the table format you are using or update the loading function')

        logging.debug(f'Prepared the following models for modelling:\n{self.LoadedModels.keys()}')

    def model_data(self, model_strategy=None, save=True, name_appendices=''):
        """
        Model the data using specified modelling strategy

        :param str model_strategy: name of SDV model to be used
        :param boolean save: specify whether model should be saved
        :param str name_appendices: optionally specify a name that is to be added to the filename
        """
        # if a specific model is given model the data with that specific model
        if model_strategy is not None:
            # ensure that a supported model is selected
            model_strategy = self._model_support(model_strategy)

            self.LoadedModels = [model_strategy]

        for model in self.LoadedModels:
            logging.info(f'Modelling data using {model}')

            # assign model from available models
            # noinspection PyTypeChecker
            sdv_model = self.LoadedModels[model]

            # fit the model
            logging.debug(f'Fitting {sdv_model} to {self.OriginalData.head}')
            sdv_model.fit(self.OriginalData)

            # save the model for later use if so desired
            if save:
                original_name = self.OriginalDataPath[self.OriginalDataPath.rfind(os.path.sep):]
                model_filename = f"{self.ModelOutputPath}" \
                                 f"{original_name.replace('.csv', f'_{name_appendices}_model_{model}.pk1')}"
                sdv_model.save(model_filename)
                logging.info(f'Saved model at {model_filename}')

    def model_sample(self, number_of_samples, identification=True, save=True, return_name=False, name_appendices=''):
        """
        Synthesise a sample using the loaded models

        :param int number_of_samples: specify the number of samples to generate
        :param bool identification: specify whether to add an uuid to the data name
        :param bool save: specify whether to save the synthesised data to a CSV file
        :param bool return_name: specify whether to return the file_identification
        :param str name_appendices: optionally provide string that is to be added at the end of the synthetic data file
        """
        file_identification = None

        # generate synthetic data
        for model in self.LoadedModels.keys():
            logging.info(f'Sampling {number_of_samples} from {model} model')
            synthetic_data = self.LoadedModels[model].sample(num_rows=number_of_samples)

            if identification:
                identifier = uuid.uuid1()
            else:
                identifier = ''

            # produce appropriate file addition
            file_identification = f'_{model}_{number_of_samples}{name_appendices}^{identifier}^'

            # allow direct export to csv
            if save:
                # create new filename with same path and addition at end of the filename
                original_name = self.OriginalDataPath[self.OriginalDataPath.rfind(os.path.sep):]
                processed_filename = f"{self.SyntheticOutputPath}" \
                                     f"{original_name.replace('.csv', f'{file_identification}.csv')}"
                file_handling.save_csv(synthetic_data, processed_filename)

            # update dictionary of synthetic datasets
            file_identification = self.OriginalDataPath.replace('.csv', f'{file_identification}')
            logging.info(f'Updating the synthetic data dictionary with synthesised set:\n{file_identification}')
            self.SyntheticData.update({file_identification: synthetic_data})

        if return_name:
            return file_identification

    def model_selection(self, model=None):
        """
        Select a model and initialise the metadata  for the specified class (SDV) table format e.g., single_table

        :param list model: specify the name of the model
        """

        model_strategies = {}
        selected_models = None

        if self.ModelDataPath is None:
            if model is None:
                selected_models = input(f'For table format {self.TableFormat} the following models are available:\n'
                                        f'{self.ModellingStrategies[self.TableFormat]}\n'
                                        f'Please select the models that you would like to select for training.\n'
                                        f'When selecting multiple models please separate them using a comma\n'
                                        f'Note that this is case sensitive and should match the format in this message.'
                                        f'\n').split(',')

            else:
                selected_models = model

        logging.info(f'The following models were selected: {selected_models}')
        if self.ModelDataPath is None:
            for model_name in selected_models:
                # ensure that no space is interfering with the process
                model_name = model_name.strip().replace("'", "")

                # clunky but functional
                if self.TableFormat == 'SDV_single_table':
                    if model_name == self.ModellingStrategies[self.TableFormat][0]:
                        model_strategies.update({'FAST_ML': TabularPreset('FAST_ML', metadata=self.MetaData)})
                    elif model_name == self.ModellingStrategies[self.TableFormat][1]:
                        model_strategies.update({'GaussianCopula': sdv.tabular.GaussianCopula(
                            table_metadata=self.MetaData)})
                    elif model_name == self.ModellingStrategies[self.TableFormat][2]:
                        model_strategies.update({'CTGAN': sdv.tabular.CTGAN(table_metadata=self.MetaData,
                                                                            verbose=True)})
                    elif model_name == self.ModellingStrategies[self.TableFormat][3]:
                        model_strategies.update({'CopulaGAN': sdv.tabular.CopulaGAN(table_metadata=self.MetaData,
                                                                                    verbose=True)})
                    elif model_name == self.ModellingStrategies[self.TableFormat][4]:
                        model_strategies.update({'TVAE': sdv.tabular.TVAE(table_metadata=self.MetaData)})
                    elif model_name == self.ModellingStrategies[self.TableFormat][5]:
                        model_strategies.update({'DP-CGAN': _dp_cgan_initiate(table_metadata=self.MetaData, epochs=None,
                                                                              batch_size=None, log_frequency=None,
                                                                              embedding_dim=None,
                                                                              generator_dim=None,
                                                                              discriminator_dim=None, generator_lr=None,
                                                                              generator_decay=None,
                                                                              discriminator_lr=None,
                                                                              discriminator_decay=None,
                                                                              discriminator_steps=None, verbose=None,
                                                                              cuda=None, private=None)})
                    else:
                        logging.critical(f'The model name {model_name} was not found for {self.TableFormat}.\n'
                                         f'Please ensure that it is included in the modelling and loading strategies '
                                         f'for it to be loaded.')
                else:
                    logging.critical(
                        f'The selected table format of {self.TableFormat}, is included in model loading.\n'
                        f'Carefully consider the table format you are using or update the loading function')

            logging.debug(f'Prepared the following models for modelling:\n{model_strategies}')
            self.LoadedModels = model_strategies

            if self.SDVDefaults is False:
                self.model_adjust()

        else:
            warnings.warn(
                f'A model was already specified (i.e., {self.ModelDataPath}), selecting a model is unnecessary.\n'
                f'Use the model_sample function to generate synthetic data using the specified model.')
            train = input(f'Would you like to train a different model? Enter yes or no')
            if train in 'yes':
                self.ModelDataPath = None
                self.model_selection()

    def _model_support(self, model_strategy):
        """
        Check whether the specified model is supported and helps the user to select one that is supported

        :param str model_strategy: name of SDV model to be used
        :return: Supported model strategy name
        """
        logging.info(f'Checking whether {model_strategy} is available')

        # check whether the specified model is in the loaded strategies and if not help user to pick one
        while model_strategy not in self.ModellingStrategies[self.TableFormat].keys():
            warnings.warn(f'{model_strategy} is not a supported strategy.\n'
                          f'Ensure that the SDV class is set up with the correct table format and '
                          f'that a therewith supported model is selected.\n'
                          f'Current table format is {self.TableFormat}\n'
                          f'Strategies supported with this format are '
                          f'{self.ModellingStrategies[self.TableFormat].keys()}')
            model_strategy = input(f'Please provide the name of a model that you would like to run and is supported\n')

        return model_strategy


def _gaussian_copula_initiate(table_metadata, field_distributions, default_distribution):
    """
    Returns a GaussianCopula model with pre-specified parameters as described in the SDV library.
    Tuning is handled separately in a function to ensure that no parameters are muted when not specified by the user

    The function creates more freedom in changing the parameters whilst mitigating the need to know the default values
    Parameter defaults were taken from the SDV repository on GitHub last changed in commit 0af85b8
    Link: https://github.com/sdv-dev/SDV/blob/53ca6beca276de0aee91311f433fc84d2a15d2c0/sdv/tabular/copulas.py
    Documentation at time of composing this function, link: https://sdv.dev/SDV/user_guides/single_table/gaussian_copula
    """
    # set certain defaults that differ from library defaults and avoid mutable parameters for Copula
    if field_distributions is None:
        field_distributions = None
    if default_distribution is None:
        default_distribution = None

    return sdv.tabular.GaussianCopula(table_metadata=table_metadata, field_distributions=field_distributions,
                                      default_distribution=default_distribution)


# noinspection DuplicatedCode
def _ctgan_initiate(table_metadata, epochs, batch_size, log_frequency, embedding_dim,
                    generator_dim, discriminator_dim, generator_lr, generator_decay,
                    discriminator_lr, discriminator_decay, discriminator_steps, verbose,
                    cuda):
    """
    Returns a CGAN model with pre-specified hyperparameters as described in the SDV library.
    Tuning is handled separately in a function to ensure that no parameters are muted when not specified by the user
    For example, if only epochs are changed, the rest of the parameters are sustained

    The function creates more freedom in changing the parameters whilst mitigating the need to know the default values
    Parameter defaults were taken from the SDV repository on GitHub last changed in commit b09131f
    Link: https://github.com/sdv-dev/SDV/blob/53ca6beca276de0aee91311f433fc84d2a15d2c0/sdv/tabular/ctgan.py
    Documentation at time of composing this function, link: https://sdv.dev/SDV/user_guides/single_table/ctgan.html
    """
    # set certain defaults that differ from library defaults and avoid mutable parameters
    if epochs is None:
        epochs = 300
    if batch_size is None:
        batch_size = 500
    if log_frequency is None:
        log_frequency = True
    if embedding_dim is None:
        embedding_dim = 128
    if generator_dim is None:
        generator_dim = (256, 256)
    if discriminator_dim is None:
        discriminator_dim = (256, 256)
    if generator_lr is None:
        generator_lr = 2E-4
    if generator_decay is None:
        generator_decay = 1E-6
    if discriminator_lr is None:
        discriminator_lr = 2E-4
    if discriminator_decay is None:
        discriminator_decay = 1E-6
    if discriminator_steps is None:
        discriminator_steps = 1
    if verbose is None:
        verbose = True
    if cuda is None:
        cuda = True

    return sdv.tabular.CTGAN(table_metadata=table_metadata, epochs=epochs, batch_size=batch_size,
                             log_frequency=log_frequency, embedding_dim=embedding_dim, generator_dim=generator_dim,
                             discriminator_dim=discriminator_dim, generator_lr=generator_lr,
                             generator_decay=generator_decay, discriminator_lr=discriminator_lr,
                             discriminator_decay=discriminator_decay, discriminator_steps=discriminator_steps,
                             verbose=verbose, cuda=cuda)


# noinspection DuplicatedCode
def _copula_gan_initiate(table_metadata, epochs, batch_size, log_frequency, embedding_dim,
                         generator_dim, discriminator_dim, generator_lr, generator_decay,
                         discriminator_lr, discriminator_decay, discriminator_steps, verbose,
                         cuda, field_distributions, default_distribution):
    """
    Returns a CopulaGAN model with pre-specified (hyper)parameters as described in the SDV library.
    Tuning is handled separately in a function to ensure that no parameters are muted when not specified by the user
    For example, if only epochs are changed, the rest of the parameters are sustained

    The function creates more freedom in changing the parameters whilst mitigating the need to know the default values
    Parameter defaults were taken from the SDV repository on GitHub last changed in commit 0af85b8
    Link: https://github.com/sdv-dev/SDV/blob/53ca6beca276de0aee91311f433fc84d2a15d2c0/sdv/tabular/copulagan.py
    Documentation at time of composing this function, link: https://sdv.dev/SDV/user_guides/single_table/copulagan
    """
    # set certain defaults that differ from library defaults and avoid mutable parameters for GAN
    if epochs is None:
        epochs = 300
    if batch_size is None:
        batch_size = 500
    if log_frequency is None:
        log_frequency = True
    if embedding_dim is None:
        embedding_dim = 128
    if generator_dim is None:
        generator_dim = (256, 256)
    if discriminator_dim is None:
        discriminator_dim = (256, 256)
    if generator_lr is None:
        generator_lr = 2E-4
    if generator_decay is None:
        generator_decay = 1E-6
    if discriminator_lr is None:
        discriminator_lr = 2E-4
    if discriminator_decay is None:
        discriminator_decay = 1E-6
    if discriminator_steps is None:
        discriminator_steps = 1
    if verbose is None:
        verbose = True
    if cuda is None:
        cuda = True

    # set certain defaults that differ from library defaults and avoid mutable parameters for Copula
    if field_distributions is None:
        field_distributions = None
    if default_distribution is None:
        default_distribution = None

    return sdv.tabular.CopulaGAN(table_metadata=table_metadata, epochs=epochs, batch_size=batch_size,
                                 log_frequency=log_frequency, embedding_dim=embedding_dim, generator_dim=generator_dim,
                                 discriminator_dim=discriminator_dim, generator_lr=generator_lr,
                                 generator_decay=generator_decay, discriminator_lr=discriminator_lr,
                                 discriminator_decay=discriminator_decay, discriminator_steps=discriminator_steps,
                                 verbose=verbose, cuda=cuda,
                                 field_distributions=field_distributions, default_distribution=default_distribution)


def _tvae_initiate(table_metadata, epochs, batch_size, embedding_dim,
                   compress_dims, decompress_dims, l2scale, loss_factor, cuda):
    """
    Returns a CGAN model with pre-specified hyperparameters as described in the SDV library.
    Tuning is handled separately in a function to ensure that no parameters are muted when not specified by the user
    For example, if only epochs are changed, the rest of the parameters are sustained

    The function creates more freedom in changing the parameters whilst mitigating the need to know the default values
    Parameter defaults were taken from the SDV repository on GitHub last changed in commit b09131f
    Link: https://github.com/sdv-dev/SDV/blob/53ca6beca276de0aee91311f433fc84d2a15d2c0/sdv/tabular/ctgan.py
    Documentation at time of composing this function, link: https://sdv.dev/SDV/user_guides/single_table/tvae
    """
    # set certain defaults that differ from library defaults and avoid mutable parameters
    if epochs is None:
        epochs = 300
    if batch_size is None:
        batch_size = 500
    if embedding_dim is None:
        embedding_dim = 128
    if compress_dims is None:
        compress_dims = (128, 128)
    if decompress_dims is None:
        decompress_dims = (128, 128)
    if l2scale is None:
        l2scale = 1E-5
    if loss_factor is None:
        loss_factor = 2
    if cuda is None:
        cuda = True

    return sdv.tabular.TVAE(table_metadata=table_metadata, epochs=epochs, batch_size=batch_size,
                            embedding_dim=embedding_dim, compress_dims=compress_dims,
                            decompress_dims=decompress_dims, l2scale=l2scale, loss_factor=loss_factor, cuda=cuda)


# noinspection PyUnusedLocal
def _dp_cgan_initiate(table_metadata, epochs, batch_size, log_frequency, embedding_dim,
                      generator_dim, discriminator_dim, generator_lr, generator_decay,
                      discriminator_lr, discriminator_decay, discriminator_steps, verbose,
                      cuda, private):
    """
    Returns a DP-CGAN model with pre-specified hyperparameters as described by Chang Sun.
    Initiation was separated in a function to ensure that practically all of Chang's defined parameters are used
    apart from parameters that the user sets e.g., if epochs are changed, the rest of the parameters are sustained.

    Paper DOI: 10.1016/j.jbi.2023.104404
    Link at time of composition of this function: https://doi.org/10.1016/j.jbi.2023.104404
    """
    # set certain defaults that differ from library defaults and avoid mutable parameters
    if epochs is None:
        epochs = 2000
    if batch_size is None:
        batch_size = 100
    if log_frequency is None:
        log_frequency = True
    if embedding_dim is None:
        embedding_dim = 128
    if generator_dim is None:
        generator_dim = (128, 128, 128)
    if discriminator_dim is None:
        discriminator_dim = (128, 128, 128)
    if generator_lr is None:
        generator_lr = 1E-4
    if generator_decay is None:
        generator_decay = 1E-6
    if discriminator_lr is None:
        discriminator_lr = 1E-4
    if discriminator_decay is None:
        discriminator_decay = 1E-6
    if discriminator_steps is None:
        discriminator_steps = 1
    if verbose is None:
        verbose = True
    if cuda is None:
        cuda = True
    if private is None:
        private = False

    return None
    # DP-CGAN was built in a different SDV version and should only be imported when used
    """return DP_CGAN(table_metadata=table_metadata, epochs=epochs, batch_size=batch_size,
                   log_frequency=log_frequency, embedding_dim=embedding_dim, generator_dim=generator_dim,
                   discriminator_dim=discriminator_dim, generator_lr=generator_lr, generator_decay=generator_decay,
                   discriminator_lr=discriminator_lr, discriminator_decay=discriminator_decay,
                   discriminator_steps=discriminator_steps, verbose=verbose, cuda=cuda, private=private)"""
