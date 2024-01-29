import logging
import warnings

import numpy as np
import pandas as pd

# private modules
from src import file_handling, generator_faker

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format=f'%(asctime)s - %(levelname)s - %(message)s')

available_formats = ['single_table']


class UnProcessedData:
    """
    Aims to preprocess the provided dataset and generate an SDV metadata dictionary
    """

    def __init__(self, data_path=None, codebook_path=None, settings_path=None):
        """
        Initialises the UnProcessedData class

        :param str data_path: specify the path to the csv data that is to be processed, will prompt if not provided
        :param str codebook_path:specify the path to the codebook that is to be used for processing,
         will prompt if not provided, when unavailable will only use corrections
        :param str settings_path: specify the path to the settings that are to be used for processing,
         will prompt if not provided
        """
        # read data
        if isinstance(data_path, str) is False:
            self.DataSetPath = input(f'Please specify the filename and path of the .csv dataset\n')
        else:
            self.DataSetPath = data_path

        # read codebook
        if isinstance(codebook_path, str) is False:
            self.CodeBookPath = input(f'Please specify the filename and path of the Excel codebook.\n'
                                      f'Press enter if no codebook is available')
        else:
            self.CodeBookPath = codebook_path

        if len(self.CodeBookPath) > 0:
            self.CodeBook = file_handling.read_excel(self.CodeBookPath)
        else:
            input(f'\nEnsure that all variables are correctly specified in the settings under "datatype"_corrections\n'
                  f'Press any key to continue')
            self.CodeBook = None

        # read settings
        if isinstance(settings_path, str) is False:
            self.SettingsPath = input(f'Please specify the filename and path of the settings\n')
        else:
            self.SettingsPath = settings_path
        self.Settings = file_handling.read_json(self.SettingsPath)
        self.PreProcessingSettings = self.Settings['Pre-processing']

        # read data with specified missing
        self.DataSet = file_handling.read_csv(self.DataSetPath, 'object',
                                              missing=self.PreProcessingSettings['General_information'][
                                                  'missing_value'])

        self.CleanDataSetPath = None
        self.MetaData = None

        # check whether format is available
        if self.PreProcessingSettings["General_information"]["format"] in available_formats:
            self.MetaDataFormat = f'SDV_{self.PreProcessingSettings["General_information"]["format"]}'
        else:
            logging.critical(f'Table format {self.PreProcessingSettings["General_information"]["format"]} is not '
                             f'supported.\nCurrently supported formats are {available_formats}')

        logging.debug(f'Initialising dataset metadata trigger words and corrections.')

        #  specify the name of the identifier in the data
        if len(self.PreProcessingSettings['General_information']['identifier']) > 0:
            self.Identifier = self.PreProcessingSettings['General_information']['identifier']
            logging.debug(f'Identifier used: {self.Identifier}')
        else:
            self.Identifier = None

        # categorical variables are integers but are stored separately as well for synthetic data generation
        self.CategoricalVariables = np.array(self.PreProcessingSettings['General_information']['categorical_variables'])
        logging.debug(f'Categorical variables used: {self.CategoricalVariables}')

        # corrections that will be added due to faulty variable names, codebook flaws and other discrepancies
        self.CategoricalCorrections = self.PreProcessingSettings['General_information']['categorical_corrections']
        logging.debug(f'Categorical corrections used: {self.CategoricalCorrections}')

        # booleans have no specified type in codebook and are determined by a special function
        if len(self.PreProcessingSettings['General_information']['boolean_triggers']) > 0:
            self.BooleanTriggers = self.PreProcessingSettings['General_information']['boolean_triggers']
        else:
            self.BooleanTriggers = None
        logging.debug(f'Boolean triggers used: {self.BooleanTriggers}')

        # corrections that will be added due to faulty variable names, codebook flaws and other discrepancies
        self.BooleanCorrections = self.PreProcessingSettings['General_information']['boolean_corrections']
        logging.debug(f'Boolean corrections used: {self.BooleanCorrections}')

        # float trigger words assessed from codebook to be a float
        if len(self.PreProcessingSettings['General_information']['float_triggers']) > 0:
            self.FloatTriggers = self.PreProcessingSettings['General_information']['float_triggers']
        else:
            self.FloatTriggers = None
        logging.debug(f'Float triggers used: {self.FloatTriggers}')

        # corrections that will be added due to faulty variable names, codebook flaws and other discrepancies
        self.FloatCorrections = self.PreProcessingSettings['General_information']['float_corrections']
        logging.debug(f'Float corrections used: {self.FloatCorrections}')

        # integer trigger words assessed from codebook to be a integer
        if len(self.PreProcessingSettings['General_information']['integer_triggers']) > 0:
            self.IntegerTriggers = self.PreProcessingSettings['General_information']['integer_triggers']
        else:
            self.IntegerTriggers = None
        logging.debug(f'Integer triggers used: {self.IntegerTriggers}')

        # corrections that will be added due to faulty variable names, codebook flaws and other discrepancies
        self.IntegerCorrections = self.PreProcessingSettings['General_information']['integer_corrections']
        logging.debug(f'Integer corrections used: {self.IntegerCorrections}')

        # string trigger words retrieved assessed from codebook as too broad to categorise
        if len(self.PreProcessingSettings['General_information']['string_triggers']) > 0:
            self.StringTriggers = self.PreProcessingSettings['General_information']['string_triggers']
        else:
            self.StringTriggers = None
        logging.debug(f'String triggers used: {self.StringTriggers}')

        # corrections that will be added due to faulty variable names, codebook flaws and other discrepancies
        self.StringCorrections = self.PreProcessingSettings['General_information']['string_corrections']
        logging.debug(f'String corrections used: {self.StringCorrections}')

        # available data types
        self.DataTriggers = {'boolean': self.BooleanTriggers,
                             'float64': self.FloatTriggers,
                             'int64': self.IntegerTriggers,
                             'string': self.StringTriggers}
        logging.info(f'Data types specified: {self.DataTriggers.keys()}')

        # available corrections
        self.DataCorrections = {'boolean': self.BooleanCorrections,
                                'float64': self.FloatCorrections,
                                'int64': self.IntegerCorrections,
                                'string': self.StringCorrections}
        logging.info(f'Data corrections specified: {self.DataCorrections.keys()}')

    def clean_data(self, save=True, filename_addition='_clean'):
        """
        Process the data as specified in the settings

        :param boolean save: specify whether the cleaned data should be saved immediately
        :param str filename_addition: text that will be added after the dataset's original name and before '.csv'
        """
        # recode string categories to numeric categories
        if isinstance(self.PreProcessingSettings['Settings']['recode_categories'], list) is True and \
                len(self.PreProcessingSettings['Settings']['recode_categories']) > 0:
            self.DataSet[self.PreProcessingSettings['Settings']['recode_categories']] = \
                self.DataSet[self.PreProcessingSettings['Settings']['recode_categories']].apply(lambda x:
                                                                                                pd.factorize(x)[0])

        # remove free strings
        if 'True' in self.PreProcessingSettings['Settings']['remove_free_text']:
            self._remove_free_text_fields()

        # remove in column string components if specified
        if isinstance(self.PreProcessingSettings['Settings']['remove_in_column_string'], dict) is True and \
                len(self.PreProcessingSettings['Settings']['remove_in_column_string']) > 0:
            for variable, string_to_remove in self.PreProcessingSettings['Settings']['remove_in_column_string'].items():
                self._remove_string_component_manual(variable, string_to_remove)

        # find discrepancies and correct for them
        if 'True' in self.PreProcessingSettings['Settings']['find_codebook_discrepancies']:
            self._find_codebook_data_discrepancies()

        # ensure that booleans are formatted 0 and 1
        if 'True' in self.PreProcessingSettings['Settings']['harmonise_booleans']:
            self._remove_two_in_bool()

        # remove identification and shuffle data
        if 'True' in self.PreProcessingSettings['Settings']['remove_identification']:
            self._remove_identification()

        # remove variables with a certain percentage missing
        if isinstance(self.PreProcessingSettings['Settings']['remove_high_percentage_missing'], list) is True and \
                len(self.PreProcessingSettings['Settings']['remove_high_percentage_missing']) > 0:
            self._remove_missing(self.PreProcessingSettings['Settings']['remove_high_percentage_missing'])

        # allow the selection of certain variables
        if isinstance(self.PreProcessingSettings['Settings']['variables_to_keep'], list) is True and \
                len(self.PreProcessingSettings['Settings']['variables_to_keep']) > 0:
            # retrieve the variables to keep
            desired_variables = self.PreProcessingSettings['Settings']['variables_to_keep']

            # ensure that the only existing variables are selected
            variables_to_keep = [variable_name for variable_name in self.DataSet.columns
                                 if variable_name in desired_variables]
            desired_but_unavailable = [variable_name for variable_name in self.DataSet.columns
                                       if variable_name not in desired_variables]
            if variables_to_keep:
                # select the variables that are to be kept and are existent in the dataset
                dataset_original = self.DataSet
                self.DataSet = self.DataSet[variables_to_keep]

                # use the original order of variables
                self.DataSet = self.DataSet.reindex(dataset_original[variables_to_keep].columns, axis=1)
            else:
                warnings.warn(f'None of the desired variables are found in the dataset, using all variables')

            logging.debug(f'Following variables were desired but unavailable in the selected dataset:'
                          f' {desired_but_unavailable}')

        # save the clean dataset
        if save:
            processed_filename = self.DataSetPath.replace('.csv', f'{filename_addition}.csv')
            self.CleanDataSetPath = processed_filename
            file_handling.save_csv(self.DataSet, processed_filename)

    def format_metadata(self, identifier=None, save=True):
        """
        Prepare a metadata dictionary of all variables and their types.

        :param str identifier: variable name of the identifier variable, e.g., patient identifier
        :param boolean save: specify whether to save to JSON
        """
        logging.info(f'Formatting metadata for {self.MetaDataFormat}')

        field_info = None

        # identifiers are considered special variables
        if identifier is None:
            identifier = self.Identifier

        if isinstance(self.CodeBook, pd.DataFrame):
            # find all numerical types
            booleans = self._find_data_type('boolean')
            floats = self._find_data_type('float64')
            integers = self._find_data_type('int64')
            categorical_variables = self.CategoricalVariables
        else:
            booleans = self.BooleanCorrections
            floats = self.FloatCorrections
            integers = self.IntegerCorrections
            categorical_variables = self.CategoricalCorrections

        # retrieve variable names from dataset
        variable_names = self.DataSet.columns.to_numpy(dtype=str).flatten()

        # prepare SDV metadata_format
        if self.MetaDataFormat == 'SDV_single_table':
            self.MetaData = {"fields": {},
                             "path": self.CleanDataSetPath,
                             "primary_key": identifier}

            if self.Identifier is None or 'hash_id' in self.Identifier:
                self.MetaData.pop('primary_key')

            if self.CleanDataSetPath is None:
                self.MetaData.pop('path')

            # evaluate type per variable and store in metadata dictionary; dict comprehension avoided for cleanliness
            for variable in variable_names:

                if self.Identifier is not None and variable in self.Identifier:
                    field_info = None
                    continue

                if booleans is not None and variable in booleans:
                    field_info = {variable: {'type': 'boolean'}}

                if floats is not None and variable in floats:
                    field_info = {variable: {'type': 'numerical', 'subtype': 'float'}}

                if identifier is not None and variable is identifier:
                    field_info = {variable: {'type': 'id', 'subtype': 'integer'}}

                if integers is not None and variable in integers:
                    field_info = {variable: {'type': 'numerical', 'subtype': 'integer'}}

                if categorical_variables is not None and variable in categorical_variables:
                    field_info = {variable: {'type': 'categorical'}}

                self.MetaData['fields'].update(field_info)
                logging.debug(f'{variable} inserted into metadata dictionary as {field_info}')

        else:
            warnings.warn(f'Metadata format not supported, no metadata was generated.\n'
                          f'Supported formats: {available_formats}')

        logging.info(f'Metadata formatted as:\n {self.MetaData}')

        # save metadata to JSON file for post-hoc retrieval
        if save and self.MetaData is not None:
            file_handling.save_json(self.MetaData, f'{self.DataSetPath.replace(".csv", "")}_'
                                                   f'{self.MetaDataFormat}_metadata.json')

    def _find_codebook_data_discrepancies(self, compensate_corrections=True, header_correction=True,
                                          underscore_correction=True):
        """
        Determines what variables are in the dataset but are incorrectly or not described in the codebook

        :param boolean compensate_corrections: specify whether manual corrections should be taken into account
        :param boolean header_correction: specify whether discrepancies with _header_ values should be corrected
        :param boolean underscore_correction: specify whether discrepancies with double underscore should be corrected
        :return: list of variable names not found in the codebook but are present in the dataset
        """
        if isinstance(self.CodeBook, pd.DataFrame):
            # retrieve variables listed in the codebook
            variables_codebook = self.CodeBook.iloc[:,
                                 self.PreProcessingSettings['General_information']
                                 ['codebook_variable_column']].to_numpy(dtype=str).flatten()
            logging.debug(f'Variables present in codebook: {variables_codebook}')
        else:
            variables_codebook = np.array([])

        # retrieve variables actually available in dataset
        variables_data = self.DataSet.columns.to_numpy(dtype=str).flatten()
        logging.debug(f'Variables present in dataset: {variables_data}')

        if header_correction:
            variables_data = self._remove_variable_by_content(variables_data, '_header_')

        # determine difference
        discrepancies = np.setdiff1d(variables_data, variables_codebook)

        # when discrepancy is manually corrected for further correction is not necessary
        if compensate_corrections:
            for data_type in self.DataCorrections.keys():
                if self.DataCorrections[data_type] is not None:
                    manual_corrections = self.DataCorrections[data_type]

                    for discrepancy in discrepancies:
                        if discrepancy in manual_corrections:
                            discrepancies = discrepancies[discrepancies != discrepancy]
                        else:
                            continue

        # certain variables contain double underscore rather than single and can easily be corrected for
        if underscore_correction:
            discrepancies = self._remove_double_underscore(variables_codebook, discrepancies)
        logging.debug(f'Codebook discrepancies that were found are: {discrepancies}')

        return discrepancies

    def _find_data_type(self, data_type):
        """
        Determine what variable names are specified in the codebook as being of a specific data type.

        :param str data_type: choose the data type you wish to find in the dataset. Must match DataTypes keys
        :return: list of variable names
        """
        if isinstance(self.CodeBook, pd.DataFrame):
            # booleans are not documented as such in codebook and require the answer options column
            if data_type == 'boolean':
                # boolean can be determined using the available answers
                variables_names = self._find_number_of_answers(data_type, np.array([]))
            else:
                # column referring to the answering style, e.g., number, else, occupation, and so forth
                style_definitions = self.CodeBook[['string']].to_numpy(dtype=str).flatten()

                # column that refers to all variable names
                variables = self.CodeBook.iloc[:,
                            self.PreProcessingSettings['General_information']
                            ['codebook_variable_column']].to_numpy(dtype=str).flatten()

                # find row numbers that have a trigger word
                triggered_rows = np.in1d(style_definitions, self.DataTriggers[data_type])

                # select the variable names
                variables_names = variables[triggered_rows]
                logging.debug(f'{data_type} variable names found at triggered rows are: {variables_names}')

                # categorical variables also should be defined as integers and are determined using the available answers
                if data_type == 'int64':
                    variables_names = self._find_number_of_answers('categorical', variables_names)

            # add manual corrections
            if self.DataCorrections[data_type] is not None:
                variables_names = np.append(variables_names, self.DataCorrections[data_type])
                logging.debug(f'Corrected {data_type} variables are: {variables_names}')

            # remove names that are not present in the data
            variables_data = self.DataSet.columns.to_numpy(dtype=str).flatten()
            variables_names = [variable for variable in variables_names if variable in variables_data]

            return variables_names
        else:
            logging.warning(f'Finding datatypes is only available when using a codebook,\n'
                            f'Specify your variables manually in the settings under "datatype"_corrections')

    def _find_number_of_answers(self, answer_type, variables_names):
        """
        Retrieves the variable names of either boolean or categorical variables.

        :param str answer_type: specify whether to find boolean or categorical variable
        :param numpy.ndarray variables_names: array that boolean or categorical variables will be appended to
        :return: list of variable names
        """
        logging.info(f'Retrieving the number of {answer_type} variables')

        for row in range(0, len(self.CodeBook.index)):
            # boolean and categories can be determined using the number of available answers
            variable_row = self.CodeBook.iloc[row].to_numpy(dtype=str).flatten()

            # if variable already has a type description in codebook checking te number of answers is pointless
            if variable_row[4] == 'nan':

                answers = list(variable_row[5:])

                logging.debug(f'{variable_row[3]} has no codebook datatype and contains '
                              f'{len(answers) - answers.count("nan")} answering options')

                # booleans will have 15 answers that are nan
                if answers.count('nan') == 15 and answer_type == 'boolean':
                    variables_names = np.append(variables_names, variable_row[3])

                # categorical will have less than 15 answers that are nan
                elif answers.count('nan') < 15 and answer_type == 'categorical':
                    variables_names = np.append(variables_names, variable_row[3])
                    # categorical variables are stored separately for later definition in metadata
                    self.CategoricalVariables = np.append(self.CategoricalVariables, variable_row[3])

                else:
                    continue
            else:
                continue

        # add manual corrections
        if self.CategoricalCorrections is not None:
            self.CategoricalVariables = np.append(self.CategoricalVariables, self.CategoricalCorrections)
            logging.debug(f'Corrected categorical variables are: {self.CategoricalVariables}')

        logging.debug(f'{answer_type} variable names found at triggered rows are: {variables_names}')
        return variables_names

    def _remove_double_underscore(self, variables_codebook, discrepancies):
        """
        Align variable names with codebook where possible by removing double underscores.

        :param numpy.ndarray variables_codebook: variable names present in the codebook
        :param numpy.ndarray discrepancies: variable names that are in data but are not found in the codebook
        :return: numpy.ndarray that no longer contains variables that are found in the codebook with a single underscore
        """
        logging.info(f'Removing double underscores in variable names to match Codebook')

        # check discrepancies individually
        for discrepancy in discrepancies:
            corrected_variable = str.replace(discrepancy, '__', '_')

            # if the single underscore variable is found in codebook then replace in dataset
            if corrected_variable in variables_codebook:
                self.DataSet.rename(columns={discrepancy: corrected_variable}, inplace=True)
                logging.debug(f'{discrepancy} was corrected to {corrected_variable}')
            else:
                continue

        # rerun discrepancy test without corrections
        discrepancies = self._find_codebook_data_discrepancies(True, False, False)

        return discrepancies

    def _remove_free_text_fields(self):
        """
        Removes the columns that contain free string. Raises an exception when column name is not found.
        """
        logging.info(f'Removing free string fields.')

        # determine which columns are string that cannot structurally be categorised
        free_text_fields = self._find_data_type('string')

        # remove free_text_fields columns
        for variable in free_text_fields:
            # prevent mismatches
            try:
                logging.debug(f'Removing {variable} from the dataset')
                del self.DataSet[variable]
            except KeyError:
                warnings.warn(f'\nThe variable "{variable}" was not found.\n'
                              f'Ensure that mismatches in format are corrected using the corrections variable.\n'
                              f'Proceeding')
                logging.debug(f'Current corrections consist of: {self.StringCorrections}\n')
                pass

    def _remove_identification(self, hash_id=True, shuffle=True):
        """
        Remove the variable specified in self.Identifier, optionally one can include a hash identifier and
        shuffle the order of original data's rows.

        :param boolean hash_id: specify whether to include a hash identifier
        :param boolean shuffle: specify whether to also shuffle the original order of the rows
        """
        logging.info(f'Removing identification variable {self.Identifier}')

        # remove identifier from dataset and set identifier name to None
        del self.DataSet[self.Identifier]
        self.Identifier = None

        # include a hash_id based on Faker data
        if hash_id:
            logging.info(f'Replacing contents of identification variable {self.Identifier} with Faker hash_id')

            # initialise Faker
            faker = generator_faker.FakerSynthetic(['nl_NL'], self.DataSet)

            # synthesise fake names, sexes, date of birth, personal pronoun, and so forth
            faker.data_synthesise('%m-%d')

            # concatenate with data
            self.DataSet = faker.data_concatenate_to_data(hash_only=True, save=False)

            # ensure primary key is coded correctly
            self.Identifier = 'hash_id'

        # shuffle the order of the rows
        if shuffle:
            logging.info(f'Shuffling original order of data rows')
            self.DataSet = self.DataSet.sample(frac=1)

    def _remove_missing(self, columns_to_check, proportion=None):
        """
        Remove rows that contain more than specified proportion of missing values in the specified columns

        :param float proportion: proportion of missing values to accept
        :param list columns_to_check: specify the columns to check for missing values
        """
        if isinstance(proportion, float) is False:
            proportion = 0.5

        # drop those rows where the proportion of missing values in the columns to check is higher than given proportion
        self.DataSet = self.DataSet.dropna(subset=columns_to_check, thresh=(len(columns_to_check) * proportion))

    def _remove_two_in_bool(self):
        """
        Reformat booleans formatted as 1 and 2 to 0 and 1
        """
        logging.info(f'Reformatting booleans formatted as 1 and 2 to 0 and 1')

        # store the variables temporarily
        variables = self._find_data_type('boolean')

        # alter the data type of every variable
        for variable in variables:
            data_of_variable = self.DataSet[[variable]].to_numpy(dtype=float).flatten()

            if 2 in data_of_variable:
                # booleans can be a mix of string and numerical
                self.DataSet[variable] = self.DataSet[variable].replace(1, 0)
                self.DataSet[variable] = self.DataSet[variable].replace('1', 0)
                self.DataSet[variable] = self.DataSet[variable].replace(2, 1)
                self.DataSet[variable] = self.DataSet[variable].replace('2', 1)

    def _remove_string_component_manual(self, variable, string_to_remove):
        """
        Remove string specified in string_to_remove from specified variable

        :param str variable: name of the variable from which string is to be removed
        :param str string_to_remove: the string that is to be removed
        """
        logging.info(f'Removing {string_to_remove} from variable: {variable}')

        if isinstance(string_to_remove, list):
            for _string_to_remove in string_to_remove:
                # retrieve variables
                entries = self.DataSet[[variable]].to_numpy(dtype=str).flatten()

                # replace string and place in DataFrame
                entries = np.char.replace(entries, _string_to_remove, '', )
                self.DataSet[variable] = entries
        else:
            # retrieve variables
            entries = self.DataSet[[variable]].to_numpy(dtype=str).flatten()

            # replace string and place in DataFrame
            entries = np.char.replace(entries, string_to_remove, '', )
            self.DataSet[variable] = entries

    def _remove_string_component_auto_detect(self, variable):
        """
        Auto determines what is string and removes it from the provided variable
        ** low performance **

        :param str variable: variable name to remove a string component of
        """
        logging.info(f'Removing string components for variable: {variable}')

        data_entries = self.DataSet[[variable]].to_numpy(dtype=str).flatten()

        # remove string and change the value in the dataframe to a string-less value
        for entry in data_entries:
            string_less_entry = int(''.join([symbol for symbol in entry if symbol.isdigit()]))
            self.DataSet[[variable]] = self.DataSet[[variable]].replace(entry, string_less_entry)
            logging.debug(f'Replaced {entry} with {string_less_entry} in column {variable}')

    def _remove_variable_by_content(self, variables_data, content, cut_off=None):
        """
        Remove variables from the dataset when the percentage of the variable's content is higher than the cut-off.

        :param numpy.ndarray variables_data: variable names to check for contents
        :param any content: variable content to trigger deletion
        :param float cut_off: value that determines what proportion is considered acceptable to not remove the variable,
        defaults to 0.25
        :return: adjusted variables names present in the dataset
        """
        if isinstance(cut_off, float) is False:
            cut_off = 0.25

        logging.info(f'Removing variables that have a higher proportion than {cut_off} of entries with {content}')

        for variable in variables_data:
            # retrieve the data per column
            variable_column = self.DataSet[[variable]].to_numpy(dtype=str).flatten()

            # determine the number of fields with the content
            number_with_content = variable_column.tolist().count(content)

            # determine the total number of fields
            number_total = len(variable_column)

            # determine percentage of content containing fields
            proportion_with_content = number_with_content / number_total
            logging.debug(f'The proportion of entries with {content} in {variable} is {proportion_with_content}')

            # remove from dataset and variable array if percentage is higher than or equal to cut-off
            if proportion_with_content >= cut_off:
                del self.DataSet[variable]
                variables_data = variables_data[variables_data != variable]
                logging.debug(f'Removed variable {variable}')
            else:
                continue

        return variables_data
