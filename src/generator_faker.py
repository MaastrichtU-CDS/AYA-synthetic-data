import faker
import hashlib
import logging
import warnings

import pandas as pd

# private modules
from src import file_handling

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def synthesise_hash(data, algorithm_name=None):
    """
    Generate an encrypted output (hash) of the provided data. Default is sha256

    :param str data: data that is to be hashed
    :param algorithm_name: specify an algorithm from hashlib that is to be used for hash generation if so desired
    :return: hash as hex-digest
    """
    # try to use a specified encryption algorithm or default to sha256
    if isinstance(algorithm_name, str):
        try:
            secure_hash = hashlib.new(algorithm_name)
        except ValueError:
            warnings.warn(f'{algorithm_name} is not supported by hashlib. Using sha256 instead.')
            algorithm_name = 'sha256'
            secure_hash = hashlib.sha256()
    else:
        algorithm_name = 'sha256'
        secure_hash = hashlib.sha256()

    logging.debug(f'Synthesising hash using {algorithm_name} for {data}')
    secure_hash.update(data.encode())
    return secure_hash.hexdigest()


def locale_support(desired_locale):
    """
    Checks whether locale is supported and in the correct metadata_format.

    :param str desired_locale: desired locale, format as 'en_GB', 'es_ES', 'zh_CN' and so forth
    """
    logging.info(f'Checking whether {desired_locale} is available')

    # ensure support of locale but avoid unnecessary change
    if desired_locale not in faker.config.AVAILABLE_LOCALES:
        while desired_locale not in faker.config.AVAILABLE_LOCALES:
            warnings.warn(
                f'Locale {desired_locale} is not supported, supported locales:\n {faker.config.AVAILABLE_LOCALES}')
            correct_locale = str(input(f'Please enter a supported locale'))
            if correct_locale == 'exit':
                exit()
            if correct_locale in faker.config.AVAILABLE_LOCALES:
                logging.info(f'Replacing locale {desired_locale} with {correct_locale}')
                return correct_locale

    else:
        logging.debug(f'{desired_locale} is available in faker.config.AVAILABLE_LOCALES')
        return desired_locale


class FakerSynthetic:
    """
    Aims to generate synthetic data using the Faker package
    """

    def __init__(self, desired_locales=None, dataset=None):
        """
        Initialises the FakerSynthetic class

        :param list desired_locales: specify the desired Faker locales as a list of strings
        :param pandas.DataFrame dataset: dataset to append the Faker data to
        """
        if dataset is None:
            # retrieve dataset
            self.DataToSupplementPath = input(
                f'Please specify the filename and path of the csv dataset you would like to add the '
                f'Faker synthetic data to')
            self.DataToSupplement = file_handling.read_csv(self.DataToSupplementPath, 'object')
        else:
            self.DataToSupplementPath = None
            self.DataToSupplement = dataset

        # empty dictionary used to store the dataframes of synthetic data
        self.DataSets = {}

        # empty dictionary used to store Faker objects with varying locale
        self.FakerSynthesisers = {}

        # ensure that there is a list of desired desired_locales
        if desired_locales is None:
            desired_locales = ['en_GB']
        elif isinstance(desired_locales, list) is False:
            desired_locales = list(desired_locales)

        # initialise faker objects for desired locales
        self.initialise_faker(desired_locales)

        # initialise special providers
        self.initialise_custom_provider({'sex': [0, 1],
                                         'personal_pronoun': ['she/her', 'he/him', 'ze/hir']})
        # retrieved pronouns from the European Institute for Gender Equality on the 5th of December 2022
        # Link: https://eige.europa.eu/publications/gender-sensitive-communication/practical-tools/pronouns

    def initialise_faker(self, desired_locales):
        """
        Set up the Faker objects using all desired locales

        :param list desired_locales: specify the desired locales as string in list
        """
        # check locale availability
        for desired_locale in desired_locales:
            desired_locale = locale_support(desired_locale)

            logging.info(f'Initialising Faker object for locale {desired_locale}')
            # initialise faker object using specific locale
            self.FakerSynthesisers.update({desired_locale: faker.Faker(locale=desired_locale)})

    def initialise_custom_provider(self, providers):
        """
        Set up a custom provider i.e. 'subjects to fake' and add them to the existing faker objects

        :param dict providers: specify the name and elements of the custom provider as {name: [element1, element2]}
        """
        # empty dictionary for desired custom provider
        custom_providers = {}
        failsafe = 0

        for provider in providers.keys():
            logging.info(f'Generating a faker DynamicProvider for {provider} using elements {providers[provider]}')
            if provider == 'sex':
                custom_sex = faker.providers.DynamicProvider(provider_name=provider, elements=providers[provider])
                custom_providers.update({provider: custom_sex})
            elif provider == 'personal_pronoun':
                custom_pronoun = faker.providers.DynamicProvider(provider_name=provider, elements=providers[provider])
                custom_providers.update({provider: custom_pronoun})
            else:
                if failsafe > 1:
                    logging.warning(
                        f'More than one custom provider that was not described in initialise_custom_provider'
                        f'has been found.\nReconsider the custom providers or update the function '
                        f'"initialise_custom_provider"')
                    exit()
                failsafe = + 1
                custom_other = faker.providers.DynamicProvider(provider_name=provider, elements=providers[provider])
                custom_providers.update({provider: custom_other})

        # add the custom provider to each faker object
        for synthesiser in self.FakerSynthesisers.keys():
            for custom_provider in custom_providers.keys():
                logging.info(f'Adding custom provider {custom_provider} to faker {synthesiser} object')
                self.FakerSynthesisers[synthesiser].add_provider(custom_providers[custom_provider])

    def data_synthesise(self, format_date='%m-%d', min_age=15, max_age=39, include_hash=True):
        """
        Generate a first and family name, sex, date of birth, personal pronoun, and postal code for the
        specified sample and locale

        :param str format_date: specify the format date of birth should be saved as e.g., '%m-%d', '%Y-%m-%d' et cetera
        :param int min_age: minimum age for date of birth
        :param int max_age: maximum age for date of birth
        :param boolean include_hash: specify whether to include an encrypted identifier (hash)
        """
        for desired_locale in self.FakerSynthesisers.keys():
            logging.info(f'Synthesising {desired_locale} first and family name, sex, date of birth, personal pronoun, '
                         f'and postal code.')

            # generate synthetic data per row
            data_rows = [{'first_name': self.FakerSynthesisers[desired_locale].first_name(),
                          'family_name': self.FakerSynthesisers[desired_locale].last_name(),
                          'sex': self.FakerSynthesisers[desired_locale].sex(),
                          'date_of_birth': self.FakerSynthesisers[desired_locale].date_of_birth(
                              minimum_age=min_age, maximum_age=max_age).strftime(format=format_date),
                          'personal pronoun': self.FakerSynthesisers[desired_locale].personal_pronoun(),
                          'postal_code': self.FakerSynthesisers[desired_locale].postcode()}
                         for _ in range(len(self.DataToSupplement))]

            # transform to pandas DataFrame
            synthetic_dataframe = pd.DataFrame(data_rows)

            # include encrypted identifier when specified
            if include_hash:
                logging.info(f'Including encrypted identifier based on synthesised data.')
                hashes = [synthesise_hash(synthetic_dataframe.iloc[row].to_string())
                          for row in range(len(synthetic_dataframe))]
                synthetic_dataframe.insert(0, 'hash_id', hashes, True)

            self.DataSets.update({desired_locale: synthetic_dataframe})

    def data_concatenate_to_data(self, hash_only=False, save=True):
        """
        Concatenate the Faker data to the dataset of interest per locale

        :param boolean hash_only: only concatenate the hash rather than all generated Faker synthetics
        :param boolean save: specify whether to save the supplemented dataset
        """
        logging.info(f'Concatenating synthetic data synthesised with Faker with dataset {self.DataToSupplementPath}')

        # concatenate Faker synthetic to DataToSupplement so that identifying variables are amongst the first columns
        for desired_locale in self.DataSets.keys():
            if hash_only:
                supplement = self.DataSets[desired_locale]['hash_id']
            else:
                supplement = self.DataSets[desired_locale]
            data_including_fakes = pd.concat([supplement, self.DataToSupplement], axis=1, join='inner')

            # save the file if so desired
            if save:
                filename = self.DataToSupplementPath.replace('.csv', f'_{desired_locale}.csv')
                file_handling.save_csv(data_including_fakes, filename)
            else:
                return data_including_fakes
