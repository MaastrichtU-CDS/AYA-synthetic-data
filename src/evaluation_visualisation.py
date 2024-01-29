"""
Aims to provide numerous functions to visualise the output of data evaluation and thus the quality of your data
"""
import logging
import os
import warnings

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

# private modules
from src import file_handling, evaluation_metrics

logging.basicConfig(filename='log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sns_context = 'paper'
sns_colour_palette = 'colorblind'
# matplotlib markers https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
markers = ['v', 'o', '^', 'X', '<', '>']


def visualise_bland_altman(reference_data, analysis_data, dataset, generator_name, settings, sample_size,
                           variable_prefix='',  effect_measure='odds ratio', context=sns_context,
                           colour_palette=sns_colour_palette, filename=None, save=True):
    """
    Creates a Bland-Altman plot for predicted probabilities.

    Parameters:
    :param dict reference_data: List of dictionaries containing coefficients for set 1 as list[dict[str: float]]
    :param list analysis_data: List of dictionaries containing coefficients for set 2 as list[dict[str: float]]
    :param pandas.DataFrame dataset: Input data with features.
    """
    repetitions = 3
    # retrieve the variables as specified in the settings
    if isinstance(settings, str):
        settings = file_handling.read_json(settings)
    # noinspection PyUnresolvedReferences
    variables = settings['Evaluation']['variables_nickname'].keys()

    coefficients_set1 = {variable: reference_data[f'{variable_prefix}{variable}{effect_measure}']
                         for variable in variables}

    coefficients_set1 = [coefficients_set1 for repeat in range(repetitions)]

    coefficients_set2 = _retrieve_coefficients(analysis_data, variables, sample_size, variable_prefix, effect_measure)


    # Predict probabilities using logistic regression for both sets of coefficients and repetitions
    predicted_probs_set1 = np.zeros((len(dataset), repetitions))
    predicted_probs_set2 = np.zeros((len(dataset), repetitions))

    for i in range(repetitions):
        predicted_probs_set1[:, i] = evaluation_metrics.linear_predictor(coefficients_set1[i], dataset,
                                                                                    convert_odds_ratios=True)

    for i in range(repetitions):
        predicted_probs_set2[:, i] = evaluation_metrics.linear_predictor(coefficients_set2[i], dataset,
                                                                                    convert_odds_ratios=True)

    # Calculate mean predicted probabilities for each set
    mean_predicted_probs_set1 = np.mean(predicted_probs_set1, axis=1)
    mean_predicted_probs_set2 = np.mean(predicted_probs_set2, axis=1)

    # Bland-Altman plot
    sns.set_context(context)
    colours = sns.color_palette(colour_palette, n_colors=len(dataset.columns)).as_hex()
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()

    sns.scatterplot(x=(mean_predicted_probs_set1 + mean_predicted_probs_set2) / 2,
                    y=mean_predicted_probs_set1 - mean_predicted_probs_set2)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)

    # place title at similar height as other plots
    ax1.title(f'Bland-Altman Plot for Predicted Probabilities of {generator_name}', y=1.1)

    ax1.xlabel('Mean Predicted Probabilities', labelpad=10, loc='center')
    # retrieve ticks and ensure there are no more than 10 for cleanliness
    x_ticks = ax1.get_xticks()
    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    ax1.ylabel('Difference between Predicted Probabilities', labelpad=-20, loc='top', rotation='horizontal')
    # adjust the y-axis limits to show both positive and negative differences
    ax1.ylim(-1, 1)

    # save when desired
    if save:
        if filename is None:
            filename = input(
                f'Please provide an appropriate filename including a path for the Bland-Altman plot of'
                f' the {generator_name} generator.')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved Bland-Altman plot for generator {generator_name} under\n{filename}')


def visualise_line_and_box(dataset, x_variable, x_label=None, x_type=int, x_ticks_from_data=False,
                           y_variables_lines=None, y_limits_lines=None, y_type_lines=float, y_line_width=None,
                           y_label_lines='Veracity\nScore\n',
                           y_variables_bars=None, y_limits_bars=None, y_bars_types_to_find=None,
                           y_label_bars='Privacy\nConcealment\nScore\n',
                           context=sns_context, colour_palette=sns_colour_palette, filename=None, save=True):
    """
    Create a line plot with a box plot in the background; e.g., used to compare the statistical similarity with privacy.
    Ideally used to get a global idea of the trade-off, not ideal for publication and sharing

    :param pandas.DataFrame dataset: dataset containing the data to plot
    :param str x_variable: the name of the variable to plot on the x-axis i.e., column name
    :param any x_label: label for the x-axis, defaults to x-variable
    :param type x_type: define the type that x should be displayed as e.g., integer or float
    :param bool x_ticks_from_data: specify whether to use np linspace or actual datapoints for x-axis ticks
    :param str y_variables_lines: the variable to draw (a) line(s) for i.e., column name
    :param float y_limits_lines: set the min and max of the y-axis related to lines, defaults to (0, 1)
    :param type y_type_lines: define the type that x should be displayed as e.g., integer or float
    :param int y_line_width: specify the width of the lines drawn in the plot
    :param str y_label_lines: label for the y-axis related to lines, defaults to statistical similarity score
    :param str y_variables_bars: the variable to draw (a) bar(s) for i.e., column name
    :param list y_bars_types_to_find: specify the privacy metric to use,
    i.e., categorical or continuous identity disclosure
    :param tuple of floats y_limits_bars: set the limits of the y-axis related to bars, no default
    :param str y_label_bars: label for the y-axis related to bars, defaults to privacy disclosure score
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    logging.info(f'Generating a box and line plot for analysis of {x_variable}')

    twin = False
    # check whether the variable is of float type
    string_label = False
    try:
        dataset = dataset.astype({x_variable: float})
    except ValueError:
        string_label = True

    # create the empty plot with general settings
    sns.set_context(context)
    colours = sns.color_palette(colour_palette, n_colors=len(dataset.columns)).as_hex()
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()

    # determine what lines to plot
    if y_variables_lines is None:
        y_variables_lines = np.array([variable_name for variable_name in ['precision', 'recall', 'density', 'coverage']
                                      if variable_name in dataset.columns])
        # reset markers to correspond with fidelity and diversity
        # noinspection PyShadowingNames
        markers = ['o', 'X', 'o', 'X']
    else:
        # noinspection PyShadowingNames
        markers = ['v', 'o', '^', 'X', '<', '>']

    if len(y_variables_lines) == 0:
        y_variables_lines = []
    else:
        # settings for y-axis for lines
        if y_limits_lines is None:
            y_limits_lines = (0, 1)

        if isinstance(y_line_width, (int, float)) is False:
            y_line_width = 1
            _y_line_style = '--'
        else:
            _y_line_style = '-'

        # add every variable to the plot
        for variable_number, variable in enumerate(y_variables_lines):
            sns.lineplot(x=dataset[x_variable], y=dataset[variable].astype(y_type_lines), label=variable,
                         color=colours[variable_number], linestyle='-', linewidth=y_line_width,
                         marker=markers[variable_number], markersize=6.5, ax=ax1)

        ax1.set_ylim(y_limits_lines)
        ax1.set_ylabel(y_label_lines, labelpad=-20, loc='top', rotation='horizontal')

    # determine what boxes to plot
    if y_variables_bars is None:
        y_variables_bars = np.array([variable_name for variable_name in ['categorical disclosure min',
                                                                         'categorical disclosure first quartile',
                                                                         'categorical disclosure median',
                                                                         'categorical disclosure third quartile',
                                                                         'categorical disclosure max',
                                                                         'continuous disclosure min',
                                                                         'continuous disclosure first quartile'
                                                                         'continuous disclosure median',
                                                                         'continuous disclosure third quartile',
                                                                         'continuous disclosure max']
                                     if variable_name in dataset.columns])
    else:
        y_variables_bars = np.array([variable_name for variable_name in y_variables_bars
                                     if variable_name in dataset.columns])

    # check whether data for a second axis is there or avoid plotting it
    if len(y_variables_bars) == 0:
        pass
    else:
        if len(y_variables_lines) != 0:
            ax2 = ax1.twinx()
            twin = True
            label_alignment = 'left'
        else:
            ax2 = ax1
            label_alignment = 'right'

        # reverse colours to avoid colour overlap with lines
        colours.reverse()

        if y_bars_types_to_find is None:
            y_bars_types_to_find = ['categorical']

        if string_label:
            _positions = ax1.get_xticks()
        else:
            _positions = dataset[x_variable]

        # add every variable to the plot per data point
        for variable_number, variable in enumerate(y_bars_types_to_find):
            bxp_stats = [{'whislo': _rescale(dataset.iloc[index][f'{variable} disclosure min'],
                                             minimum=0, maximum=dataset[f'{variable} disclosure max'].max()),
                          'q1': _rescale(dataset.iloc[index][f'{variable} disclosure first quartile'],
                                         minimum=0, maximum=dataset[f'{variable} disclosure max'].max()),
                          'med': _rescale(dataset.iloc[index][f'{variable} disclosure median'],
                                          minimum=0, maximum=dataset[f'{variable} disclosure max'].max()),
                          'q3': _rescale(dataset.iloc[index][f'{variable} disclosure third quartile'],
                                         minimum=0, maximum=dataset[f'{variable} disclosure max'].max()),
                          'whishi': _rescale(dataset.iloc[index][f'{variable} disclosure max'],
                                             minimum=0, maximum=dataset[f'{variable} disclosure max'].max())}
                         for index in range(0, len(dataset[x_variable]))]

            boxes = ax2.bxp(bxp_stats, positions=_positions, showfliers=False, patch_artist=True)

            # specify items to change the aesthetics of
            box_aesthetics = ['whiskers', 'caps', 'boxes', 'medians']

            # force aesthetics; workaround is necessary as aesthetics otherwise cannot be altered with ax bxp
            for graph_component in box_aesthetics:
                for box in boxes[graph_component]:
                    box.set_color(colours[variable_number])
                    # emphasise boxes
                    if graph_component in 'boxes':
                        box.set_alpha(0.86)
                    else:
                        box.set_alpha(0.33)

        # settings for y-axis for bars
        if y_limits_bars is not None:
            ax2.set_ylim(y_limits_bars)
        else:
            # force a zero as lower limit and one as upper
            ax2.set_ylim((0, 1))

        ax2.set_ylabel(y_label_bars, labelpad=20, horizontalalignment=label_alignment, rotation='horizontal')

        # workaround for twin y-axis label being stubborn
        if twin:
            ax2.yaxis.set_label_coords(1, 1.13)
        else:
            ax2.yaxis.set_label_coords(0, 1.13)

    # add a legend based on the different variables in the graph
    ax1.legend(loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), ncols=len(y_variables_lines), frameon=False)

    # retrieve ticks
    x_ticks = dataset[x_variable].to_numpy()

    # avoid decimals in case of numeric ticks for x-axis
    if string_label is False:
        x_ticks = x_ticks.astype(x_type)
        if len(x_ticks) >= 11:
            # only use ticks that are actually in the data
            x_ticks_positions = np.linspace(min(x_ticks), max(x_ticks), num=10).astype(x_type)
            if x_ticks_from_data:
                x_ticks = [_find_nearest(x_ticks, position) for position in x_ticks_positions]
            else:
                x_ticks = x_ticks_positions
            ax1.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
            ax1.set_xticklabels([x for x in x_ticks])
    else:
        if len(x_ticks) >= 11:
            ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.set_xticks(dataset[x_variable], labels=dataset[x_variable])

    if string_label is False:
        # ensure there is space in the start and end of the graph
        whitespace = min(x_ticks) * 0.5
        ax1.set_xlim(whitespace, max(x_ticks) + whitespace)

    if x_label is not None:
        ax1.set_xlabel(x_label.replace('_', ' '), labelpad=10, loc='center')
    else:
        ax1.set_xlabel(x_variable.replace('_', ' '), labelpad=10, loc='center')

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path for the box and line '
                             f'plot for analysis of {x_variable}')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved statistical similarity plot for analysis of {x_variable} under\n{filename}')


def visualise_effect_single_measure(dataset, y_variable, x_variable, reference_data=None, y_type=int, settings=None,
                                    y_ticks_from_data=False, remove_y_axis=False,
                                    effect_measure_identifiers=None, effect_measure=None,
                                    x_label=None, y_label='Synthetic dataset\nsample size',
                                    statistical_similarity_limit=None, filter_data=True,
                                    effect_measure_limit=None, effect_ci_limit=None, format_axes=True,
                                    context=sns_context, colour_palette=sns_colour_palette, filename=None, save=True):
    """
    Create a forest plot for a single association/effect, reference data is shown as a coloured box within the plot

    :param pandas.DataFrame dataset: dataset containing the data to plot
    :param str x_variable: the name of the variable to plot on the x-axis e.g., association of age with y
    :param str y_variable: the name of the variable to plot on the y-axis e.g., sample size
    :param dict reference_data: outcomes of the data from which the synthetic data is derived from
    :param type y_type: define the type that y should be displayed as, e.g., integer or float
    :param dict settings: dict that contains nicknames for renaming of the variables
    :param bool y_ticks_from_data: specify whether to extract the y ticks from the data or use interpolated values
    :param bool remove_y_axis: specify whether to remove the y-axis, useful when displayed several forest plots
    :param str effect_measure: specify whether to use odds ratio
    :param bool filter_data: specify whether to remove certain data that is outside specified limits
    :param float statistical_similarity_limit: minimum score to accept for statistical similarity metrics
    :param tuple effect_measure_identifiers: keywords used to denote the effect measures e.g., 'odds ratio'
    :param float effect_measure_limit: limit to filter on regarding the main effect
    :param float effect_ci_limit: limit to filter on regarding the confidence intervals
    :param bool format_axes: specify whether axes aesthetics should be changed
    :param str x_label: label that is to be displayed on the x-axis
    :param str y_label: label that is to be displayed on the y-axis
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    # try loop as in the automatic generation and plotting flow errors will unnecessarily interrupt the process
    try:
        if effect_measure is None:
            effect_measure = 'OR'

        if x_label is None:
            if effect_measure == 'OR':
                x_label = 'odds ratio'

        if effect_measure_identifiers is None:
            effect_measure_identifiers = (f'0.95% confidence interval (lower)',
                                          f'{x_label}',
                                          f'0.95% confidence interval (upper)')

        if filter_data:
            # filter out those datasets that have below satisfactory statistical similarity scores
            if statistical_similarity_limit is None:
                statistical_similarity_limit = 0.15
            for statistical_similarity_metric in ['precision', 'recall', 'density', 'coverage']:
                dataset = dataset[(dataset.filter(like=statistical_similarity_metric)
                                   > statistical_similarity_limit).any(axis=1)]

            # filter out those datasets that have negative a negative effect measure
            dataset = dataset[(dataset.filter(like=f'{x_variable} {effect_measure_identifiers[0]}') >= 0).any(axis=1)]
            dataset = dataset[(dataset.filter(like=f'{x_variable} {effect_measure_identifiers[1]}') >= 0).any(axis=1)]
            dataset = dataset[(dataset.filter(like=f'{x_variable} {effect_measure_identifiers[2]}') >= 0).any(axis=1)]

            # filter out those datasets that have an effect measure value that is larger than set value
            if effect_measure_limit is None:
                effect_measure_limit = 7.5
            dataset = dataset[
                (dataset.filter(like=f'{x_variable} {effect_measure_identifiers[1]}') < effect_measure_limit).any(
                    axis=1)]

            # filter out those datasets that have an effect measure confidence interval value that is larger than set value
            if effect_ci_limit is None:
                effect_ci_limit = 15
            dataset = dataset[
                (dataset.filter(like=f'{x_variable} {effect_measure_identifiers[0]}') < effect_ci_limit).any(axis=1)]
            dataset = dataset[
                (dataset.filter(like=f'{x_variable} {effect_measure_identifiers[2]}') < effect_ci_limit).any(axis=1)]
        try:
            sns.set_context(context)
            colours = sns.color_palette(colour_palette, n_colors=len(dataset.columns)).as_hex()
            fig, ax = plt.subplots(figsize=(5.625, 10))

            if remove_y_axis:
                plt.tick_params(left=False)
                ax.get_yaxis().set_visible(False)
                sns.despine(left=True)
            else:
                sns.despine()

            # add every variable to the plot
            bxp_stats = [{'whislo': dataset.iloc[index][f'{x_variable} {effect_measure_identifiers[0]}'],
                          'q1': dataset.iloc[index][f'{x_variable} {effect_measure_identifiers[1]}'],
                          'med': dataset.iloc[index][f'{x_variable} {effect_measure_identifiers[1]}'],
                          'q3': dataset.iloc[index][f'{x_variable} {effect_measure_identifiers[1]}'],
                          'whishi': dataset.iloc[index][f'{x_variable} {effect_measure_identifiers[2]}']}
                         for index in range(len(dataset[y_variable]))]

            if isinstance(dataset[y_variable], str):
                _positions = dataset[y_variable].index
            else:
                _positions = dataset[y_variable]

            boxes = ax.bxp(bxp_stats, positions=_positions, showfliers=False,
                           patch_artist=True, vert=False,
                           widths=len(dataset[y_variable]), capwidths=len(dataset[y_variable]) * 1.5)

            # force aesthetics on specific items of the boxplot
            for graph_component in ['whiskers', 'caps', 'boxes', 'medians']:
                for box in boxes[graph_component]:
                    box.set_color(colours[2])
                    # emphasise boxes
                    if graph_component in 'boxes':
                        box.set_alpha(1)
                    else:
                        box.set_alpha(0.53)

            if format_axes:
                # retrieve ticks
                y_ticks = dataset[y_variable].to_numpy()

                # check whether the variable is of float type
                string_label = False
                try:
                    dataset = dataset.astype({y_variable: float})
                except ValueError:
                    string_label = True

                # avoid decimals in case of numeric ticks for x-axis
                if string_label is False:
                    y_ticks = y_ticks.astype(y_type)
                    if len(y_ticks) >= 11:
                        # only use ticks that are actually in the data
                        y_ticks_positions = np.linspace(min(y_ticks), max(y_ticks), num=10).astype(y_type)
                        if y_ticks_from_data:
                            y_ticks = [_find_nearest(y_ticks, position) for position in y_ticks_positions]
                        else:
                            y_ticks = y_ticks_positions
                        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
                        ax.set_yticklabels([x for x in y_ticks])
                else:
                    if len(y_ticks) >= 11:
                        ax.yaxis.set_major_locator(mticker.MaxNLocator(10))

                if isinstance(y_label, str):
                    ax.set_ylabel(f'{y_label}', labelpad=-35, loc='top', rotation='horizontal')
                else:
                    ax.set_ylabel(f'{y_variable.replace("_", " ")}', labelpad=-35, loc='top', rotation='horizontal')

                # allow renaming of the variables
                if isinstance(settings, dict):
                    variables = settings['variables_nickname']
                    ax.set_xlabel(f'Odds ratio\n{variables[f"{x_variable} "]}', labelpad=10, loc='center')
                else:
                    ax.set_xlabel(f'Odds ratio\n{x_variable}', labelpad=10, loc='center')

                # ensure 0 is included in the plot for consistency; upper limit is arbitrary
                ax.set_xlim(0, 5)

                # ensure there is space in the start and end of the graph
                whitespace = min(dataset[y_variable]) * 0.5
                ax.set_ylim(whitespace, max(dataset[y_variable]) + whitespace)

            if reference_data is not None:
                # retrieve the data
                lower_confidence_ref = reference_data[f'{x_variable} {effect_measure_identifiers[0]}']
                odds_ratio = reference_data[f'{x_variable} {effect_measure_identifiers[1]}']
                upper_confidence_ref = reference_data[f'{x_variable} {effect_measure_identifiers[2]}']

                # add odds ratio reference line, i.e., odds ratio equal to 1 / no effect
                plt.axvline(x=1, linewidth=0.8, linestyle='--', color='#808080')

                # add real / non-synthetic data odds ratio line
                plt.axvline(x=odds_ratio, linewidth=0.8, linestyle='-', color=colours[0])

                # add a rectangle that covers the area of the real / non-synthetic confidence interval
                ax.add_patch(patches.Rectangle(xy=(lower_confidence_ref, 0.0),
                                               width=upper_confidence_ref - lower_confidence_ref,
                                               height=max(dataset[y_variable]), linewidth=1, color=colours[0],
                                               alpha=0.05))

                # add text for each respective addition
                ax.text(lower_confidence_ref, max(dataset[y_variable]), f'lower\nCI\n', horizontalalignment='right',
                        color='#808080', fontstyle='italic', fontsize='small', fontweight='ultralight')
                ax.text(upper_confidence_ref, max(dataset[y_variable]), f'upper\nCI\n', horizontalalignment='left',
                        color='#808080', fontstyle='italic', fontsize='small', fontweight='ultralight')
                ax.text(odds_ratio, max(dataset[y_variable]), f'Original odds ratio\n\n\n\n',
                        horizontalalignment='center',
                        color='#808080', fontstyle='italic', fontsize='small', fontweight='light')
            else:
                # add odds ratio reference line, i.e., odds ratio equal to 1 / no effect
                plt.axvline(x=1, linewidth=0.8, linestyle='--', color='#808080')

            # save when desired
            if save:
                if filename is None:
                    filename = input(f'Please provide an appropriate filename including a path for the effect measure'
                                     f'plot for analysis of {y_variable}, variable {x_variable}')
                plt.savefig(f'{filename}', dpi=350)
                logging.info(f'Saved effect measure plot for analysis of {y_variable}, variable {x_variable}'
                             f'under\n{filename}')
        except ValueError:
            logging.warning(
                f'Unable to generate effect measure plot for analysis of {y_variable}, variable {x_variable}')

    except KeyError:
        logging.warning(f'Unable to generate effect measure plot for analysis of {y_variable}, variable {x_variable}')


def visualise_effect_all_measures(dataset=None, reference_data=None, settings=None, sample_size=3100,
                                  remove_y_axis=False, x_label=None, metric=None, privacy=False, format_axes=True,
                                  effect_measure_identifiers=None, effect_measure=None, x_unit=None,
                                  variable_prefix=None, context=sns_context, colour_palette=sns_colour_palette,
                                  filename=None, save=True):
    """
    Create a forest plot for different associations/effects for a single sample, reference data is displayed next to the data

    :param pandas.DataFrame dataset: dataset of the analysis to plot
    :param dict reference_data: outcomes of the data from which the synthetic data is derived from
    :param dict settings: dict that contains nicknames for renaming of the variables
    :param bool remove_y_axis: specify whether to remove the y-axis, useful when displayed several forest plots
    :param str x_label: specify the label for the x-axis
    :param int sample_size: the sample size to use for plotting; must be available in the provided analysis/dataset
    :param str metric: specify a metric to find the best sample of; not used when sample size is defined
    i.e., select the sample size with the best metric score
    :param bool privacy: account for identity disclosure; not used when sample size is defined
    i.e., select the sample size with the best metric score given that it does not have any original samples
    :param str effect_measure: specify whether to use odds ratio
    :param tuple effect_measure_identifiers: keywords used to denote the effect measures e.g., 'odds ratio'
    :param bool format_axes: specify whether axes aesthetics should be changed
    :param str x_unit: unit that was used in the analysis, e.g., odds ratio
    :param str variable_prefix: specify a prefix used for in the analysis, e.g., univariable
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    :return:
    """
    # set certain defaults
    if variable_prefix is None:
        variable_prefix = ''

    if isinstance(reference_data, str):
        reference_data = file_handling.read_json(reference_data)

    if effect_measure is None:
        effect_measure = 'OR'

    if x_unit is None:
        if effect_measure == 'OR':
            x_unit = 'odds ratio'

    if effect_measure_identifiers is None:
        effect_measure_identifiers = (f'0.95% confidence interval (lower)',
                                      f'{x_unit}',
                                      f'0.95% confidence interval (upper)')

    # retrieve variable names
    if isinstance(settings, str):
        settings = file_handling.read_json(settings)
        variables = settings['Evaluation']['variables_nickname']
    else:
        if isinstance(dataset, pd.DataFrame):
            variables = {variable[:variable.rfind(effect_measure_identifiers[1])]: variable
                         for variable in dataset.filter(like=effect_measure_identifiers[1], axis=1)}
        else:
            variables = {variable[:variable.rfind(effect_measure_identifiers[1])]: variable
                         for variable in reference_data.keys() if effect_measure_identifiers[1] in variable}

    sns.set_context(context)
    colours = sns.color_palette(colour_palette, n_colors=5).as_hex()
    fig, ax = plt.subplots(figsize=(5.625, 10))

    if remove_y_axis:
        plt.tick_params(left=False)
        ax.get_yaxis().set_visible(False)
        sns.despine(left=True)
    else:
        sns.despine()

    if reference_data is not None:
        # add every variable to the plot
        bxp_stats = [{'whislo': reference_data[f'{variable_prefix}{variable}{effect_measure_identifiers[0]}'],
                      'q1': reference_data[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'med': reference_data[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'q3': reference_data[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'whishi': reference_data[f'{variable_prefix}{variable}{effect_measure_identifiers[2]}'],
                      'label': label}
                     for variable, label in variables.items()]

        # fill plot with boxes
        boxes = ax.bxp(bxp_stats, showfliers=False, patch_artist=True, vert=False)

        if isinstance(dataset, pd.DataFrame):
            # force aesthetics on specific items of the boxplot
            for graph_component in ['whiskers', 'caps', 'boxes', 'medians']:
                for box in boxes[graph_component]:
                    box.set_color(colours[0])
                    # emphasise boxes
                    if graph_component in 'boxes':
                        box.set_alpha(1)
                        box.set_linewidth(2)
                    elif graph_component in 'caps':
                        box.set_alpha(1)
                        box.set_linewidth(1.5)
                    else:
                        box.set_alpha(0.75)
        else:
            # force aesthetics on specific items of the boxplot
            for graph_component in ['whiskers', 'caps', 'boxes', 'medians']:
                for box in boxes[graph_component]:
                    box.set_color(colours[0])
                    # emphasise boxes
                    if graph_component in 'boxes' or graph_component in 'med':
                        box.set_alpha(1)
                        box.set_linewidth(2)
                    elif graph_component in 'caps':
                        box.set_alpha(1)
                        box.set_linewidth(1.5)
                    else:
                        box.set_alpha(0.75)

        if format_axes:
            ax.set_yticklabels(variables)

    if isinstance(dataset, pd.DataFrame):
        dataset = _retrieve_best_sample(dataset, sample_size, account_for_privacy=privacy, metric=metric)
        try:
            _sample_size = int(dataset['n_output'].values[0])
        except AttributeError:
            _sample_size = int(dataset['n_output'])

        # add every variable to the plot
        bxp_stats = [{'whislo': dataset[f'{variable_prefix}{variable}{effect_measure_identifiers[0]}'],
                      'q1': dataset[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'med': dataset[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'q3': dataset[f'{variable_prefix}{variable}{effect_measure_identifiers[1]}'],
                      'whishi': dataset[f'{variable_prefix}{variable}{effect_measure_identifiers[2]}'],
                      'label': label}
                     for variable, label in variables.items()]

        # fill plot with boxes
        boxes = ax.bxp(bxp_stats, showfliers=False, patch_artist=True, vert=False)

        # force aesthetics on specific items of the boxplot
        for graph_component in ['whiskers', 'caps', 'boxes', 'medians']:
            for box in boxes[graph_component]:
                box.set_color(colours[1])
                # emphasise boxes
                if graph_component in 'boxes':
                    box.set_alpha(1)
                    box.set_linewidth(2)
                elif graph_component in 'caps':
                    box.set_alpha(1)
                    box.set_linewidth(1.5)
                else:
                    box.set_alpha(0.75)
                    box.set_linestyle('-.')

    # noinspection PyUnboundLocalVariable
    ax.set_xlabel(f'{x_label}\nSynthetic dataset with {_sample_size} samples', labelpad=10, loc='center')
    # ensure 0 is included in the plot for consistency
    ax.set_xlim(0, 5)

    # add odds ratio reference line, i.e., odds ratio equal to 1 / no effect
    plt.axvline(x=1, linewidth=0.8, linestyle='--', color='#808080')

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path for the effect measure plot '
                             f'containing all effect measures')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved effect measure plot for analysis under\n{filename}')


def visualise_performance_multi(evaluation_folders, sample_sizes_to_include, metrics_to_show, title=None,
                                x_label='Synthetic dataset sample size', y_label='Statistical\nSimilarity\nScore\n',
                                context=sns_context, colour_palette=sns_colour_palette,
                                filename=None, save=True):
    """
    Create a line plot containing the statistical similarity scores of multiple generators

    :param dict evaluation_folders: specify the folders as {'generator_name': 'folder'}
    :param list sample_sizes_to_include: specify the sample sizes to plot as string in list
    :param list metrics_to_show: specify the metrics which to draw in the plot
    :param str title: provide a title for the plot
    :param str x_label: label for the x-axis
    :param str y_label: label for the y-axis
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    logging.info(f'Generating a statistical similarity plot for multiple generators')

    # create the empty plot with general settings
    sns.set_context(context)
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()

    y_line_width = 1
    _y_line_style = ['-', '--', '-.', ':']

    _artificial_index = 0

    # extract the data and retrieve it in a long format so that seaborn can do the work
    dataset = _extract_statistical_similarity_multi(evaluation_folders, metrics_to_show, sample_sizes_to_include)

    sns.lineplot(data=dataset, x='input', y='output', palette=colour_palette,
                 markers=True, hue='model', style='input sample size', linewidth=y_line_width, ax=ax1)

    # manually extract the legend information to exclude certain components generated by seaborn/matplotlib
    handles, labels = ax1.get_legend_handles_labels()

    # remove the grouping variable name
    del handles[labels.index('model')]
    del labels[labels.index('model')]

    # add a legend based on the generators in the graph
    lines_legend = ax1.legend(handles=handles[:labels.index('input sample size')],
                              labels=labels[:labels.index('input sample size')],
                              loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2),
                              ncols=len(evaluation_folders.keys()), frameon=False)

    # add a separate legend related to the sample size
    ax1.legend(handles=handles[labels.index('input sample size') + 1:],
               labels=labels[labels.index('input sample size') + 1:],
               loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
               title='Original dataset\n    sample size', alignment='center', title_fontsize='small')

    # add the previously generated legend
    ax1.add_artist(lines_legend)

    # format the axes
    ax1.set_xlabel(x_label.replace('_', ' '), labelpad=10, loc='center')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(y_label, labelpad=-20, loc='top', rotation='horizontal')

    # retrieve ticks and ensure there are no more than 10 for cleanliness
    x_ticks = ax1.get_xticks()
    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    # place the title above the legend in the top
    if isinstance(title, str):
        ax1.set_title(title, y=1.1)

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path for the statistical similarity '
                             f'plot for analyses of {evaluation_folders.keys()}')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved multiple generator statistical similarity plot for analysis of under\n{filename}')


def visualise_number_needed_to_print(evaluation_folders, privacy_disclosure_limit=None, title=None,
                                     x_label='Original dataset sample size\n', y_label='Synthetic dataset\nsample size',
                                     context=sns_context, colour_palette=sns_colour_palette, filename=None, save=True):
    """
    Generate a plot with the maximum number of synthetic samples whilst not exceeding the identity disclosure limit
    for multiple generators

    :param dict evaluation_folders: specify the folders as {'generator_name': 'folder'}
    :param float privacy_disclosure_limit: minimally accepted identity disclosure limit, equals to zero when set to none
    :param str x_label: label for the x-axis
    :param str y_label: label for the y-axis
    :param str title: provide a title for the plot
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    logging.info(f'Generating a statistical similarity plot for analysis of')

    # create the empty plot with general settings
    sns.set_context(context)
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()
    y_line_width = 1

    # extract the data and retrieve it in a long format so that seaborn can do the work
    nnt_data = evaluation_metrics.compute_number_needed_to_print(evaluation_folders,
                                                                 privacy_disclosure_limit=privacy_disclosure_limit)

    sns.lineplot(data=nnt_data, x='input', y='output', palette=colour_palette,
                 hue='model', style='model', linewidth=y_line_width, ax=ax1)

    # add a legend based on the different variables in the graph
    lines_legend = ax1.legend(loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2),
                              ncols=len(evaluation_folders.keys()),
                              frameon=False)

    scatter = sns.scatterplot(data=nnt_data, x='input', y='output', palette=colour_palette,
                              size='coverage/density', sizes=(10, 100),
                              hue='model', markers=True, ax=ax1, legend='brief')

    # manually extract the legend information to exclude certain components generated by seaborn/matplotlib
    scatter_handles, scatter_labels = scatter.get_legend_handles_labels()

    # generate a legend
    ax1.legend(handles=scatter_handles[scatter_labels.index('coverage/density') + 1:],
               labels=scatter_labels[scatter_labels.index('coverage/density') + 1:],
               loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
               title='           Average\nCoverage and Density', alignment='center', title_fontsize='small')

    # add separate legend lines
    ax1.add_artist(lines_legend)

    # format the axes
    ax1.set_xlabel(x_label.replace('_', ' '), labelpad=10, loc='center')
    ax1.set_ylabel(y_label, labelpad=-20, loc='top', rotation='horizontal')

    # retrieve ticks and ensure there are no more than 10 for cleanliness
    x_ticks = ax1.get_xticks()
    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    # place the title above the legend in the top
    if isinstance(title, str):
        ax1.set_title(title, y=1.1)

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path for the concordance statistic '
                             f'plot for analysis of')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved statistical similarity plot for analysis of under\n{filename}')


def visualise_duplicates(evaluation_folders, sample_size=None, title=None,
                         x_label='Synthetic dataset sample size', y_label='Original samples\namongst synthetics',
                         context=sns_context, colour_palette=sns_colour_palette, filename=None, save=True):
    """
    Generate a plot in which identity and attribute concealment in relation to varying synthetic dataset sample size


    :param dict evaluation_folders: specify the folders as {'generator_name': 'folder'}
    :param int sample_size:
    :param str x_label: label for the x-axis
    :param str y_label: label for the y-axis
    :param str title: provide a title for the plot
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    logging.info(f'Generating a statistical similarity plot for analysis of')

    # create the empty plot with general settings
    sns.set_context(context)
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()
    y_line_width = 1

    nnt_data = _extract_duplicates_multi(evaluation_folders, sample_sizes_to_include=[sample_size])

    sns.lineplot(data=nnt_data, x='input', y='output', palette=colour_palette,
                 hue='model', style='model', linewidth=y_line_width, ax=ax1)

    # add a legend based on the different variables in the graph
    lines_legend = ax1.legend(loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2),
                              ncols=len(evaluation_folders.keys()),
                              frameon=False)

    # ensure that the marker size is small values, small size, large values, large size
    sizes_order = nnt_data['attribute disclosure'].to_list()
    sizes_order.sort(reverse=True)

    scatter = sns.scatterplot(data=nnt_data, x='input', y='output', palette=colour_palette,
                              size='attribute disclosure', sizes=(10, 100), size_order=sizes_order,
                              hue='model', markers=True, ax=ax1, legend='brief')

    scatter_handles, scatter_labels = scatter.get_legend_handles_labels()

    ax1.legend(handles=scatter_handles[scatter_labels.index('attribute disclosure') + 1:],
               labels=scatter_labels[scatter_labels.index('attribute disclosure') + 1:],
               loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
               title='Attribute disclosure\n          score', alignment='center', title_fontsize='small')

    # add separate legend for lines
    ax1.add_artist(lines_legend)

    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax1.set_xlabel(f'{x_label}\n generated from {sample_size} original samples', labelpad=10, loc='center')
    ax1.set_ylabel(y_label, labelpad=-20, loc='top', rotation='horizontal')

    # retrieve ticks
    x_ticks = ax1.get_xticks()

    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    if isinstance(title, str):
        ax1.set_title(title, y=1.1)

    # save when desired
    if save:
        if filename is None:
            filename = input(
                f'Please provide an appropriate filename including a path for the attribute x identity plot')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved statistical similarity plot for analysis of under\n{filename}')


def tabularise_utility(evaluation_folders, reference_data, file_key_word='_n_output.csv', visualise=True,
                       save_to_csv=True, csv_name=None,
                       univariate=False, effect_measure_identifiers=None, effect_measure=None):
    """
    Create a tabel of the utility of the synthetic data, i.e., determines the number of overlapping, shifted and
    newly significant associations

    :param evaluation_folders:
    :param reference_data:
    :param file_key_word:
    :param visualise:
    :param save_to_csv:
    :param csv_name:
    :param univariate:
    :param effect_measure_identifiers:
    :param effect_measure:
    """
    if isinstance(reference_data, str):
        reference_data = file_handling.read_json(reference_data)
    elif isinstance(reference_data, dict):
        pass
    else:
        warnings.warn(f'Cannot tabularise utility without reference data.\n'
                      f'please provide reference data that contains the same column names as the data to evaluate.')
        return None

    if effect_measure is None:
        effect_measure = 'odds ratio'

    if effect_measure_identifiers is None:
        effect_measure_identifiers = (f'0.95% confidence interval (lower)',
                                      f'{effect_measure}',
                                      f'0.95% confidence interval (upper)')

    if isinstance(reference_data, dict):
        if univariate:
            variables = {variable[:variable.rfind(effect_measure_identifiers[1])]: variable
                         for variable in reference_data.keys() if effect_measure_identifiers[1] in variable
                         and 'univariate' in variable}
        else:
            variables = {variable[:variable.rfind(effect_measure_identifiers[1])]: variable
                         for variable in reference_data.keys() if effect_measure_identifiers[1] in variable
                         and 'univariate' not in variable}

    _reference_data = {}
    # noinspection PyUnboundLocalVariable
    for variable in variables.keys():
        _reference_data.update({
            f'{variable}{effect_measure_identifiers[0]}': reference_data[
                f'{variable}{effect_measure_identifiers[0]}'],
            f'{variable}{effect_measure_identifiers[1]}': reference_data[
                f'{variable}{effect_measure_identifiers[1]}'],
            f'{variable}{effect_measure_identifiers[2]}': reference_data[
                f'{variable}{effect_measure_identifiers[2]}']})

    table_columns = ['model', 'sample_size',
                     'number_overlapping', 'per_cent_overlapping', 'overlapping',
                     'number_shifted', 'per_cent_shifted', 'shifted',
                     'number_of_new_associations', 'per_cent_new_associations', 'new_associations',
                     'c-statistic', 'adjusted c-statistic']
    table = pd.DataFrame(columns=table_columns)

    for name, data_directory in evaluation_folders.items():
        datasets = [file for file in os.listdir(data_directory)
                    if file_key_word in file]

        for data_to_read in datasets:
            # load dataset
            data_to_read = file_handling.read_csv(f'{data_directory}{os.path.sep}{data_to_read}',
                                                  datatype=float, remove='file_identification')

            column_to_check = [col for col in data_to_read.columns if 'n_output' in col]
            if len(column_to_check) > 1:
                logging.critical(f'Unexpected number of columns found containing {file_key_word}.\n'
                                 f'Consider using different keywords.')
                return None

            for sample in range(len(data_to_read.index)):
                # determine which scores e.g., odds ratios, fall in the confidence interval of the synthetic score
                overlaps = [variable for variable in variables.keys() if
                            data_to_read[f'{variable}{effect_measure_identifiers[0]}'].iloc[sample]
                            < _reference_data[f'{variable}{effect_measure_identifiers[1]}'] <
                            data_to_read[f'{variable}{effect_measure_identifiers[2]}'].iloc[sample]]

                # determine which scores e.g., odds ratios, have shifted effect and do not overlap with the real score
                shifts = [variable for variable in variables.keys() if not
                data_to_read[f'{variable}{effect_measure_identifiers[0]}'].iloc[sample]
                < _reference_data[f'{variable}{effect_measure_identifiers[1]}'] <
                data_to_read[f'{variable}{effect_measure_identifiers[2]}'].iloc[sample]

                          and ((_reference_data[f'{variable}{effect_measure_identifiers[1]}'] > 1 >
                                data_to_read[f'{variable}{effect_measure_identifiers[1]}'].iloc[sample])
                               or
                               (_reference_data[f'{variable}{effect_measure_identifiers[1]}'] < 1 <
                                data_to_read[f'{variable}{effect_measure_identifiers[1]}'].iloc[sample]))]
                # determine which associations have become significant but are not in the real data
                new_as = [variable for variable in variables.keys() if
                          (_reference_data[f'{variable}{effect_measure_identifiers[0]}']
                           <= 1 <=
                           _reference_data[f'{variable}{effect_measure_identifiers[2]}']
                           and not
                           data_to_read[f'{variable}{effect_measure_identifiers[0]}'].iloc[sample]
                           <= 1 <=
                           data_to_read[f'{variable}{effect_measure_identifiers[2]}'].iloc[sample])
                          or
                          (_reference_data[f'{variable}{effect_measure_identifiers[0]}']
                           <= 1 >=
                           _reference_data[f'{variable}{effect_measure_identifiers[2]}']
                           and not
                           data_to_read[f'{variable}{effect_measure_identifiers[0]}'].iloc[sample]
                           <= 1 >=
                           data_to_read[f'{variable}{effect_measure_identifiers[2]}'].iloc[sample])
                          ]

                _table = pd.DataFrame({table_columns[0]: f'{name}',
                                       table_columns[1]: int(data_to_read[column_to_check].iloc[sample]),
                                       table_columns[2]: len(overlaps),
                                       table_columns[3]: (len(overlaps) / len(variables)) * 100,
                                       table_columns[4]: str(overlaps),
                                       table_columns[5]: len(shifts),
                                       table_columns[6]: (len(shifts) / len(variables)) * 100,
                                       table_columns[7]: str(shifts),
                                       table_columns[8]: len(new_as),
                                       table_columns[9]: (len(new_as) / len(variables)) * 100,
                                       table_columns[10]: str(new_as),
                                       table_columns[11]: float(data_to_read['c-statistic'].iloc[sample]),
                                       table_columns[12]: float(data_to_read['adjusted c-statistic'].iloc[sample])},
                                      index=[len(table.index) + 1])

                table = pd.concat([table, _table])

    if save_to_csv:
        if univariate is True and csv_name is None:
            csv_name = 'univariable_utility_output.csv'
        elif csv_name is None:
            csv_name = 'multivariable_utility_output.csv'

        file_handling.save_csv(table, csv_name)

    if visualise:
        # extract and visualise overlapping effects
        _table = pd.melt(table, [table_columns[0], table_columns[1]],
                         [table_columns[3]])
        _table = _table.replace(0, np.nan)
        _visualise_utility(_table, y_label='Overlapping\nassociations\n')

        # extract and visualise shifted effects
        _table = pd.melt(table, [table_columns[0], table_columns[1]],
                         [table_columns[6]])
        _table = _table.replace(0, np.nan)
        _visualise_utility(_table, y_label='Shifted\nassociations\n')

        # extract and visualise new associations
        _table = pd.melt(table, [table_columns[0], table_columns[1]],
                         [table_columns[9]])
        _table = _table.replace(0, np.nan)
        _visualise_utility(_table, y_label='New\nassociations\n')

        if univariate is False:
            # extract and visualise adjusted concordance statistic
            _table = pd.melt(table, [table_columns[0], table_columns[1]],
                             [table_columns[11]])
            _table = _table.replace(0, np.nan)
            _visualise_utility(_table, y_label='Concordance\nstatistic\n')

            # extract and visualise adjusted concordance statistic
            _table = pd.melt(table, [table_columns[0], table_columns[1]],
                             [table_columns[12]])
            _table = _table.replace(0, np.nan)
            _visualise_utility(_table, y_label='Adjusted\nconcordance\nstatistic\n')


def visualise_loss(loss_file, title=None, x_label='Epochs', y_label='Loss', context=sns_context,
                   colour_palette=sns_colour_palette,
                   filename=None, save=True):
    """"""
    logging.info(f'Generating a similarity plot for multiple generators')

    # create the empty plot with general settings
    sns.set_context(context)
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()

    y_line_width = 1
    _y_line_style = ['-', '--', '-.', ':']

    _artificial_index = 0

    dataset = pd.melt(pd.read_csv(loss_file), id_vars=['Epochs'], var_name='Loss Type', value_name='Loss')

    sns.lineplot(data=dataset, x='Epochs', y='Loss', hue='Loss Type', style='Loss Type',
                 dashes={'Generator': (), 'Discriminator': (2, 2)},
                 linewidth=y_line_width, palette=colour_palette, ax=ax1)

    handles, labels = ax1.get_legend_handles_labels()

    # add a legend based on the different variables in the graph
    ax1.legend(handles=handles, labels=labels,
               loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2),
               ncols=dataset['Loss Type'].nunique(), frameon=False)

    ax1.set_xlabel(x_label, labelpad=10, loc='center')
    ax1.set_ylabel(y_label, labelpad=-20, loc='top', rotation='horizontal')

    # retrieve ticks and ensure there are no more than 10 for cleanliness
    x_ticks = ax1.get_xticks()
    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    # place the title above the legend in the top
    if isinstance(title, str):
        ax1.set_title(title, y=1.1)

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path for the m similarity '
                             f'plot for {title}')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved multiple generator statistical similarity plot for analysis of under\n{filename}')


def _extract_duplicates_multi(evaluation_folders, identity_disclosure='categorical duplicates',
                              attribute_disclosure='attribute disclosure min',
                              sample_sizes_to_include=None,
                              file_key_word=('n_input_', '_n_output.csv')):
    """"""
    # create output frame
    _data_frame_columns = ['input', 'output', 'attribute disclosure', 'input sample size', 'model']
    dataset = pd.DataFrame(columns=_data_frame_columns)

    for name, data_directory in evaluation_folders.items():
        datasets = [file for file in os.listdir(data_directory)
                    if file_key_word[0] in file and file_key_word[1] in file]

        for sample_size in sample_sizes_to_include:
            for data_to_read in datasets:
                if sample_size in data_to_read:
                    # load dataset
                    data_to_read = file_handling.read_csv(f'{data_directory}{os.path.sep}{data_to_read}',
                                                          datatype=float, remove='file_identification')
                else:
                    continue

                column_to_check = [col for col in data_to_read.columns if file_key_word[0] in col]
                if len(column_to_check) > 1:
                    logging.critical(f'Unexpected number of columns found containing {file_key_word[0]}.\n'
                                     f'Consider using different keywords.')
                    exit()
                else:
                    column_to_check = column_to_check[0]

                for sample in range(len(data_to_read.index)):
                    _dataset = pd.DataFrame({_data_frame_columns[0]: data_to_read[column_to_check].iloc[sample],
                                             _data_frame_columns[1]: (data_to_read[identity_disclosure].iloc[sample] /
                                                                      data_to_read[column_to_check].iloc[sample]) * 100,
                                             _data_frame_columns[2]: data_to_read[attribute_disclosure].iloc[sample],
                                             _data_frame_columns[3]: sample_size,
                                             _data_frame_columns[4]: f'{name}'},
                                            index=[len(dataset.index) + 1])

                    dataset = pd.concat([dataset, _dataset])

        dataset[_data_frame_columns[0]] = dataset[_data_frame_columns[0]].astype(int)
        dataset[_data_frame_columns[1]] = dataset[_data_frame_columns[1]].astype(float)

    return dataset


def _extract_statistical_similarity_multi(evaluation_folders, statistical_similarity_metric_to_extract=None,
                                          sample_sizes_to_include=None,
                                          file_key_word=('n_input_', '_n_output.csv')):
    """"""
    # create output frame
    _data_frame_columns = ['input', 'output', 'metric', 'input sample size', 'model']
    dataset = pd.DataFrame(columns=_data_frame_columns)

    for name, data_directory in evaluation_folders.items():
        datasets = [file for file in os.listdir(data_directory)
                    if file_key_word[0] in file and file_key_word[1] in file]

        for sample_size in sample_sizes_to_include:
            for data_to_read in datasets:
                if sample_size in data_to_read:
                    # load dataset
                    data_to_read = file_handling.read_csv(f'{data_directory}{os.path.sep}{data_to_read}',
                                                          datatype=float, remove='file_identification')
                else:
                    continue

                column_to_check = [col for col in data_to_read.columns if file_key_word[0] in col]
                if len(column_to_check) > 1:
                    logging.critical(f'Unexpected number of columns found containing {file_key_word[0]}.\n'
                                     f'Consider using different keywords.')
                    exit()
                else:
                    column_to_check = column_to_check[0]

                for metric in statistical_similarity_metric_to_extract:
                    for sample in range(len(data_to_read.index)):
                        _dataset = pd.DataFrame({_data_frame_columns[0]: data_to_read[column_to_check].iloc[sample],
                                                 _data_frame_columns[1]: data_to_read[metric].iloc[sample],
                                                 _data_frame_columns[2]: metric,
                                                 _data_frame_columns[3]: sample_size,
                                                 _data_frame_columns[4]: f'{name}'},
                                                index=[len(dataset.index) + 1])

                        dataset = pd.concat([dataset, _dataset])

        dataset[_data_frame_columns[0]] = dataset[_data_frame_columns[0]].astype(int)
        dataset[_data_frame_columns[1]] = dataset[_data_frame_columns[1]].astype(float)

    return dataset


def _find_nearest(array, value):
    """
    Retrieve the array value that is nearest to the given value
    Taken from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    :param numpy.ndarray array: array to inspect
    :param float value: value to find the closest array value for
    :return: the closest value
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _retrieve_best_sample(dataset, sample_size=None, account_for_privacy=False, metric='adjusted c-statistic',
                          key_word='n_output'):
    """
    Retrieve the sample that can be considered best with regard to a certain metric

    :param dataset:
    :param sample_size:
    :param account_for_privacy:
    :param metric:
    :param key_word:
    :return:
    """
    #
    if isinstance(sample_size, (int, float)):
        sample = dataset.loc[dataset[key_word] == sample_size]
        return sample

    elif account_for_privacy:
        # try to find the first row with disclosure breach, use the largest sample size if not present
        try:
            first_violation = dataset[dataset['categorical disclosure min'] == 0].index.values.astype(int)[0]
            try:
                highest_statistical_similarity_sample = dataset.loc[first_violation - 1]
            except KeyError:
                logging.critical(f'Provided dataset does not contain samples that preserve privacy,\n'
                                 f'No sample is considered the best in this scenario.')
                return None
        except IndexError:
            try:
                highest_statistical_similarity_sample = dataset.nlargest(n=1, columns=metric, keep='last')
            except KeyError:
                highest_statistical_similarity_sample = dataset.nlargest(n=1, columns=key_word, keep='last')

        return highest_statistical_similarity_sample

    else:
        try:
            highest_statistical_similarity_sample = dataset.nlargest(n=1, columns=metric, keep='last')
        except KeyError:
            highest_statistical_similarity_sample = dataset.nlargest(n=1, columns=key_word, keep='last')

        return highest_statistical_similarity_sample


def _retrieve_coefficients(repetition_paths, variables, sample_size, prefix, suffix):
    """
    Reads CSV files containing odds ratios
    and returns a list of dictionaries for each repetition for a specific sample size.

    :param list repetition_paths: List of paths to CSV files for each repetition.
    :param list variables: List of variables to construct the odds ratio column names.
    :param list sample_size: The specific sample size to extract odds ratios for.
    :param str prefix: The prefix that can be used to extract coefficients e.g., univariate
    :param str suffix: The suffix that can be used to extract coefficients e.g., odds ratio

    :return List of dictionaries containing coefficients for each repetition.
    """
    coefficients_set = []

    for repetition_path in repetition_paths:
        # Read the CSV file
        df = pd.read_csv(repetition_path)

        # Find the row index where "sample_size" column matches the target sample size
        row_index = df.index[df['n_output'] == sample_size].tolist()

        if not row_index:
            raise ValueError(f"No data found for sample size {sample_size} in {repetition_path}")

        row_index = row_index[0]  # Assuming there is a unique row for the given sample size

        # Dictionary to store coefficients for the current repetition
        coefficients_repetition = {}

        # Loop through variables and extract odds ratios from the specified row
        for variable in variables:
            # Construct the column name for odds ratios
            odds_ratio_column = f"{prefix}{variable}{suffix}"

            # Extract odds ratios from the specified column and row
            coefficients_repetition[variable] = df.at[row_index, odds_ratio_column]

        # Append the dictionary of beta coefficients for the current repetition to the main list
        coefficients_set.append(coefficients_repetition)

    return coefficients_set


def _rescale(value_to_scale, minimum, maximum):
    """
    Rescale a sequence of values on a zero to one scale

    :param integer or float value_to_scale: number to scale
    :param int minimum: the minimal value to scale to
    :param int maximum: the maximum value to scale to
    :return: rescaled sequence of numbers
    """
    # return the input if value is not integer or float
    try:
        value_to_scale = float(value_to_scale)
        minimum = float(minimum)
        maximum = float(maximum)
    except ValueError:
        return value_to_scale

    # return scaled values
    return (value_to_scale - minimum) / (maximum - minimum)


# noinspection SpellCheckingInspection
def _visualise_utility(dataset, x_label='Synthetic dataset sample size', y_label='Overlapping\neffects\n', title=None,
                       context=sns_context, colour_palette=sns_colour_palette,
                       filename=None, save=True):
    """
    Create a plot that visualises the utility of the data through the number of overlapping effects; determined in
    tabularise utility

    :param pandas.DataFrame dataset: dataframe representing the tabularised utility
    :param str x_label: label for the x-axis
    :param str y_label: label for the y-axis
    :param str title: provide a title for the plot
    :param str context: specify seaborn context, defaults to 'paper'
    :param str colour_palette: specify seaborn colour palette, defaults to 'colorblind'
    :param str filename: specify the filename if the plot is to be saved
    :param bool save: specify whether to save the plot
    """
    logging.info(f'Generating a utility plot for multiple models')

    # create the empty plot with general settings
    sns.set_context(context)
    fig, ax1 = plt.subplots(figsize=(10, 5.625))
    sns.despine()
    y_line_width = 1

    sns.lineplot(data=dataset, x='sample_size', y='value', palette=colour_palette,
                 markers=True, style='model', hue='model', linewidth=y_line_width, ax=ax1)

    handles, labels = ax1.get_legend_handles_labels()

    # add a legend based on the different variables in the graph
    lines_legend = ax1.legend(handles=handles, labels=labels,
                              loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2),
                              ncols=dataset['model'].nunique(), frameon=False)

    ax1.add_artist(lines_legend)

    ax1.set_xlabel(x_label.replace('_', ' '), labelpad=10, loc='center')
    ax1.set_ylabel(y_label, labelpad=-20, loc='top', rotation='horizontal')

    # first letter excluded to prevent errors with capital letters
    if 'oncordance' not in y_label:
        ax1.set_ylim(0, 100)
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    else:
        ax1.set_ylim(0, 1)

    # retrieve ticks and limit to 10 if necessary
    x_ticks = ax1.get_xticks()
    if len(x_ticks) >= 11:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    if isinstance(title, str):
        ax1.set_title(title, y=1.1)

    # save when desired
    if save:
        if filename is None:
            filename = input(f'Please provide an appropriate filename including a path the plot of {y_label}')
        plt.savefig(f'{filename}', dpi=350)
        logging.info(f'Saved statistical similarity plot for analysis of under\n{filename}')
