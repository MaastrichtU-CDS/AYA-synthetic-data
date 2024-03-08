# AYA-synthetic-data

**What does this repository represent?**

This repository contains the research code and scripts used for
an investigation of the role of sample size in synthetic data.
The code in this repository was specifically designed to investigate the effects of
variation in input (training data) and output (produced synthetic data) sample size on synthetic data veracity, privacy
concealment, and utility.
_A more extensive description of the methodology that this repository represents can be found in the associated
scientific publication: currently as preprint on MedRxiv; https://doi.org/10.1101/2024.03.04.24303526_

**Where have the contents of this repository been used and reported?**

The role of sample size was investigated in a rare and heterogeneous healthcare demographic:
adolescents and young adults with cancer.
The findings of this investigation can be found in the associated scientific publication:

**Can this code be re-used to investigate sample size effects in other demographics or datasets?**

A large proportion of this code should be re-usable with another single-table dataset
(i.e., not time-series or multi-table datasets), given that the dataset is appropriately cleaned.
However, certain components such as _data_preprocessing.py_, _evaluation_visualisation.py_ and
the utility assessment in _evaluation_metric.py_ were specifically designed for the aforementioned dataset and
publication.

**How are the code and scripts in this repository to be used?**

There is a worked-out example provided in the _example_exercise.ipynb_ Jupyter notebook.
This example makes use of a public dataset on paediatric bone marrow transplantation developed by ... that is available
through:

**How was this work funded?**

This work and the associated scientific publication were predominantly
supported by the European Union’s Horizon 2020 research and innovation programme through _The STRONG-AYA Initiative_
(Grant agreement ID: 101057482).

**What are the main libraries that this research code relied on?**

The synthetic data was generated using

* _Synthetic Data Vault_ (SDV) (https://github.com/sdv-dev/SDV), and
* _Differentially Private - Conditional Generative Adversarial Networks_ (
DP-CGAN) (https://github.com/sunchang0124/dp_cgans).

The evaluations were performed using:

* _prdc_ (https://github.com/clovaai/generative-evaluation-prdc),
* _scipy_ (https://github.com/scipy/scipy),
* _SDmetrics_ (https://github.com/sdv-dev/SDMetrics),
* _sklearn_ (https://github.com/scikit-learn/scikit-learn), and
* _statsmodels_ (https://github.com/statsmodels/statsmodels/).

Versions of all necessary libraries can be found in the _requirements.txt_ file
_Please note that the second branch that DP-CGAN was developed in requires slightly different versions for some
libraries_