import sys

import src.evaluation_general

# retrieve path, model, and analysis to run
directory = sys.argv[1]
model = sys.argv[2]
analysis = sys.argv[3]

if model in ['CopulaGAN','CTGAN', 'TVAE', 'DP-CGAN']:
    if analysis == 'epochs':
        # initiate evaluation
        evaluation = src.evaluation_general.DataEvaluation(directory=directory)
        evaluation.generator_evaluate_epochs(models_to_evaluate=[model])

    if analysis == 'embedding_dim':
        # initiate evaluation
        evaluation = src.evaluation_general.DataEvaluation(directory=directory)
        evaluation.generator_evaluate_embedding_dim(models_to_evaluate=[model])

    if analysis == 'batch_size':
        # initiate evaluation
        evaluation = src.evaluation_general.DataEvaluation(directory=directory)
        evaluation.generator_evaluate_batch_size(models_to_evaluate=[model])

    if analysis == 'log_frequency':
        # initiate evaluation
        evaluation = src.evaluation_general.DataEvaluation(directory=directory)
        evaluation.generator_evaluate_log_frequency(models_to_evaluate=[model])

if model in ['GaussianCopula', 'CopulaGAN']:
    if analysis == 'distribution':
        # initiate evaluation
        evaluation = src.evaluation_general.DataEvaluation(directory=directory)
        evaluation.generator_evaluate_distribution(models_to_evaluate=[model])

if analysis == 'n_output':
    # initiate evaluation
    evaluation = src.evaluation_general.DataEvaluation(directory=directory)
    evaluation.generator_evaluate_n_output(models_to_evaluate=[model], save_model=True, default_model=True)

if analysis == 'n_input':
    # initiate evaluation
    evaluation = src.evaluation_general.DataEvaluation(directory=directory)
    evaluation.generator_evaluate_n_input_random(models_to_evaluate=[model], default_model=True)

exit()