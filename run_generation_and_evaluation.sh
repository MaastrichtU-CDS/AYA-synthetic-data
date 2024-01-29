#!/bin/sh

ask()
{
  declare -g "$1"="$2"
  if [ -z "${!1}" ]; then
    echo "$3"
    read "$1"
  fi
}

ask DATA_PATH "$1" "Provide the directory where the data and it's metadata resides"
ask MODEL   "$2" "Provide the model abbreviation of the generative model that is to be used"

REPO_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#EVALUATIONS=("epochs embedding_dim batch_size log_frequency distribution n_output n_input")
EVALUATIONS=("n_output n_input")

if [ "$MODEL" == $"DP-CGAN" ]; then
  cd "$REPO_DIR" || exit
  cd ..
  if ! source dp_cgan_env/bin/activate ; then
    if ! python3 -m venv dp_cgan_env ; then
      echo "Install python3-venv via: apt-install python3-venv"
      exit
    fi
    source dp_cgan_env/bin/activate
  fi
  cd "$REPO_DIR"
  pip install -r requirements.txt
else
  cd "$REPO_DIR" || exit
  cd ..
  if ! source main_env/bin/activate ; then
    if ! python3 -m venv main_env ; then
      echo "Install python3-venv via: apt-install python3-venv"
      exit
    fi
    source main_env/bin/activate
  fi
  cd "$REPO_DIR"
  pip install -r requirements.txt
fi

for _EVALUATION in $EVALUATIONS
do
  python3 run_generation_and_evaluation.py "$DATA_PATH" "$MODEL" "$_EVALUATION"

  if [[ -z $(grep '[^[:space:]]' "$REPO_DIR"/log ) ]] ; then
    echo "Evaluation ""$_EVALUATION"" for model ""$MODEL"" was not performed or logged."

    else
    mkdir -p "$DATA_PATH""$MODEL"_analyses/logs/
    mv "$REPO_DIR"/log  "$DATA_PATH""$MODEL"_analyses/logs/log_"$_EVALUATION"
  fi

  if [[ -z $(grep '[^[:space:]]' "$REPO_DIR"/loss_output* ) ]] ; then
    echo "Loss ""$_EVALUATION"" for model ""$MODEL"" was not determined or logged."

    else
    mkdir -p "$DATA_PATH""$MODEL"_analyses/logs/
    mv "$REPO_DIR"/loss_output*  "$DATA_PATH""$MODEL"_analyses/logs/loss_"$_EVALUATION"
  fi

  if [ "$(ls -A "$DATA_PATH"evaluation/*)" ]; then
    mkdir -p "$DATA_PATH""$MODEL"_analyses/evaluation/"$_EVALUATION"/
    mv "$DATA_PATH"evaluation/* "$DATA_PATH""$MODEL"_analyses/evaluation/"$_EVALUATION"/
  fi

  if [ "$(ls -A "$DATA_PATH"synthetic/*)" ]; then
    mkdir -p "$DATA_PATH""$MODEL"_analyses/synthetic/"$_EVALUATION"/
    mv "$DATA_PATH"synthetic/* "$DATA_PATH""$MODEL"_analyse
    synthetic/"$_EVALUATION"/
  fi

  if [ "$(ls -A "$DATA_PATH"models/*)" ]; then
    mkdir -p "$DATA_PATH"/"$MODEL"_analyses/models/"$_EVALUATION"/
    mv "$DATA_PATH"models/* "$DATA_PATH""$MODEL"_analyses/models/"$_EVALUATION"/
  fi

  if [[ -z $(grep '[^[:space:]]' "$REPO_DIR"/nohup* ) ]] ; then
    echo "Terminal output ""$_EVALUATION"" for model ""$MODEL"" was not saved or is unavailable."

    else
    mkdir -p "$DATA_PATH""$MODEL"_analyses/logs/
    mv "$REPO_DIR"/nohup*  "$DATA_PATH""$MODEL"_analyses/logs/terminal_output_"$_EVALUATION"
  fi

done

deactivate
