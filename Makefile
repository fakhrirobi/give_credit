.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = give_me_credit
PYTHON_INTERPRETER = python3  
BASE_DATA_DIR = 
ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: 
	$(PYTHON_INTERPRETER) src/data/make_dataset.py --raw_input_path=data/raw/cs-training.csv \
	 --interim_output_path=data/interim/interim_training_5_exp_tuned.csv \
	 --processed_output_path=data/processed/processed_training_5_exp_tuned.csv \
	  --experiment_name=fourth_exp_tuned --training_req=True
	$(PYTHON_INTERPRETER) src/data/make_dataset.py --raw_input_path=data/raw/cs-test.csv \
	--interim_output_path=data/interim/interim_5_forth_exp_tuned.csv \
	--processed_output_path=data/processed/processed_5_forth_exp_tuned.csv \
	 --experiment_name=fourth_exp_tuned --training_req=False
## Delete all compiled Python files
prepare_score_card_data : 
	$(PYTHON_INTERPRETER) src/score_card/prepare_data.py --raw_input_path=data/raw/cs-training.csv \
       							   --processed_output_path=data/processed/scorecard_training_processed.csv \
								   --experiment_name=scorecard_exp --training_req=True \
								   --interim_output_path=data/interim/scorecard_training_interim.csv

	$(PYTHON_INTERPRETER) src/score_card/prepare_data.py --raw_input_path=data/raw/cs-test.csv \
								--processed_output_path=scorecard_test_processed.csv \
								--experiment_name=scorecard_exp --training_req=False \
								--interim_output_path=data/interim/scorecard_test_interim.csv

train_scorecard : 
	$(PYTHON_INTERPRETER) src/score_card/train_logistic_model.py --training_data_path=data/processed/scorecard_training_processed.csv \
                                                                --experiment_name=scorecard_exp

create_scorecard : 
	$(PYTHON_INTERPRETER) src/score_card/turn_scorecard.py --score_data=data/processed/scorecard_training_processed.csv \
                                                                --filename=scoring
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	black .
tracking_server : 
	mlflow server --backend-store-uri sqlite:///mlflow.db --backend-store-uri ./mlruns
## Lint using flake8
lint:
	flake8 src --ignore=E501,E712

tuning : 
	$(PYTHON_INTERPRETER) src/models/param_tuning.py \
	--experiment_name=tuning \
	--training_data_path=data/processed/processed_training_5_exp_tuned.csv --num_trials=100
train : 
	$(PYTHON_INTERPRETER) src/models/train_model.py  \
	--experiment_name=exp_class_weight --training_data_path=data/processed/processed_training_5_exp_tuned.csv \
	--config_path=src/experiment_config/exp_class_weight.yaml

predict_batches_sample : 
	$(PYTHON_INTERPRETER) src/models/predict_model_batch.py  \
	--input_path=data/processed/processed_test_forth_exp_tuned.csv \
	--output_path=data/prediction/prediction_fixed_test_imputation.csv \
	--model_path=models/LGBMClassifier_fourth_exp_tuned.joblib

predict_single_sample : 
	$(PYTHON_INTERPRETER) src/models/predict_model_single.py  \

predict_single_sample : 
	$(PYTHON_INTERPRETER) src/score_card/predict_single_scorecard.py \      
	--utilization_rate=0.5 \
	--age=20 \
	
      "number30_59daysdue": 0,
      "debtratio": 0.35,
      "monthlyincome": 10000,
      "numopencredit_loans": 10,
      "number90dayslate": 3,
      "numberrealestate_loans": 2,
      "number60_89daysdue": 20,
      "numof_dependents": 3 \
	--
	 
run_api : 
	$(PYTHON_INTERPRETER) src/api/app.py

curl_api_sample : 
	curl -X 'POST' \
  'http://127.0.0.1:5000/predict_single' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_input": [ \
    { 
      "customer_id": "ID501",
      "utilization_rate": 0.5,
      "age": 20,
      "number30_59daysdue": 0,
      "debtratio": 0.35,
      "monthlyincome": 10000,
      "numopencredit_loans": 10,
      "number90dayslate": 3,
      "numberrealestate_loans": 2,
      "number60_89daysdue": 20,
      "numof_dependents": 3 \
    }, \
    { 
      "customer_id": "ID501",
      "utilization_rate": 0.5,
      "age": 20,
      "number30_59daysdue": 0,
      "debtratio": 0.35,
      "monthlyincome": 10000,
      "numopencredit_loans": 10,
      "number90dayslate": 3,
      "numberrealestate_loans": 2,
      "number60_89daysdue": 20,
      "numof_dependents": 3 \
    } \
  ] }'
## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/app.py
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
