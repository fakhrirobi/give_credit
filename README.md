Give Me Some Credit
==============================
### Business Objective :
Before Applying Credit to Some Finance Creditor. Creditor Will Have an Assesment of The Risk of Debitor. 




In This Kaggle Competition we are tasked to predict the probabilities that somebody will experience financial distress(Delinquency) in the next 2 Years. 

According to Gartner 
![Gartner](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/gartner-model.png)




Objective the Task is Concluded as Predictive Task.

Delivered Value : Reduce Bad Loans and Minimize Risk 

---
Next we are moving to the context of its dataset . 
The Dataset Consists of : 
1. Training Set 
   - Column Number : 11 
   - Row Number : 150.000
   - Column Name : 
  
       'SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents'
2. Test Set 
   - Column Number  : 
   - Row Number : 
   - Column Name : 
  
       'SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents'
To know what each column represent here is the column dictionary. 



---
### Column Dictionary
![Column Dictionary](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/column_dictionary.PNG)

### Metrics 
Metrics  used in this competition is Area Under The Receiving Operator Curve. 
Receiving Operating Curve consists of False Positive Rate (FP) as  x-axis and True Positive Rate (TP) as y-axis. This curve show sensitivity of the classifier as how many correct positive classification as adding more false positive prediction. 
![RO Curve](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/ROC.PNG)


The Values are ranging from 0 to 1. 
The Closer to 1 Mean the Model --> if it closer to 0 means 


This Metrics is suitable for dataset that has unbalanced class (such as fraud / default detection case) whose positive class is small percentage from all class. 


## Expected Product 
1. Kaggle Submission 

2. Inference API with FastAPI
---
## Library/Package Used 
1. API : 
   - FastAPI 
   - pydantic
2. Tracking  : 
   - MLFlow 
3. CLI : 
   - argparse
4. Data Processing 
   - Pandas
   - Numpy
5. Modelling 
   - scikit-learn
   - lightgbm

---
## Project Steps 
### Data Preprocessing 
    1. Dropping Duplicates 
    2. Filling Missing Values : 
        a. MonthlyIncome -> Imputed with Median 
        b. NumberOfDependents -> Imputed with Mode 
    

### Exploratory Data Analysis
    Explain All Findings 
### Feature Engineering
    1. Log1p Transformation : 
        a. LogIncome -> log1p(MonthlyIncome)
        b. LogRevolvingUtilizationOfUnsecuredLines -> log1p(RevolvingUtilizationOfUnsecuredLines)
        c.LogDebtRatio -> log1p(DebtRatio)
    2.Correlation Improvement 
        a.MonthlyIncome (-0.017151) --> LogIncome(-0.017617)
        b.RevolvingUtilizationOfUnsecuredLines (-0.001802) --> LogRevolvingUtilizationOfUnsecuredLines(0.178767)
    2. In terms of CV AUC 
        Base (0.86458) --> After Feature Engineering (0.86458)
### Model Comparison / Decision 
![AUC](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/auc_score_5_models.PNG)
![Fitting Time](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/fitting_time.PNG)

According to Cross Validated AUC and time to fit the model -> we choose LightGBM as our model and will continue to Hyperparameter Tuning. 

Technically There are two features that makes LightGBM run faster : 
1. Exclusive Feature Bundling 
2. Gradient Based One Side Sampling 
### Hyperparameter Tuning 
    I used optuna package to run some hyperparameter tuning 
    with option 

    ```
            param = {
            "objective": "binary", --> our task is binary classification
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True), --> L1 Regularization 
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True), --> L2 Regularization 
            "num_leaves": trial.suggest_int("num_leaves", 2, 256), --> number of maximal leaves (sum of all leaf)
            "max_depth": trial.suggest_int("max_depth", 2, 30), --> max depth of tree
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300), --> minimal sample before splitting into below child 
        }


    ```
        ```
            param = {
                    "bagging_fraction": 0.8462612247644394
                    "bagging_freq": 1
                    "feature_fraction": 0.47645733210236857
                    "lambda_l1": 1.0570508475331433
                    "lambda_l2": 1.0897731039221187e-07
                    "max_depth": 5
                    "min_child_samples": 205
                    "num_leaves": 255
                    }
        ```
    With Average 5-Fold CV AUC -> 0.866022. Improvement from untuned models (AUC : 0.864567 )
## Feature Contribution  
We aim to answer which variable contribute more to this model. There are two options to find model feature importance in this case : 
1. Permutaion Importance 
2. and for Tree based model usually there is Feature Importance 

Permutaian Importance :
![Permutation Importance](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/permutation_importance.png)

![Permutation Importance DataFrame](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/mean_importance.PNG)

defined as a decrease in model performance if a single feature value is randomly shuffled. Permutation Importance offer model agnoticness or it doesnot affected by model itself. 


Feature Importance (LightGBM) : 
![Feature Importance](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/LGBM_Feature_Importance.png)
In Tree Models there is a term about impurity ,which described as probability of misclassification (classification task.). The Decision to split to the next node is based on impurity which can be calculated with gini/entropy / etc. 
To calculate feature importance we need to calculate Mean Decrease Impurity (MDI). In simple Terms the Feature Importance rank features based on its feature split gain the least impurity. 

However in Feature Importance method if the features contain high cardinality values it will biased the feature importance and feature importance are measured on training statistics and not unseen data. 

We can see the difference between permutation importance and feature importance variable : 
In Permutation importance LogRevolvingUtilizationofUnsecuredLines will affect the model auc the most if shuffled. On the other hand LogDebtRatio gain the highest feature importance.



## Result 
![Kaggle Submission](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/best_submission.PNG)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── feature_eng.py
    │   │   └── validation_features.py <- 
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model_batch.py <- to generate prediction batches (.csv) file 
    │   │   ├── predict_model_single.py <- to generate prediction on single inference (CLI)
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------







## How to Reproduce Machine Learning Training 
This Repository Aim to track every experimentation (including retraining). Hence each experiment is tracked with mlflow. 
In order to create experiment. Several step needs to be done 

1. Clone this Project 
   ```
   git clone https://github.com/fakhrirobi/give_credit.git
   cd give_credit
   ```
   ---
2. Prepare the Environment 
   I recommend to learn the basic of Makefile
   make command : 
    ```
    make requirements
    ```
    or if you have not used makefile
    ```
    python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt -->install all dependency

    ```
    ---
3. Create Configuration File(*.yaml)
   Folder to locate the experiment_config is in [src/experiment_config](https://github.com/fakhrirobi/give_credit/tree/main/src/experiment_config) folder
   The Contain of Experiment Config are 
   ```
    params : --> params for the model we are going to train -> if you want to train base model leave it blank. this params i usually yield from parameter tuning 
        lambda_l1: 2.8611070127189538e05 
        lambda_l2: 0.6807987664418726 
        num_leaves: 9 
        max_depth: 19 
        feature_fraction: 0.5332191589953184 
        bagging_fraction: 0.9238892410770332 
        bagging_freq: 3 
        min_child_samples: 73

    feature_engineering :  --> explanation of feature engineering step like new column plus its explanation 
    LogDebtRatio : {
        feature_source : DebtRatio,
        description : 
                The form log np.log1p(DebtRatio)
    }
    LogIncome : {
        feature_source : MonthlyIncome,
        description : 
                The form log np.log1p(MonthlyIncome)
    }
    log_RevolvingUtilizationOfUnsecuredLines : {
        feature_source : RevolvingUtilizationOfUnsecuredLines,
        description : 
                The form log np.log1p(RevolvingUtilizationOfUnsecuredLines)
    }


    preprocessing_step : --> explaininig preprocessing step in data wrangling such as imputation 
        NumberOfDependents : {
        feature_source : NumberOfDependents,
        description : 
                Filling Missing Values based on median value on training data 
    }
        MonthlyIncome : {
        feature_source : MonthlyIncome,
        description : 
                Filling Missing Values based on median value on training data 
    }

    data_wrangling_output_dtypes : --> list of columns name that supposted to be yielded with its acceptable datatypes (for data validation) (required)
        age: ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberOfTime30-59DaysPastDueNotWorse : ['int64','int32','int16','int8','float64','float32','float16','float8'] 
        MonthlyIncome : ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberOfOpenCreditLinesAndLoans : ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberOfTimes90DaysLate: ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberRealEstateLoansOrLines : ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberOfTime6089DaysPastDueNotWorse : ['int64','int32','int16','int8','float64','float32','float16','float8']
        NumberOfDependents : ['int64','int32','int16','int8','float64','float32','float16','float8']
        DebtRatio :  ['int64','int32','int16','int8','float64','float32','float16','float8']
        RevolvingUtilizationOfUnsecuredLines :  ['int64','int32','int16','int8','float64','float32','float16','float8']


    
    feature_eng_model_input : --> model input column / feature engineering result and its dtypes(required)
        {age: ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberOfTime30-59DaysPastDueNotWorse : ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberOfOpenCreditLinesAndLoans : ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberOfTimes90DaysLate: ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberRealEstateLoansOrLines : ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberOfTime6089DaysPastDueNotWorse : ['int64','int32','int16','int8','float64','float32','float16','float8'],
        NumberOfDependents : ['int64','int32','int16','int8','float64','float32','float16','float8'],
        LogDebtRatio :  ['int64','int32','int16','int8','float64','float32','float16','float8'],
        log_RevolvingUtilizationOfUnsecuredLines :  ['int64','int32','int16','int8','float64','float32','float16','float8'],
        LogIncome  : ['int64','int32','int16','int8','float64','float32','float16','float8']}

        Tips : The Name of Config file better to be the name of experiment to easier tracking


   ```
   ---

2. Prepare The Dataset 
   Execute python file in [src/data/make_dataset.py](https://github.com/fakhrirobi/give_credit/tree/main/src/data/make_dataset.py)
   This process only create single training or test data not both. 
   This python file needs CLI argument : 

   a. --experiment_name = the name of experiment (make sure its consistent)

   b. --raw_input_path = the source of data (.csv file). Default in  [give_credit/data/raw](https://github.com/fakhrirobi/give_credit/tree/main/data/raw) Folder

   c. --interim_output_path = the output file (.csv) to store wrangled dataset (optional)

   d. --processed_output_path = the processed output file (feature engineered) data and ready for training (modelling) / inference (test_data)

   e. --training_req = wheter or not the process is for creating training data or test data . True means preparing for Training Data. False mean for test data. The difference is that The Training Data Contains Labels.

    The Example can be seen in [Makefile](https://github.com/fakhrirobi/give_credit/blob/main/Makefile) in "data" command. 

    Notes : to change the data wrangling you need to modify in [src/data/wrangling.py](https://github.com/fakhrirobi/give_credit/blob/main/src/data/wrangling.py) the same apply to [feature engineering](https://github.com/fakhrirobi/give_credit/blob/main/src/features/feature_eng.py) processand dont forget to update the experiment_config.yaml

   ---

3. Train the Model 
    After Preparing The Dataset you can move to the model 
    to train model you can run the [src/models/train_model.py](https://github.com/fakhrirobi/give_credit/blob/main/src/models/train_model.py)

    This python file needs CLI argument : 

    a. --experiment_name = the name of experiment (make sure its consistent)

    b. --training_path = the source of training datadata (.csv file). Default in  [give_credit/data/processed](https://github.com/fakhrirobi/give_credit/tree/main/data/processed) Folder
    c. --config_path = The config.yaml file contain information of the process. This file later will be stored and logged as artifact in mlflow. 

    The Trained model willbe stored as {Modelname}_{experiment_name}.joblib file in [give_credit/models](https://github.com/fakhrirobi/give_credit/tree/main/models) folder. 

    The Example can be seen in [Makefile](https://github.com/fakhrirobi/give_credit/blob/main/Makefile) in "training" command. 

    ---
4. Tune the Model 
    You can run  [src/models/param_tuning.py](https://github.com/fakhrirobi/give_credit/blob/main/src/models/param_tuning.py)

    This python file needs CLI argument : 

    a. --experiment_name = the name of experiment (make sure its consistent)

    b. --training_data_path = the source of training datadata (.csv file). Default in  [give_credit/data/processed](https://github.com/fakhrirobi/give_credit/tree/main/data/processed) Folder
    c. --num_trials = Number of study in optuna. usually 100

    After Tuning the Params the tuning properties such as params metric and best_params will be logged to each runned tuning. 

    to access best params in local you can open folder [src/models/tuning_result](https://github.com/fakhrirobi/give_credit/blob/main/src/models/tuning_result). 

    The Example can be seen in [Makefile](https://github.com/fakhrirobi/give_credit/blob/main/Makefile) in "tuning" command. 

    ---

5. Show All Experiment Tracking 
   in cmd run 
   ```
    mlflow server --backend-store-uri sqlite:///mlflow.db --backend-store-uri ./mlruns
   ```
   or just run make : 
    ```
    make tracking_server
    ```
    If after cloned this repository you find all artifact was lost dont be panic. This because mlflow still implement absolute path instead of relative path. To restore it 
    open [mlruns/0](https://github.com/fakhrirobi/give_credit/tree/main/mlruns/0) open the folder and you will find meta.yml . After that change artifact_uri key from 
    ```
    artifact_uri:file:///home/fakhri/pacmann_project/give_credit/mlruns/<some_runid>
    to 
    artifact_uri:../give_credit/mlruns/<some_runid>
    ```
    and run tracking server again

    ---
6. Serving on API 
   I Created an API with FastAPI. Inspired with madewithml example. 
   The API itself for now only contain /predict_single endpoint.
   to run the api just run [src/api/app.py](https://github.com/fakhrirobi/give_credit/tree/main/src/api/app.py)

    after run the file check localhost:5000/docs to see detailed documentation. 
    The API POST method require input 
    ```
    {model_input": [
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
      "numof_dependents": 3
    }]}
    ```

    and the responses are :
    ``` 
    {
    "message": "OK",
    "method": "POST",
    "status-code": 200,
    "timestamp": "2023-01-11T00:43:21.830087",
    "url": "http://127.0.0.1:5000/predict_single",
    "data": [
        {
        "data": {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 20,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.35,
            "MonthlyIncome": 10000,
            "NumberOfOpenCreditLinesAndLoans": 10,
            "NumberOfTimes90DaysLate": 3,
            "NumberRealEstateLoansOrLines": 2,
            "NumberOfTime60-89DaysPastDueNotWorse": 20,
            "NumberOfDependents": 3,
            "customer_id": "ID501"
        },
        "output": {
            "proba": [
            [
                0.39221000427500563,
                0.6077899957249944
            ]
            ],
            "label": [
            1
            ]
        }
        },
    ```
    Tips : When Constructing JSON Response if you have inference result such as proba in numpy types convert it as list first cause it wont accept numpy. 



## Reference 
---
1. Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.
2. https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
3. https://scikit-learn.org/stable/modules/permutation_importance.html
4. https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3
5. Pacmann Materials 


