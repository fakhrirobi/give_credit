Give Me Some Credit
==============================
### Business Objective :
Before Applying Credit to Some Finance Creditor. Creditor Will Have an Assesment of The Risk of Debitor. 




In This Kaggle Competition we are tasked to predict the probabilities that somebody will experience financial distress(Delinquency) in the next 2 Years. 

According to Gartner 
![Gartner](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/gartner-model.png)




Objective the Task is Concluded as Predictive Task.


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
2. Inference API 


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
        a.MonthlyIncome () --> LogIncome()
        b.RevolvingUtilizationOfUnsecuredLines () --> LogRevolvingUtilizationOfUnsecuredLines()
        c.DebtRatio() -> LogDebtRatio()
    3. In terms of CV AUC 
        Base () --> After Feature Engineering ()
### Model Comparison / Decision 
    Insert Table Model Comparison 
    From the Table Above LGBMClassifier yield the best models. 
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
                    "lambda_l1": 2.8611070127189538e05 
                    "lambda_l2": 0.6807987664418726 
                    "num_leaves": 9 
                    "max_depth": 19 
                    "feature_fraction": 0.5332191589953184 
                    "bagging_fraction": 0.9238892410770332 
                    "bagging_freq": 3 
                    "min_child_samples": 73
        }
    With Average 5-Fold CV AUC -> 0.8656. Improvement from untuned models (AUC : x )



    ```



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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Findings 
------------



## How to Reproduce this Project
Main Functionality 
------------

Introduction 

### 1. Preparing Dataset 
    ```

    ```

### 2. Training Model 
    ```

    ```


### 3. Hyperparameter Tuning
    ```

    ```

### 4. Inference
    ```

    ```
### 5. Tracking Experiment Dashboard 
    ```
    make tracking_server
    ```



## Reference 
---
1. Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.
2. 