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
3. Credit Scorecard
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
<details>
    <summary></summary>
    <ol>
    <li>
      <a href="#data-preprocessing">Data Preprocessing</a>
    </li>
    <li>
      <a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
    </li>
    <li>
      <a href="#feature-engineering">Feature Engineering</a>
    </li>
    <li>
      <a href="#feature-contribution">Feature Engineering</a>
    </li>
    <li><a href="#model-evaluation">Model Evaluation and Calibration</a></li>
    <li><a href="#scorecard">Credit Scorecard</a></li>
  </ol>
</details>

### Data Preprocessing 
    1. Dropping Duplicates 
    2. Filling Missing Values : 
        a. MonthlyIncome -> Imputed with Median 
        b. NumberOfDependents -> Imputed with Mode 
    

### Exploratory Data Analysis
EDA can be found [here](https://github.com/fakhrirobi/give_credit/blob/main/notebooks/score_card_from%20scratch.ipynb)

---
### Feature Engineering
 1. Binning Feature 
     Since this goal is to create credit scoring we need to use binning in order to calculate weight of evidence . Here is my [binning](https://github.com/fakhrirobi/give_credit/blob/main/assets/binning.txt)
     From the Credit Risk Scorecard Book by Naeem Siddiqi its better to create many bins in order to create diverse credit score. 

 2. After the binning process is done now we are going to calculate Weight of Evidence 
   
   $$ WOE = ln\left(\frac{P(\text{class 0 (non events) outcome})}{P(\text{class 1 (events)outcome})}\right) $$
   

    Explanation : 
    WOE of a bin from a feature is calculated by calculating the log of distribution of of non events divided by events in a bin from a feature .The context of distribution is from all available bin in the features. 
    
    Python Implementation : 
    ```
    def create_woe_reference(feature,binned_data) :
        crosstab_data = (pd.crosstab(binned_data[feature],
                    binned_data[TARGET],rownames=[feature], 
                    colnames=[TARGET])
                    .reset_index()
                    )
        total_0 = np.sum(crosstab_data[0])
        total_1 = np.sum(crosstab_data[1])

        crosstab_data[0] = crosstab_data[0]/total_0
        crosstab_data[1] = crosstab_data[1]/total_1

        crosstab_data[f'{feature}_woe'] = np.log(crosstab_data[0]/crosstab_data[1])
        crosstab_data[f'{feature}_iv'] = (crosstab_data[0]-crosstab_data[1])*crosstab_data[f'{feature}_woe']
        
        temp  = {}
        for x,y,z in zip (crosstab_data[feature],
                    crosstab_data[f'{feature}_woe'],crosstab_data[f'{feature}_iv']
                    ) : 
            
            temp[x] = {'woe':y,'iv':z}

        return temp
    ```

WoE of each features can be seen [here](https://github.com/fakhrirobi/give_credit/blob/main/assets/woe_values.txt) 

3. Feature Selection Using Information Value 
   $$IV = ∑\left(P(\text{class 0 (non events) outcome}) - P(\text{class 1 (events) outcome})\right) * WOE


to simplify IV of a feature is a sum of bin  the distribution margin of class 0 (non-events) and class 1 (events) multiplied by its WoE value of bin.



The category of IV score can be explained as :

if IV 
- less than 0.02 -> feature is unpredictive 
- 0.02 to 0.1 -> considered weak 
- 0.1 to 0.3 -> medium 
- 0.3 + -> considered strong / need suspicion. 

Result : 

![IV](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/information_value_logistic_regression.PNG)

Weak Predictor : 
- NumberofDependents
- MonthlyIncome 
- NumberOfOpenCreditLinesAndLoans
- NumberOfRealEstateLoansOrLines
- 
Medium Predictor :
- DebtRatio
- age

Strong Predictor : 
- RevolvingUtilizationOfUnsecuredLines
- NumberOfTimes90DaysLate
- NumberOfTime30-59DaysPasrDueNotWorse
-  NumberOfTime60-89DaysPasrDueNotWorse

The feature that was fitted to the model was woe transformed data not the binning. 
---
## Model Evaluation and Calibration
Calibration Plot
![Calibration Plot](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/calibration_plot_logistic_regression.PNG)
The base model and the class_weight=balanced model has different proba calibration result. 
The  class_weight=balanced model has tendency to have higher mean probability compared to its fraction. 



ROC Curve : 

![ROC Curve](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/roc_auc_logistic_regression.PNG)

Confusion Matrix : 
![Confusion ](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/confusion_matrix_logistic_regression.PNG)


In this task we use AUC metrics.Given Confusion Matrix Figure above false negative is quite small.


--- 
## Feature Contribution  
We aim to answer which variable contribute more to this model. There are two options to find model feature importance in this case : 
1. Permutaion Importance 
2. Tree based model usually there is Feature Importance 

Permutaian Importance :
defined as a decrease in model performance if a single feature value is randomly shuffled. Permutation Importance offer model agnoticness or it doesnot affected by model itself. 

![Permutation Importance](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/permutation_importance_logistic_reg.png)

![Permutation Importance DataFrame](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/permutation_importance_logistic_regression_df.PNG)




Feature Importance : 

In Tree Models there is a term about impurity ,which described as probability of misclassification (classification task.). The Decision to split to the next node is based on impurity which can be calculated with gini/entropy / etc. 
To calculate feature importance we need to calculate Mean Decrease Impurity (MDI). In simple Terms the Feature Importance rank features based on its feature split gain the least impurity. 

However in Feature Importance method if the features contain high cardinality values it will biased the feature importance and feature importance are measured on training statistics and not unseen data. 


In Permutation importance LogRevolvingUtilizationofUnsecuredLines will affect the model auc the most if shuffled.The Feature Importance gives make sense result since according to [Investopedia Article](https://www.investopedia.com/ask/answers/110614/what-are-differences-between-revolving-credit-and-line-credit.asp) utilization rate impact 30% of credit score.


## Scorecard

$$ Score = Offset + Factor * \ln(odds) $$ 

$$ Score+pdo = Offset + Factor * \ln(odds) $$ 

$$ pdo = Factor * \ln(2) $$ 
$$ factor = pdo /  \ln(2) $$ 


Explanation  : 
pdo stands for point to double odds 
Offset -> Base Score 

We need to define the Offset and Factor. 

In score 650 i want odds to be 50: 1 and double every additional 20 points 

$$ factor = 20 / \ln(2) $$
$$ factor = 28.8539 $$

Now we calculate the offset 

$$Offset = Score  - Factor * \ln(odds) $$
$$Offset = 650  - 28.8539 * \ln(50/1) $$
$$Offset = 537.1228794036768 $$


Our Final Score Scaling 

$$ Score = 537.1228794036768 + 28.8539 * \ln(odds) $$ 


to calculate odds we need to sum the calculation woe of each feature multiply by its feature coefficient + intercept / feature then add with factor and offset 

![Model Coefficient](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/logistic_regression_model_coef.PNG)

Implementation in Python 
```
def create_score_card(model_coef_dict,factor=28.8539) :
    temp = []
    for key in woe_reference : 
        data = pd.DataFrame()

        ref = woe_reference.get(key)
        data['bin'] = list(ref.keys())
        data['features'] = key
        data['woe'] = list(ref.values())
        data['woe']= data['woe'].apply(lambda x : x['woe'])
        data['woe'] = data['woe'].replace(np.inf,0)
        data['woe'] = data['woe'].replace(-np.inf,0)
        feature_coef = model_coef_dict.get(key)
        data['score'] = -feature_coef*data['woe']*factor
        data['score'] =data['score'].astype('int')
        temp.append(data)
    scorecard_df = pd.concat(temp)
    scoring_dict = {}
    for feature in scorecard_df.features.unique() : 
        scoring_dict[feature] = {}
        sliced_df = scorecard_df.loc[scorecard_df.features==feature]
        for binrange,score in zip(sliced_df['bin'],sliced_df['score']) : 
            scoring_dict[feature][binrange] =score
    return scorecard_df,scoring_dict

scorecard_df,scoring_dict =create_score_card(model_coef_dict=model_coef)

```
then to apply into training / upcoming dataset 
I created a function to access created scoring dictionary above 

```
def apply_scoring(data,scoring_dict,feature_names:list,offset=537.1228794036768) : 
    for feature in feature_names : 
        #create new feature scorer 
        feature_dict = scoring_dict.get(feature)
        data[f'score_{feature}'] =data[feature].apply(lambda x:feature_dict.get(x))
        score_col = [x for x in data.columns if x.startswith('score_')]
        data['score'] = data[score_col].sum(axis=1) + offset
        data.drop(score_col,axis=1,inplace=True)
    return data
```

## Cut off Decision
Cut off Decision can be concluded from approval rate and required bad rates. Unfortunately I only have bad rates information. I set 8% Bad Rates (Umm this quite bad actually) but for learning I considered this as okay. 

![bad](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/bad_rates.PNG)

With 8% Bad Rates we can accept credit with score 537 or above.




___
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

3. Prepare The Dataset 
   Execute python file in [src/score_card/make_dataset.py](https://github.com/fakhrirobi/give_credit/tree/main/src/score_card/prepare_data.py)
   This process only create single training or test data not both. 
   This python file needs CLI argument : 

   a. --experiment_name = the name of experiment (make sure its consistent)

   b. --raw_input_path = the source of data (.csv file). Default in  [give_credit/data/raw](https://github.com/fakhrirobi/give_credit/tree/main/data/raw) Folder

   c. --interim_output_path = the output file (.csv) to store wrangled dataset (optional)

   d. --processed_output_path = the processed output file (feature engineered) data and ready for training (modelling) / inference (test_data)

   e. --training_req = wheter or not the process is for creating training data or test data . True means preparing for Training Data. False mean for test data. The difference is that The Training Data Contains Labels.

    The Example can be seen in [Makefile](https://github.com/fakhrirobi/give_credit/blob/main/Makefile) in "prepare_score_card_data" command. 

    Notes : to change the data wrangling you need to modify in [src/score_card/cleaning_data.py](https://github.com/fakhrirobi/give_credit/blob/main/src/score_card/cleaning_data.py) the same apply to [feature engineering](https://github.com/fakhrirobi/give_credit/blob/main/src/score_card/make_feature.py) process.

   ---

4. Train the Model 
    After Preparing The Dataset you can move to the model 
    to train model you can run the [src/score_card/train_logistic_model.py](https://github.com/fakhrirobi/give_credit/blob/main/src/score_card/train_logistic_model.py)

    This python file needs CLI argument : 

    a. --experiment_name = the name of experiment (make sure its consistent)

    b. --training_path = the source of training datadata (.csv file). Default in  [give_credit/data/processed](https://github.com/fakhrirobi/give_credit/tree/main/data/processed) Folder
    c. --config_path = The config.yaml file contain information of the process. This file later will be stored and logged as artifact in mlflow. 

    The Trained model willbe stored as {Modelname}_{experiment_name}.joblib file in [give_credit/models](https://github.com/fakhrirobi/give_credit/tree/main/models) folder. 

    The Example can be seen in [Makefile](https://github.com/fakhrirobi/give_credit/blob/main/Makefile) in "train_scorecard" command. 

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
  "timestamp": "2023-01-29T01:18:56.963103",
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
            0.04816276206531589,
            0.9518372379346841
          ]
        ],
        "label": [
          1
        ],
        "credit_score": 544.1228794036768,
        "accept_credit": "Approved"
      }
    },,
    ```
    Tips : When Constructing JSON Response if you have inference result such as proba in numpy types convert it as list first cause it wont accept numpy. 



## Reference 
---

1. https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
2. https://scikit-learn.org/stable/modules/permutation_importance.html
3. https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3
4. https://medium.com/@Sanskriti.Singh/an-emphasis-on-the-minimization-of-false-negatives-false-positives-in-binary-classification-9c22f3f9f73
5. https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html#id-55f8f5

