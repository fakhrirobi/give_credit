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
    <li><a href="#model-comparison">Model Comparison / Decision</a></li>
    <li><a href="#hyperparameter-tuning">Hyperparameter Tuning</a></li>
    <li><a href="#feature-contribution">Feature Contribution</a></li>
    <li><a href="#result">Kaggle Submission Result</a></li>
  </ol>
</details>

### Data Preprocessing 
    1. Dropping Duplicates 
    2. Filling Missing Values : 
        a. MonthlyIncome -> Imputed with Median 
        b. NumberOfDependents -> Imputed with Mode 
    

### Exploratory Data Analysis
<details>
    <summary></summary>
    <ol>
        <li>
        <a>Target Class</a>
        <p>Mostly Label 0 with percentage of roughly 93.3% and the rest is label 1 </p>
        <image>      </image>
        </li>
        <li>
        <a>RevolvingUtilizationofUnsecuredLines</a>
        <p>Mostly Label 0 with percentage of roughly 93.3% and the rest is label 1 </p>

        </li>
    </ol>
</details>

### Feature Engineering
    1. Binning Feature 
        Since this goal is to create credit scoring we need to use binning in order to calculate weight of evidence . Here is my binning 
        ```
        age_binning = [-math.inf, 20,25,30,35, 40,45, 50,55,60,65, 70, math.inf]
        dependent_bin = [-math.inf,0,1,2,3,4,5,6,7,8,math.inf]
        dependent_binning = [0,2,4,5,6,7,8,9,math.inf]
        binning_late_90days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
        binning_late_3059days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
        binning_late_6089days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
        interval_revolving_rate = [-math.inf,0.000, 0.00342,0.0215,0.0489,0.0954,0.174,0.297,0.468,0.708,0.973,math.inf]
        debt_ratio_interval = [-math.inf,0.000,0.0144,0.0954,0.165,0.226,0.284,0.346,0.417,0.512,0.684,1.903,math.inf]
        monthlyincome_interval = [-math.inf,3198.0, 4755.0, 6400.0, 9150.0,math.inf]
        logmonthlyincome_interval = [-math.inf,0.001, 7.748,8.071,8.294,8.467,8.613,8.764,8.927,9.122,9.365,14.917,math.inf]
        creditlines_interval  = [-math.inf,1,2,3,4.0,5,6,7,8, 9.0, 12.0,math.inf]
        realestatelines_interval = [-math.inf,1,2,3,4.0,5,6,7,8, 9.0, 12.0,math.inf]

        ```
        From the Credit Risk Scorecard Book by Naeem Siddiqi its better to create many bins in order to create diverse credit score. 

    2. After the binning process is done now we are going to calculate Weight of Evidence 
   
   $$ WOE = ln\left(\frac{P(\text{class 0 (non events) outcome})}{P(\text{class 1 (events)outcome})}\right) $$
   

    Interpretation : 
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

    WoE of each features 

    {'bin_age': {'(20.0, 25.0]': {'woe': -0.5029778008571553,
   'iv': 0.006114302046664361},
  '(25.0, 30.0]': {'woe': -0.5744835047352527, 'iv': 0.02306912293486296},
  '(30.0, 35.0]': {'woe': -0.44273113180112517, 'iv': 0.01834339827340103},
  '(35.0, 40.0]': {'woe': -0.29524942339237314, 'iv': 0.009538179102773575},
  '(40.0, 45.0]': {'woe': -0.21493823443427776, 'iv': 0.005784051013270645},
  '(45.0, 50.0]': {'woe': -0.1728852313034842, 'iv': 0.00420029323227334},
  '(50.0, 55.0]': {'woe': -0.035912776219466286, 'iv': 0.00015882672214590274},
  '(55.0, 60.0]': {'woe': 0.2342845364331494, 'iv': 0.0055225069026100795},
  '(60.0, 65.0]': {'woe': 0.532489628459634, 'iv': 0.023401830509758165},
  '(65.0, 70.0]': {'woe': 0.926085919972863, 'iv': 0.03939504116655464},
  '(70.0, inf]': {'woe': 1.0603505244294575, 'iv': 0.07594171159791153}},
 'bin_NumberOfDependents': {'(-inf, 0.0]': {'woe': 0.16192507634701706,
   'iv': 0.013369760082315225},
  '(0.0, 1.0]': {'woe': -0.06715983128081378, 'iv': 0.0009445195622924607},
  '(1.0, 2.0]': {'woe': -0.17974735201038614, 'iv': 0.005192668410109127},
  '(2.0, 3.0]': {'woe': -0.2955551702197531, 'iv': 0.007127557982724367},
  '(3.0, 4.0]': {'woe': -0.4437687564578963, 'iv': 0.005226408948367431},
  '(4.0, 5.0]': {'woe': -0.35018591238339003, 'iv': 0.0008082808157530725},
  '(5.0, 6.0]': {'woe': -0.7075605240085275, 'iv': 0.0008454878573382016},
  '(6.0, 7.0]': {'woe': -0.5031744770304708, 'iv': 0.00012335752382213716},
  '(7.0, 8.0]': {'woe': -0.30472353830663257, 'iv': 1.985183244125116e-05},
  '(8.0, inf]': {'woe': inf, 'iv': inf}},
 'bin_NumberOfTimes90DaysLate': {'(-inf, 0.0]': {'woe': 0.36826452497339185,
   'iv': 0.10946953257943327},
  '(0.0, 1.0]': {'woe': -1.9163035436697873, 'iv': 0.28414139377024417},
  '(1.0, 2.0]': {'woe': -2.602176028074158, 'iv': 0.17418594164224596},
  '(10.0, inf]': {'woe': -2.9061640043505843, 'iv': 0.04088154963694475},
  '(2.0, 3.0]': {'woe': -2.8871122161803795, 'iv': 0.09973053825813732},
  '(3.0, 4.0]': {'woe': -3.369448683347575, 'iv': 0.06051917157990356},
  '(4.0, 5.0]': {'woe': -3.141391117230936, 'iv': 0.021617744583479016},
  '(5.0, 6.0]': {'woe': -2.9174635596045175, 'iv': 0.010266794560591659},
  '(6.0, 7.0]': {'woe': -3.9423096980330183, 'iv': 0.009108444717117605},
  '(7.0, 8.0]': {'woe': -2.943780867921891, 'iv': 0.0024207838838483384},
  '(8.0, 9.0]': {'woe': -3.906591615430939, 'iv': 0.005221772929749199},
  '(9.0, 10.0]': {'woe': -3.5235993631748332, 'iv': 0.002120335775445685}},
 'bin_NumberOfTime30-59DaysPastDueNotWorse': {'(-inf, 0.0]': {'woe': 0.5262776651202775,
   'iv': 0.18456054814747894},
  '(0.0, 1.0]': {'woe': -0.8277866119422927, 'iv': 0.10928381433800137},
  '(1.0, 2.0]': {'woe': -1.5608458014559794, 'iv': 0.1504715977950691},
  '(10.0, inf]': {'woe': -2.9268313096081178, 'iv': 0.04019047478385672},
  '(2.0, 3.0]': {'woe': -1.9734229147774605, 'iv': 0.10619724750313386},
  '(3.0, 4.0]': {'woe': -2.3760199035220673, 'iv': 0.0718948171949233},
  '(4.0, 5.0]': {'woe': -2.379218656051143, 'iv': 0.03239343574012651},
  '(5.0, 6.0]': {'woe': -2.6244030646599783, 'iv': 0.017809589181043153},
  '(6.0, 7.0]': {'woe': -2.651760393871512, 'iv': 0.0070298878940651885},
  '(7.0, 8.0]': {'woe': -1.8451685792537815, 'iv': 0.0013486409489439246},
  '(8.0, 9.0]': {'woe': -2.201843523192514, 'iv': 0.000971389711744192},
  '(9.0, 10.0]': {'woe': -3.3004558118606235, 'iv': 0.0007883879021730193}},
 'bin_NumberOfTime60-89DaysPastDueNotWorse': {'(-inf, 0.0]': {'woe': 0.2740025289064378,
   'iv': 0.06333198744667347},
  '(0.0, 1.0]': {'woe': -1.7688631189946284, 'iv': 0.2536348158507391},
  '(1.0, 2.0]': {'woe': -2.602602740263266, 'iv': 0.12670578496938556},
  '(10.0, inf]': {'woe': -2.9214239608206145, 'iv': 0.03941837485538247},
  '(2.0, 3.0]': {'woe': -2.8949907037524594, 'iv': 0.04883748307021512},
  '(3.0, 4.0]': {'woe': -2.939442466323293, 'iv': 0.01588072441014534},
  '(4.0, 5.0]': {'woe': -2.7614593111279366, 'iv': 0.0044912092656616345},
  '(5.0, 6.0]': {'woe': -3.6189095429791585, 'iv': 0.004804151390455038},
  '(6.0, 7.0]': {'woe': -2.607308631300678, 'iv': 0.0005989743742358782},
  '(7.0, 8.0]': {'woe': -2.607308631300678, 'iv': 0.0002994871871179391}},
 'bin_RevolvingUtilizationOfUnsecuredLines': {'(-inf, 0.0]': {'woe': 0.7679625432486045,
   'iv': 0.029377983395282914},
  '(0.0, 0.00342]': {'woe': 1.6060034499854656, 'iv': 0.03966072862244436},
  '(0.00342, 0.0215]': {'woe': 1.6494639471958061, 'iv': 0.14189219963525201},
  '(0.0215, 0.0489]': {'woe': 1.5044276736332916, 'iv': 0.1241189283213773},
  '(0.0489, 0.0954]': {'woe': 1.2023009005609702, 'iv': 0.08884679136798479},
  '(0.0954, 0.174]': {'woe': 0.8942804199856766, 'iv': 0.05530737389599213},
  '(0.174, 0.297]': {'woe': 0.6727279233623773, 'iv': 0.034422038897358315},
  '(0.297, 0.468]': {'woe': 0.20155444717670443, 'iv': 0.0037394774305777793},
  '(0.468, 0.708]': {'woe': -0.32057996028328106, 'iv': 0.011852141697369695},
  '(0.708, 0.973]': {'woe': -0.9679317511217411, 'iv': 0.1425434338268058},
  '(0.973, inf]': {'woe': -1.4180965190526664, 'iv': 0.358356476998304}},
 'bin_DebtRatio': {'(-inf, 0.0]': {'woe': -0.33225969611523803,
   'iv': 0.0037739522300778523},
  '(0.0, 0.0144]': {'woe': 0.7583820619045673, 'iv': 0.027534931333345204},
  '(0.0144, 0.0954]': {'woe': 0.04922158486827305,
   'iv': 0.00023836023229700703},
  '(0.0954, 0.165]': {'woe': 0.10557668390582804, 'iv': 0.001062772752885419},
  '(0.165, 0.226]': {'woe': 0.1597210342130207, 'iv': 0.00239770308027819},
  '(0.226, 0.284]': {'woe': 0.3067087451708338, 'iv': 0.008273253025081433},
  '(0.284, 0.346]': {'woe': 0.2851806498853937, 'iv': 0.007340693811711259},
  '(0.346, 0.417]': {'woe': 0.1098449553815002, 'iv': 0.0011529436715332715},
  '(0.417, 0.512]': {'woe': -0.05093283967246327, 'iv': 0.000265653589081457},
  '(0.512, 0.684]': {'woe': -0.3107771074332837, 'iv': 0.011087283323248383},
  '(0.684, 1.903]': {'woe': -0.5927893774329883, 'iv': 0.04569648473470617},
  '(1.903, inf]': {'woe': -0.5278670896208424, 'iv': 5.369506045673205e-05}},
 'bin_MonthlyIncome': {'(-inf, 3198.0]': {'woe': -0.34335544561697623,
   'iv': 0.027396024218654832},
  '(3198.0, 4755.0]': {'woe': -0.19915781025325746,
   'iv': 0.008692144550092365},
  '(4755.0, 6400.0]': {'woe': 0.012661454453368371,
   'iv': 3.16701332556103e-05},
  '(6400.0, 9150.0]': {'woe': 0.25498266145471854, 'iv': 0.011608136487665899},
  '(9150.0, inf]': {'woe': 0.4554791544000379, 'iv': 0.0343463675044932}},
 'bin_NumberOfOpenCreditLinesAndLoans': {'(-inf, 1.0]': {'woe': -1.074965273408164,
   'iv': 0.06840103955120491},
  '(1.0, 2.0]': {'woe': -0.3434835281161273, 'iv': 0.005373996525928137},
  '(12.0, inf]': {'woe': 0.017137113428482297, 'iv': 5.733955494509562e-05},
  '(2.0, 3.0]': {'woe': -0.16057266248516788, 'iv': 0.0015182887553258184},
  '(3.0, 4.0]': {'woe': 0.028852153250569335, 'iv': 5.94887038073605e-05},
  '(4.0, 5.0]': {'woe': 0.024558341592005012, 'iv': 4.982153866511331e-05},
  '(5.0, 6.0]': {'woe': 0.16770433455060918, 'iv': 0.0023466692358632183},
  '(6.0, 7.0]': {'woe': 0.19072570795079663, 'iv': 0.002976928386993973},
  '(7.0, 8.0]': {'woe': 0.33000392910803766, 'iv': 0.008063926381951126},
  '(8.0, 9.0]': {'woe': 0.11231455535919027, 'iv': 0.000938688860160284},
  '(9.0, 12.0]': {'woe': 0.11401807473450114, 'iv': 0.002155014380216844}},
 'bin_NumberRealEstateLoansOrLines': {'(-inf, 1.0]': {'woe': -0.030651873585237664,
   'iv': 0.0006775548618829203},
  '(1.0, 2.0]': {'woe': 0.1936641311490197, 'iv': 0.007564451761504452},
  '(12.0, inf]': {'woe': -1.390913306976185, 'iv': 0.001036495451307138},
  '(2.0, 3.0]': {'woe': 0.011661939747842957, 'iv': 5.971295313795687e-06},
  '(3.0, 4.0]': {'woe': -0.2565519946332669, 'iv': 0.0011086687738968237},
  '(4.0, 5.0]': {'woe': -0.6879036016253619, 'iv': 0.003012424394735633},
  '(5.0, 6.0]': {'woe': -0.8756530861423287, 'iv': 0.0025340520240293963},
  '(6.0, 7.0]': {'woe': -1.0162198575347745, 'iv': 0.0017689164598253847},
  '(7.0, 8.0]': {'woe': -1.489278256775467, 'iv': 0.0024315087477641477},
  '(8.0, 9.0]': {'woe': -1.0597461225846654, 'iv': 0.00085875210403469},
  '(9.0, 12.0]': {'woe': -1.1765625076099537, 'iv': 0.0011100737304696265}}}
    ```
## Model Evaluation and Calibration

![Calibration Plot](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/calibration_plot.PNG)

ROC Curve : 
![Tuned Model ROC Curve](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/roc%20curve.PNG)

Confusion Matrix : 
![Confusion ](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/confusion%20matrix.png)


In this task we use AUC metrics. However Given Confusion Matrix Chart The False Negative still plenty compared to its True Positive.There are several ways i planned to reduce the False Negative since in credit scoring we rather predict false positive than predicting false negative (hidden risk)


--- 
## Feature Contribution  
We aim to answer which variable contribute more to this model. There are two options to find model feature importance in this case : 
1. Permutaion Importance 
2. and for Tree based model usually there is Feature Importance 

Permutaian Importance :
![Permutation Importance](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/permutation_importance.png)

![Permutation Importance DataFrame](https://raw.githubusercontent.com/fakhrirobi/give_credit/main/assets/mean_importance.PNG)

defined as a decrease in model performance if a single feature value is randomly shuffled. Permutation Importance offer model agnoticness or it doesnot affected by model itself. 


Feature Importance : 

In Tree Models there is a term about impurity ,which described as probability of misclassification (classification task.). The Decision to split to the next node is based on impurity which can be calculated with gini/entropy / etc. 
To calculate feature importance we need to calculate Mean Decrease Impurity (MDI). In simple Terms the Feature Importance rank features based on its feature split gain the least impurity. 

However in Feature Importance method if the features contain high cardinality values it will biased the feature importance and feature importance are measured on training statistics and not unseen data. 


In Permutation importance LogRevolvingUtilizationofUnsecuredLines will affect the model auc the most if shuffled.The Feature Importance gives make sense result since according to [Investopedia Article](https://www.investopedia.com/ask/answers/110614/what-are-differences-between-revolving-credit-and-line-credit.asp) utilization rate impact 30% of credit score.







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
    │   │   └── wrangling.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── feature_eng.py
    │   │   └── validation_features.py <-data validation 
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model_batch.py <- to generate prediction batches (.csv) file 
    │   │   ├── predict_model_single.py <- to generate prediction on single inference (CLI)
    │   │   └── train_model.py
    │   │   └── param_tuning.py -> hyperparameter tuning 
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
5. https://medium.com/@Sanskriti.Singh/an-emphasis-on-the-minimization-of-false-negatives-false-positives-in-binary-classification-9c22f3f9f73


