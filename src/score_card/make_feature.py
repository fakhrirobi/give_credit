import pandas as pd 
import math 
import os 
import numpy as np
import joblib
from config.path_config import ROOT_DIR
TARGET = "SeriousDlqin2yrs"

WOE_REFERENCE_PATH = os.path.join(ROOT_DIR,"models","woe_reference.joblib")

age_binning = [-math.inf, 20,25,30,35, 40,45, 50,55,60,65, 70, math.inf]
dependent_bin = [-math.inf,0,1,2,3,4,5,6,7,8,math.inf]
dependent_binning = [0,2,4,5,6,7,8,9,math.inf]
binning_late_90days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
binning_late_3059days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
binning_late_6089days = [-math.inf,0,1,2,3,4,5,6,7,8,9,10,math.inf]
interval_revolving_rate = [-math.inf,0.000, 0.00342,0.0215,0.0489,0.0954,0.174,0.297,0.468,0.708,0.973,math.inf]
debt_ratio_interval = [-math.inf,0.000,0.0144,0.0954,0.165,0.226,0.284,0.346,0.417,0.512,0.684,1.903,math.inf]
monthlyincome_interval = [-math.inf,3198.0, 4755.0, 6400.0, 9150.0,math.inf]
creditlines_interval  = [-math.inf,1,2,3,4.0,5,6,7,8, 9.0, 12.0,math.inf]
realestatelines_interval = [-math.inf,1,2,3,4.0,5,6,7,8, 9.0, 12.0,math.inf]

def create_woe_reference(feature,binned_data:pd.DataFrame) :
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
    for x,y,z in zip (crosstab_data[feature],crosstab_data[f'{feature}_woe'],crosstab_data[f'{feature}_iv']) : 
        
        temp[x] = {'woe':y,'iv':z}

    return temp

def get_woe_value(x,col,woe_reference) :
    try : 
        
        ref = woe_reference.get(col) 
        return ref.get(x)['woe'] 
    except : 
        print(f'value of x : {x}, col = {col}')
        return 0


def create_binning(data:pd.DataFrame) : 
    data['bin_age'] = pd.cut(data['age'],bins=age_binning).astype('str')
    data['bin_NumberOfDependents'] = pd.cut(data['NumberOfDependents'],bins=dependent_bin).astype('str')
    data['bin_NumberOfTimes90DaysLate'] = pd.cut(data['NumberOfTimes90DaysLate'],bins=binning_late_90days).astype('str')
    data['bin_NumberOfTime30-59DaysPastDueNotWorse'] = pd.cut(data['NumberOfTime30-59DaysPastDueNotWorse'], bins=binning_late_3059days).astype('str')
    data['bin_NumberOfTime60-89DaysPastDueNotWorse'] = pd.cut(data['NumberOfTime60-89DaysPastDueNotWorse'], bins=binning_late_6089days).astype('str')

    data['bin_RevolvingUtilizationOfUnsecuredLines'] = pd.cut(data['RevolvingUtilizationOfUnsecuredLines'],bins=interval_revolving_rate).astype('str')
    data['bin_DebtRatio'] = pd.cut(data['DebtRatio'],bins=debt_ratio_interval).astype('str')
    data['bin_MonthlyIncome'] = pd.cut(data['MonthlyIncome'],bins=monthlyincome_interval).astype('str')
    data['bin_NumberOfOpenCreditLinesAndLoans'] = pd.cut(data['NumberOfOpenCreditLinesAndLoans'],bins=creditlines_interval).astype('str')
    data['bin_NumberRealEstateLoansOrLines'] = pd.cut(data['NumberRealEstateLoansOrLines'],bins=realestatelines_interval).astype('str')
    return data 





def feature_engineering(data : pd.DataFrame,training_req) : 
    
    binned_data = create_binning(data)
    ## Creating Feature from calculated woe
    binned_col = [x for x in binned_data.columns if x.startswith('bin')]
    woe_reference = {}
    if (training_req=='True') or (training_req == True) : 
        for feature in binned_col : 
            woe_reference[feature] = create_woe_reference(feature=feature,binned_data=binned_data)

        for feature in binned_col : 
            binned_data[f'woe_{feature}'] = binned_data[feature].apply(get_woe_value,col=feature,woe_reference=woe_reference)
            
        binned_data= binned_data.replace([np.inf, -np.inf], 0)
        
        joblib.dump(woe_reference,WOE_REFERENCE_PATH)
        return binned_data
    if (training_req == "False") or (training_req == False): 
        
        #try to load joblib file 
        woe_reference_loaded = joblib.load(WOE_REFERENCE_PATH)
        for feature in binned_col : 
            binned_data[f'woe_{feature}'] = binned_data[feature].apply(get_woe_value,col=feature,woe_reference=woe_reference_loaded)
            
        binned_data= binned_data.replace([np.inf, -np.inf], 0) 
        return binned_data
    return data



