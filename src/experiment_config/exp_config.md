
# About Experiment Config : 



Concept : 

Each Model Training considered as experiment. 
```yaml
params : 
  lambda_l1: 2.8611070127189538e-05 
  lambda_l2: 0.6807987664418726 
  num_leaves: 9 
  max_depth: 19 
  feature_fraction: 0.5332191589953184 
  bagging_fraction: 0.9238892410770332 
  bagging_freq: 3 
  min_child_samples: 73

feature_engineering_steps : 
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


preprocessing_step : 
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

number_dtypes : ['int64','int32','int16','int8','float64','float32','float16','float8']




data_wrangling_output_dtypes : 
    age: *number_dtypes,
    NumberOfTime30-59DaysPastDueNotWorse : *number_dtypes ,
    MonthlyIncome : *number_dtypes,
    NumberOfOpenCreditLinesAndLoans : *number_dtypes,
    NumberOfTimes90DaysLate: *number_dtypes,
    NumberRealEstateLoansOrLines : *number_dtypes,
    NumberOfTime60-89DaysPastDueNotWorse : *number_dtypes,
    NumberOfDependents : *number_dtypes,
    DebtRatio :  *number_dtypes, 
    RevolvingUtilizationOfUnsecuredLines :  *number_dtypes



feature_engineering_output_dtypes : 
    age: *number_dtypes,
    NumberOfTime30-59DaysPastDueNotWorse : *number_dtypes ,
    NumberOfOpenCreditLinesAndLoans : *number_dtypes,
    NumberOfTimes90DaysLate: *number_dtypes,
    NumberRealEstateLoansOrLines : *number_dtypes,
    NumberOfTime60-89DaysPastDueNotWorse : *number_dtypes,
    NumberOfDependents : *number_dtypes,
    LogDebtRatio :  *number_dtypes, 
    LogRevolvingUtilizationOfUnsecuredLines :  *number_dtypes,
    LogIncome  : *number_dtypes, 
```

each 'experiment_name'.yaml file should contain minimal : 

1. feature_engineering_steps (step conducted in feature_engineering process)
   
   example : 
   ```yaml
        feature_engineering_steps : 
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

    ```
2. preprocessing_steps (step conducted in data wrangling phase )

    example : 
   ```yaml
        preprocessing_step : 
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

    ```
3. data_wrangling_output_dtypes (expected column and its dtypes) -> to ensure data integrity
        example : 
   ```yaml
            data_wrangling_output_dtypes : 
                age: *number_dtypes,
                NumberOfTime30-59DaysPastDueNotWorse : *number_dtypes ,
                MonthlyIncome : *number_dtypes,
                NumberOfOpenCreditLinesAndLoans : *number_dtypes,
                NumberOfTimes90DaysLate: *number_dtypes,
                NumberRealEstateLoansOrLines : *number_dtypes,
                NumberOfTime60-89DaysPastDueNotWorse : *number_dtypes,
                NumberOfDependents : *number_dtypes,
                DebtRatio :  *number_dtypes, 
                RevolvingUtilizationOfUnsecuredLines :  *number_dtypes



    ```
4. feature_engineering_output_dtypes (expected column and its dtypes) -> to ensure data integrity

    example : 
    ```yaml
        feature_engineering_output_dtypes : 
            age: *number_dtypes,
            NumberOfTime30-59DaysPastDueNotWorse : *number_dtypes ,
            NumberOfOpenCreditLinesAndLoans : *number_dtypes,
            NumberOfTimes90DaysLate: *number_dtypes,
            NumberRealEstateLoansOrLines : *number_dtypes,
            NumberOfTime60-89DaysPastDueNotWorse : *number_dtypes,
            NumberOfDependents : *number_dtypes,
            LogDebtRatio :  *number_dtypes, 
            LogRevolvingUtilizationOfUnsecuredLines :  *number_dtypes,
            LogIncome  : *number_dtypes, 


        ```