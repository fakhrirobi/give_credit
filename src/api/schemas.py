from typing import List, Union

from fastapi import Query
from pydantic import BaseModel, validator

# input_dictionary = {
#     "RevolvingUtilizationOfUnsecuredLines": utilization_rate,
#     "age": age,
#     "NumberOfTime30-59DaysPastDueNotWorse": number30_59daysdue,
#     "DebtRatio": debtratio,
#     "MonthlyIncome": monthlyincome,
#     "NumberOfOpenCreditLinesAndLoans": numopencredit_loans,
#     "NumberOfTimes90DaysLate": number90dayslate,
#     "NumberRealEstateLoansOrLines": numberrealestate_loans,
#     "NumberOfTime60-89DaysPastDueNotWorse": number60_89daysdue,
#     "NumberOfDependents": numof_dependents,
# }


class UserCreditData(BaseModel):
    utilization_rate: float
    age: int
    number30_59daysdue: int
    debtratio: float
    monthlyincome: float
    numopencredit_loans: int
    number90dayslate: int
    numberrealestate_loans: int
    number60_89daysdue: int
    numof_dependents: int
    customer_id: str

    # @validator("utilization_rate")
    # def utilization_rate_validation(cls, value) :
    #     pass

    # @validator("utilization_rate")
    # def utilization_rate_validation(cls, value) :
    #     pass

    # @validator("age")
    # def age_validation(cls, value) :
    #     pass

    # @validator("number30_59daysdue")
    # def age_validation(cls, value) :
    #     pass
    # @validator("number30_59daysdue")
    # def age_validation(cls, value) :
    #     pass

    # @validator("debtratio")
    # def age_validation(cls, value) :
    #     pass

    # @validator("monthlyincome")
    # def age_validation(cls, value) :
    #     pass

    # @validator("numopencredit_loans")
    # def age_validation(cls, value) :
    #     pass
    # @validator("number90dayslate")
    # def age_validation(cls, value) :
    #     pass

    # @validator("numberrealestate_loans")
    # def age_validation(cls, value) :
    #     pass

    # @validator("number60_89daysdue")
    # def age_validation(cls, value) :
    #     pass

    # @validator("numof_dependents")
    # def age_validation(cls, value) :
    #     pass


class PredictionInput(BaseModel):
    model_input: Union[UserCreditData, List[UserCreditData]]

    class Config:
        schema_extra = {
            "example": {
                "model_input": [
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
                        "numof_dependents": 3,
                    },
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
                        "numof_dependents": 3,
                    },
                ]
            },
        }
