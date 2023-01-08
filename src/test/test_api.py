from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


def test_prediction():
    response = client.post(
        "/predict_single/",
        json={
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
    )
    assert response.status_code == 200
