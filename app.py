from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LogisticRegression
from transformers import WOETransformer, BinningTransformer
from bins import bins
import numpy as np

app = FastAPI()

# Cargar datos desde el archivo CSV al iniciar la aplicación
df = pd.read_csv("new_model.csv")

model = LogisticRegression()
model.intercept_ = df.iloc[0, 0]
model.coefs_ = df.iloc[0, 1:]

pandas_df = pd.read_csv(
    "/Users/reginaflores/Library/CloudStorage/OneDrive-ITESO/ITESO/4° Semestre/INGENIERÍA/credit_data.csv").dropna()

print(pandas_df.head())


def categorizar_loan_status(estado):
    if estado in ['Charged Off', 'Default', 'Late (31-120 days)',
                  'Does not meet the credit policy. Status:Charged Off']:
        return "0"
    else:
        return "1"


pandas_df["loan_status"] = pandas_df["loan_status"].map(categorizar_loan_status)

pandas_df["loan_amnt"] = pd.to_numeric(pandas_df["loan_amnt"])
pandas_df["installment"] = pd.to_numeric(pandas_df["installment"])
pandas_df["annual_inc"] = pd.to_numeric(pandas_df["annual_inc"])
pandas_df["loan_status"] = pd.to_numeric(pandas_df["loan_status"])

bin_trans = BinningTransformer(bins)

df = bin_trans.transform(pandas_df)

print(df.iloc[0, 1])

df = df[df['home_ownership'] != 'ANY']

X = df[["loan_amnt", "term", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "purpose"]]
Y = df["loan_status"]

woe_transf = WOETransformer(X).fit(X, Y)

X = woe_transf.transform(X)


@app.get("/get_data")
async def get_data(loan_amnt: float, term: str, installment: float, grade: str, emp_length: str,
                   home_ownership: str, annual_inc: float,
                   purpose: str):  # Poner los parametros del modelo, (income, purpose, etc...)

    df_ = pd.DataFrame({"loan_amnt": [loan_amnt],
                        "term": [term],
                        "installment": [installment],
                        "grade": [grade],
                        "emp_length": [emp_length],
                        "home_ownership": [home_ownership],
                        "annual_inc": [annual_inc],
                        "purpose": [purpose]})

    df_ = bin_trans.transform(df_)
    df_ = woe_transf.transform(df_)
    x = df_.values[0]
    b = np.array(
        [-0.9226845117445188, -0.014844772258375103, 1.2091211585390806, 0.9551104641241108, 0.8923720619909415,
         0.7044332449496643, 0.0, 0.4855591292387278])
    b0 = 2.108784358460791
    result = np.dot(x, b) + b0
    logit = 1 / (1 + np.exp(-result))
    return {"pd": (1 - logit)}
