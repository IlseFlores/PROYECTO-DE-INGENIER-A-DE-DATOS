from pyspark import RDD
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType, BooleanType, DateType, DoubleType, StringType
from pyspark.sql.functions import udf, struct, col
import pandas as pd
from bins import bins
from typing import Dict, List
import pandas as pd
import numpy as np
import time
import pickle

# Configuración de Spark
spark_conf = {
    "spark.driver.cores": "1",
    "spark.driver.memory": "1g",
    "spark.executor.memory": "1g",
}

# Crear una sesión de Spark
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Word Count") \
    .config(map=spark_conf) \
    .getOrCreate()

# Leer el archivo CSV
data = spark.read.csv(
    "/Users/reginaflores/Library/CloudStorage/OneDrive-ITESO/ITESO/4° Semestre/INGENIERÍA/credit_data.csv",
    header=True).dropna()


# Define una función para mapear las categorías
def categorizar_loan_status(estado):
    if estado in ['Charged Off', 'Default', 'Late (31-120 days)',
                  'Does not meet the credit policy. Status:Charged Off']:
        return "0"
    else:
        return "1"


# Aplica la función para crear la nueva variable Y
udf_cat = udf(categorizar_loan_status, StringType())

data_cat = data.withColumn("loan_status", udf_cat(col("loan_status"))) \
    .withColumn("loan_status", col("loan_status").cast(IntegerType()))

data_cat.show()

pandas_df = data_cat.select("*").toPandas()
print(pandas_df)


class BinningTransformer:
    def __init__(self, bins: Dict):
        self.bins = bins

    def __find_bin(self, value: float, mappings: List):
        for mapping in mappings:
            if value <= mapping["max"]:
                return mapping["label"]
        return "Error"

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        X = X.copy()
        for key in self.bins.keys():
            X.loc[:, key] = X.loc[:, key].transform(lambda x: self.__find_bin(x, self.bins[key]))
        return X


pandas_df["loan_amnt"] = pd.to_numeric(pandas_df["loan_amnt"])
pandas_df["installment"] = pd.to_numeric(pandas_df["installment"])
pandas_df["annual_inc"] = pd.to_numeric(pandas_df["annual_inc"])
pandas_df["loan_status"] = pd.to_numeric(pandas_df["loan_status"])

x = BinningTransformer(bins)

df = x.transform(pandas_df)
print(df)

df = df[df['home_ownership'] != 'ANY']
print(df)

X = df[["loan_amnt", "term", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "purpose"]]
Y = df["loan_status"]


class WOETransformer:
    def __init__(self, columns: List[str], target_mappings: Dict = {1: "good", 0: "bad"}):
        self.target_mappings = target_mappings
        self.columns = columns
        self.woe_mappings = None

    def __get_absolute_odds(self, df: pd.DataFrame, col: str):
        key_first, key_second = list(self.target_mappings.keys())
        return (
            df.query(f"status=={key_first}")
            .groupby(col).size().reset_index()
            .rename(columns={0: self.target_mappings[key_first]})
            .set_index(col)
        ).join(
            df.query(f"status=={key_second}")
            .groupby(col).size().reset_index()
            .rename(columns={0: self.target_mappings[key_second]})
            .set_index(col)
        ).reset_index()[[col, "good", "bad"]]

    @staticmethod
    def __calculate_relative_odds(row: pd.Series, total_good: int, total_bad: int) -> pd.Series:
        return pd.Series(
            {
                **row.to_dict(),
                "good": row["good"] / total_good,
                "bad": row["bad"] / total_bad
            }
        )

    def __get_odds(self, df: pd.DataFrame, col: str,
                   absolute_values: bool = False) -> pd.DataFrame:
        key_first, key_second = list(self.target_mappings.keys())
        odds_absolute = self.__get_absolute_odds(df, col)

        if absolute_values:
            return odds_absolute

        # Relative Odds
        total_good = odds_absolute["good"].sum()
        total_bad = odds_absolute["bad"].sum()
        return odds_absolute.apply(
            lambda row: WOETransformer.__calculate_relative_odds(row, total_good, total_bad),
            axis=1
        )

    @staticmethod
    def __calculate_woe(row: pd.Series) -> pd.Series:
        return pd.Series(
            {
                **row.to_dict(),
                "woe": np.log(row["good"] / row["bad"]),
                "info_val": (row["good"] - row["bad"]) * np.log(row["good"] / row["bad"])
            }
        )

    def __set_woe_mappings(self, X: pd.DataFrame, y: pd.Series,
                           absolute_values: bool = False) -> None:
        df = X.copy()
        df["status"] = y

        self.woe_mappings = {
            col: self.__get_odds(df, col, absolute_values) \
                .apply(lambda row: WOETransformer.__calculate_woe(row), axis=1) \
                .sort_values(by="woe", axis=0, ascending=True)
            for col in self.columns
        }

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        self.__set_woe_mappings(X, y, *args, **kwargs)
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        if self.woe_mappings is None:
            raise NotFittedError(
                f"This {self} instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer.")
        df = X.copy()
        out = pd.DataFrame([])
        for col in df.columns:
            mapping = self.woe_mappings[col].set_index(col)
            categories = list(mapping.index)
            out[col] = df.loc[:, col].apply(lambda cat: mapping.loc[cat, "woe"])
        return out


x = WOETransformer(X).fit(X, Y)

print(x)

X = x.transform(X)
print(X)

df_woes = pd.concat([X, Y], axis=1)
print(df_woes)

# TODO: Crear dataframe de Spark de df_woes
df_woes_spark = spark.createDataFrame(df_woes)
df_woes_spark.printSchema()
# TODO: Guardar dataframe como parquet
df_woes_spark.write.parquet(f"{time.strftime('%Y%m%d%H%M%S')}/woes.parquet")
