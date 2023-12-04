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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

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

data = spark.read.parquet("./20231129214521/woes.parquet")
data.show()
df_woes = data.toPandas()
print(df_woes)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# TODO: Cargar datos de archivo parquet

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_woes.drop('loan_status', axis=1),
                                                    df_woes['loan_status'], test_size=0.2, random_state=42)

# Inicializar el modelo de regresión logística
model = LogisticRegression(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Calcular el informe de clasificación, que incluye F1-score
class_report = classification_report(y_test, y_pred)
print("\nInforme de Clasificación:")
print(class_report)

# Calcular la probabilidad de predicción para las clases positivas
y_prob = model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC y el área bajo la curva (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Imprimir el AUC
print("\nÁrea bajo la curva (AUC): {:.2f}".format(roc_auc))

# TODO: Exportar modelo, puede ser por parametros y guardarlos en un txt, o serializar con pickle.
# Supongamos que tienes un modelo llamado mi_modelo
# y quieres obtener los parámetros del modelo
coef = model.coef_
intercept = model.intercept_

df_data = np.c_[intercept, coef]
dataframe_1 = pd.DataFrame(df_data, columns=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"])

# Guarda los parámetros en un archivo de texto
dataframe_1.to_csv("new_model.csv", index=False)
