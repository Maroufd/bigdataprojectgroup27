#################
# STATISTICS.PY #
#################

from pyspark import *
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

def print_linear_regression_summary(linear_model):
    coefficients = str(linear_model.coefficients)
    intercept = str(linear_model.intercept)
    print("** LIST OF REGRESSION COEFFICIENTS **")
    print("- Coefficients: " + str(coefficients))
    print("- Intercept: " + str(intercept))
    return coefficients, intercept

def print_summary(df_results):
    evaluator = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction")
    rmse = evaluator.evaluate(df_results, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(df_results, {evaluator.metricName: "r2"})

    print("- RMSE: ", rmse)
    print("- R2: ", r2)

    return rmse, r2
