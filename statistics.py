from pyspark import *
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

def print_linear_regression_summary(lr_model):
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    return str(lr_model.coefficients),str(lr_model.intercept)

def print_training_summary(trainingSummary):
    print("*** LINEAR MODEL ***")
    print("RMSE (train): ", trainingSummary.rootMeanSquaredError)
    print("R2 (train): ", trainingSummary.r2)
    print("Adjusted R2 (train): ", trainingSummary.r2adj)
    return trainingSummary.rootMeanSquaredError,trainingSummary.r2,trainingSummary.r2adj

def print_test_summary(test_result):
    print("RMSE (test): ", test_result.rootMeanSquaredError)
    print("R2 (test): ", test_result.r2)
    print("Adjusted R2 (test): ", test_result.r2adj)
    return test_result.rootMeanSquaredError,test_result.r2,test_result.r2adj
