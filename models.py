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

def LinearRegressionmodel(train_set,test_set):
    lr = LinearRegression(featuresCol = "features", labelCol="ArrDelay")
    lr_model = lr.fit(train_set)
    trainingSummary = lr_model.summary
    test_result = lr_model.evaluate(test_set)
    return lr_model,trainingSummary, test_result
