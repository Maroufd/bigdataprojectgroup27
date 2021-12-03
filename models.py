from pyspark import *
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


## Multiple Linear Regression (DEFAULT) ##
## Multiple Linear Regression with INTERACTIONS ##
## SIMPLE Linear Regression ##

def select_LinearRegressionModel(train_set, test_set):
  lr = LinearRegression(featuresCol = "features", labelCol="ArrDelay")
  lr_model = lr.fit(train_set)

  train_predictions = lr_model.transform(train_set)
  test_predictions = lr_model.transform(train_set)

  return lr_model, train_predictions, test_predictions


## REGULARIZED Multiple Linear Regression ##

def select_RegularizedLinearRegressionModel(train_set, test_set):
  # Trains a Linear Regression model
  lr_elasticnet = LinearRegression(featuresCol = "features", labelCol="ArrDelay")

  # Hyper-parameter alpha tuned with grid
  paramGrid = ParamGridBuilder().addGrid(lr_elasticnet.elasticNetParam, [0, 0.05, 0.1, 0.15, 0.2]).build()
  lr_elasticnet_evaluator = RegressionEvaluator().setLabelCol("ArrDelay").setPredictionCol("prediction")

  crossval = CrossValidator(estimator=lr_elasticnet,
                            estimatorParamMaps=paramGrid,
                            evaluator=lr_elasticnet_evaluator,
                            numFolds=3)

  lr_elasticnet_cv = crossval.fit(train_set)
  lr_cv_bestmodel = lr_elasticnet_cv.bestModel

  train_predictions = lr_cv_bestmodel.transform(train_set)
  test_predictions = lr_cv_bestmodel.transform(train_set)
  return lr_cv_bestmodel, train_predictions, test_predictions


## DECISION TREE Regression ##

def select_DecisionTreeRegressionModel(train_set, test_set):
  # Train a DecisionTree model
  dtr = DecisionTreeRegressor(featuresCol = "features", labelCol="ArrDelay")
  dtr_model = dtr.fit(train_set)

  train_predictions = dtr_model.transform(train_set)
  test_predictions = dtr_model.transform(train_set)

  return dtr_model, train_predictions, test_predictions
