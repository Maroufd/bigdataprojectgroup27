import argparse
import preprocessing
import models
import statistics
import ExploratoryDataAnalysis as eda
from pyspark import *
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

def run_application(config, *, seed=69, verbose=False):
    """Runs a simulation using dynamic routing.
    :param config: namespace with the configuration for the run
    :param seed: seed to use for the random cars
    :param verbose: whether or not to show user output
    """

    file=config.file
    trainsize=config.trainsize
    testsize=config.testsize
    model=config.model

    #Load data and the SQL and Spark Context

    path_to_csv = "/job/"+config.file
    sc = SparkSession.builder.master("local[1]") \
                        .appName('Bigdatagroup27.com') \
                        .getOrCreate()
    sqlContext = SQLContext(sc)
    try:
        path_to_csv = "/job/"+config.file
        df = sqlContext.read.csv(path_to_csv, header=True)
    except:
        path_to_csv = config.file
        df = sqlContext.read.csv(path_to_csv, header=True)
    #Drop the forbidden columns
    df=df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

    #Delete cancelled flies
    df=df.filter(df.CancellationCode.isNull()).drop()
    df=df.drop("Cancelled", "CancellationCode")

    #Correct the types of the variables
    with_letters = ["TailNum", "UniqueCarrier", "Origin", "Dest"]
    df = df.select([col(c).cast(IntegerType()).alias(c) if c not in with_letters else col(c) for c in df.columns])

    #If flag = True: perform the data analysis
    if config.analysis:
         eda.performExploratoryAnalysis(df)

    #Data preprocessing
    df=preprocessing.date_preprocess(df)
    #Drop variables
    df = df.drop("Date", "Distance", "FlightNum", "CRSElapsedTime", "TailNum", "CRSDepTime", "CRSArrTime", "Year")

    #If there is nulls in target variable (ArrDelay)
    df = df.na.drop(subset=["ArrDelay"])
    #df=preprocessing.filter_null(df)
    df=preprocessing.categoricalToNumerical(df)

    #Modeling
    df=preprocessing.create_features_vector(df,model)
    df_features = df.select(["features", "ArrDelay"])
    train_set,test_set=preprocessing.split_set(df,trainsize,testsize)

    if model=="RegularizedLinearRegression":
        trained_model, train_predictions, test_predictions = models.select_RegularizedLinearRegressionModel(train_set,test_set)
        statistics.print_linear_regression_summary(trained_model)

    elif model=="DecisionTreeRegression":
        trained_model, train_predictions, test_predictions = models.select_DecisionTreeRegressionModel(train_set,test_set)

    else:
        trained_model, train_predictions, test_predictions = models.select_LinearRegressionModel(train_set,test_set)
        statistics.print_linear_regression_summary(trained_model)

    statistics.print_summary(test_predictions)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="2008.csv.bz2")
    parser.add_argument("--analysis", default=False, action="store_true")
    parser.add_argument("--trainsize", type=float, default=0.7)
    parser.add_argument("--testsize", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="LinearRegression")
    config = parser.parse_args()

    run_application(config, verbose=True)
