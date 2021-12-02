import argparse
import preprocessing
import models
import statistics
import ExploratoryDataAnalysis.py as eda

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
    path_to_csv = config.file
    sc = SparkContext.getOrCreate();
    sqlContext = SQLContext(sc)
    df = sqlContext.read.csv(path_to_csv, header=True)

    #Drop the forbidden columns
    df=df.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
    
    #Delete cancelled flies
    df=df.filter(df.CancellationCode.isNull()).drop()
    df=df.drop("Cancelled", "CancellationCode", "FlightNum")

    #Correct the types of the variables
    categorical = ["Year", "Month", "DayofMonth", "TailNum", "UniqueCarrier", "Origin", "Dest"]
    df = df.select([col(c).cast(IntegerType()).alias(c) if c not in categorical else col(c) for c in df.columns])

    #If flag = True: perform the data analysis
    if config.analysis:
         eda.performExploratoryAnalysis(df)
            
    #Data preprocessing
    df=preprocessing.date_preprocess(df)
    df=preprocessing.filter_null(df)
    df=preprocessing.categoricalToNumerical(df)

    #Modeling

    if config.model=="MultipleLinearRegression":
        lr_model,trainingSummary, test_result=models.LinearRegressionmodel(train_set,test_set)

    statistics.print_linear_regression_summary(lr_model)

    statistics.print_training_summary(trainingSummary)

    statistics.print_test_summary(test_result)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="2008.csv.bz2")
    parser.add_argument("--analysis", default=False, action="store_true")
    parser.add_argument("--trainsize", type=float, default=0.7)
    parser.add_argument("--testsize", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="LinearRegression")
    config = parser.parse_args()
    
    run_application(config, verbose=True)
