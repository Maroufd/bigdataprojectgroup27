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
# Loads data and the SQL and Spark Context

def readandcreate(raw_data):
    path_to_csv = raw_data
    sc = SparkContext.getOrCreate();
    sqlContext = SQLContext(sc)
    df = sqlContext.read.csv(path_to_csv, header=True)
    return df

# Drops the forbidden columns

def drop_forbidden(df):
    df=df.drop("ArrTime",
               "ActualElapsedTime",
               "AirTime",
               "TaxiIn",
               "Diverted",
               "CarrierDelay",
               "WeatherDelay",
               "NASDelay",
               "SecurityDelay",
               "LateAircraftDelay")
    return df

def day_of_year(date):
  return datetime.timetuple(datetime.combine(date, datetime.min.time())).tm_yday

def date_preprocess(df):
    df = df.withColumn("Date", concat_ws('-',df.Year,df.Month,df.DayofMonth))
    dayOfYear = udf(day_of_year, returnType=IntegerType())
    df = df.withColumn("Date", to_date(df["Date"]))
    df = df.withColumn("DayOfYear", dayOfYear(df.Date))
    return df

def drop_not_needed(df):
    df=df.filter(df.CancellationCode.isNull()).drop()
    df=df.drop("Cancelled",
           "CancellationCode",
           "Year",
           "Month",
           "DayOfMonth",
           "Date",
           "FlightNum")
    return df

def filter_null(df):
    # If the response variable is null, filter the row
    df=df.filter(df.ArrDelay.isNotNull() & (df.ArrDelay != "NA"))
    # TO DO: Manage nulls in a smarter way
    df=df.filter(df.TailNum.isNotNull() & (df.ArrDelay != "NA"))
    return df
def coltoint(df):
    for col in df.columns:
        isNumeric = col not in ["TailNum", "UniqueCarrier", "Origin", "Dest"]
        if isNumeric:
            df = df.withColumn(col, df[col].cast(IntegerType()))
        else:
            indexer = StringIndexer(inputCol=col, outputCol=str(col + "_id"))
            df = indexer.fit(df).transform(df)
            df = df.withColumn(str(col + "_id"), df[str(col + "_id")].cast(IntegerType()))
    df = df.drop("TailNum", "UniqueCarrier", "Origin", "Dest")
    return df

def convert_to_vector(df):
    vector_col = "features"
    assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    return df_vector

def get_correlation_matrix(df_vector,vector_col):
    corr_matrix_df = Correlation.corr(df_vector, vector_col)
    for i in range(0, n_corr):
        corr_matrix[:, i] = corr_matrix_df.collect()[0]["pearson({})".format(vector_col)].values[(n_corr*i):(n_corr*(i+1))]
    rdd_matrix = sc.parallelize(corr_matrix.tolist())
    df_autocorr = sqlContext.createDataFrame(rdd_matrix, schema=df.columns)
    for c in df_autocorr.columns:
        df_autocorr = df_autocorr.select("*", round(col(c), 8))
    return df_autocorr

def create_features_vector(df):
    input_cols = df.columns
    input_cols.remove("ArrDelay")
    vectorAssembler = VectorAssembler(inputCols = input_cols, outputCol = "features")
    df = vectorAssembler.transform(df)
    df_features = df.select(["features", "ArrDelay"])
    return df_features

def split_set(df_features,trainpercent,testpercent):
    sets = df_features.randomSplit([trainpercent, testpercent])
    train_set = sets[0]
    test_set = sets[1]
    return train_set,test_set


def interactionprocess(df):
    df = df.drop("features")
    df_interaction = df.withColumn("DepTime_DepDelay", df["DepTime"]*df["DepDelay"])
    df_interaction = df.withColumn("DepTime_TaxiOut", df["DepTime"]*df["TaxiOut"])
    df_interaction = df.withColumn("TaxiOut_DepDelay", df["TaxiOut"]*df["DepDelay"])
    input_cols_interaction = df_interaction.columns
    input_cols_interaction.remove("ArrDelay")
    vectorAssembler = VectorAssembler(inputCols = input_cols_interaction, outputCol = "features")
    df_interaction = vectorAssembler.transform(df_interaction)
    df_features_interaction = df_interaction.select(["features", "ArrDelay"])
    return df_features_interaction
