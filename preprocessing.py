###################
# Data processing #
###################

'''This file contains the functions to perform the data processing on a dataframe'''

from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import VectorAssembler

def day_of_year(date):
    '''Converts a variable with format 'dd-mm-YYYY' to a number in [1, 365]'''

    return datetime.timetuple(datetime.combine(date, datetime.min.time())).tm_yday

def date_preprocess(df):
    '''Convert the variables Year Month and DayofMonth into a number in [0, 365] representing the day of the year
    Input: dataframe
    Output: dataframe'''

    df = df.withColumn("Date", concat_ws('-',df.Year,df.Month,df.DayofMonth))
    dayOfYear = udf(day_of_year, returnType=IntegerType())
    df = df.withColumn("Date", to_date(df["Date"]))
    df = df.withColumn("DayOfYear", dayOfYear(df.Date))
    df=df.drop("Year", "Month", "DayOfMonth", "Date")
    return df

def filter_null(df):
    '''Treatment of nulls
    Input: dataframe
    Output: dataframe'''

    #If there is nulls in target variable (ArrDelay)
    df = df.na.drop(subset=["ArrDelay"])

    #For numerical: mean
    #Fill nulls with mean value of the column excluding variables on list categorical
    categorical = ["TailNum", "UniqueCarrier", "Origin", "Dest"]
    #stats = df.agg(*(avg(c).alias(c) for c in df.columns if c not in categorical))
    #withoutNulls = df.na.fill(stats.first().asDict())

    #For categorical: most common
    for c in categorical:
        print(c)
        frec = df.groupBy(c).count()
        print(frec)
        frec= frec.orderBy('count', ascending=False)
        frec.show()
        mode = frec.first()[c]
        print(mode)
        df = df.na.fill(value=mode,subset=[c])

    return df

def categoricalToNumerical(df):
    '''Convert categorical variables to integer
    Input: dataframe
    Outpu: dataframe'''

    categorical = ["TailNum", "UniqueCarrier", "Origin", "Dest"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_index").fit(df) for c in df.columns if c in categorical]
    pipeline = Pipeline(stages=indexers)
    df_types = pipeline.fit(df).transform(df)
    df_types = df_types.drop(*categorical)
    return df

def create_features_vector(df, modelSelected):
    # You can also change DepDelay for the most correlated var
    if modelSelected == "SimpleLinearRegression":
      vectorAssembler = VectorAssembler(inputCols = ["DepDelay"], outputCol = "features")
      df = vectorAssembler.transform(df)

    elif modelSelected == "MultipleLinearRegressionInteraction":
      df = df.withColumn("DepDelay_TaxiOut", df["DepDelay"]*df["TaxiOut"])
      df = df.withColumn("DepDelay_DepTime", df["DepDelay"]*df["DepTime"])

      vectorAssembler = VectorAssembler(inputCols = input_cols, outputCol = "features")
      df = vectorAssembler.transform(df)

    # Else = for the default/elasticnet/decisiontree models
    else:
      input_cols = df.columns
      input_cols.remove("ArrDelay")

      vectorAssembler = VectorAssembler(inputCols = input_cols, outputCol = "features")
      df = vectorAssembler.transform(df)

    return df


def split_set(df_features,trainpercent,testpercent):
    sets = df_features.randomSplit([trainpercent, testpercent])
    train_set = sets[0]
    test_set = sets[1]
    return train_set,test_set
