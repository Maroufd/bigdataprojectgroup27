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
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c not in categorical))
    withoutNulls = df.na.fill(stats.first().asDict())

    #For categorical: most common
    for c in categorical:
      frec = df.groupBy(c).count().orderBy('count', ascending=False)
      mode = frec.first()[c]
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
