# Libraries used
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.stat import Statistics

#Load data and the SQL and Spark Context
path_to_csv = '2008.csv.bz2'
sc = SparkContext.getOrCreate();
sqlContext = SQLContext(sc)
df = sqlContext.read.csv(path_to_csv, header=True)

#############
# Functions #
#############

def day_of_year(date):
  return datetime.timetuple(datetime.combine(date, datetime.min.time())).tm_yday

#################
# Data cleaning #
#################

#Drop the forbidden columns
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

#Delete cancelled flies
df=df.filter(df.CancellationCode.isNull()).drop()
df=df.drop("Cancelled",
           "CancellationCode")

#Correct the types of the variables
categorical = ["Year", "Month", "DayofMonth", "TailNum", "UniqueCarrier", "Origin", "Dest"]
df = df.select([col(c).cast(IntegerType()).alias(c) if c not in categorical else col(c) for c in df.columns])

###################
# Data processing #
###################

#Creates a UDF to transform the date to an integer that represents the day of the year.
df = df.withColumn("Date", concat_ws('-',df.Year,df.Month,df.DayofMonth))
dayOfYear = udf(day_of_year, returnType=IntegerType())
df = df.withColumn("Date", to_date(df["Date"]))
df = df.withColumn("DayOfYear", dayOfYear(df.Date))

#Drop unusefull variables
df=df.drop("Year",
           "Month",
           "DayOfMonth",
           "Date",
           "FlightNum")

categorical = [ "UniqueCarrier", "Origin", "Dest"]



#Treatment of nulls
#If there is nulls in target variable (ArrDelay)
df = df.na.drop(subset=["ArrDelay"])

#For numerical: mean
#Fill nulls with mean value of the column excluding variables on list categorical
stats = df.agg(*(avg(c).alias(c) for c in df.columns if c not in categorical))
df = df.na.fill(stats.first().asDict())

#For categorical: most common
for c in categorical:
  frec = df.groupBy(c).count().orderBy('count', ascending=False)
  mode = frec.first()[c]
  df = df.na.fill(value=mode,subset=[c])
