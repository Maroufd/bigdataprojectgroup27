# Libraries used

from pyspark import *
from pyspark.sql.context import SQLContext
from pyspark.sql.types import IntegerType
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression

# Loads data and the SQL and Spark Context
path_to_csv = '/dataverse_files/2008.csv.bz2'
sc = SparkContext.getOrCreate();
sqlContext = SQLContext(sc)
df = sqlContext.read.csv(path_to_csv, header=True)

# Drops the forbidden columns
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

print("Number of rows: ", df.count())
print('Number of columns: ', len(df.columns))

df.show(10)

df = df.withColumn("Date", concat_ws('-',df.Year,df.Month,df.DayofMonth))

def day_of_year(date):
  return datetime.timetuple(datetime.combine(date, datetime.min.time())).tm_yday

dayOfYear = udf(day_of_year, returnType=IntegerType())

df = df.withColumn("Date", to_date(df["Date"]))
df = df.withColumn("DayOfYear", dayOfYear(df.Date))

df=df.filter(df.CancellationCode.isNull()).drop()
df=df.drop("Cancelled",
           "CancellationCode",
           "Year",
           "Month",
           "DayOfMonth",
           "Date",
           "FlightNum")

print('Types of the columns: ')
df.printSchema()

print('Number of nulls in the columns: ')
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

print('Number of NA in the columns: ')
df.select([count(when(df[c] == "NA", True)).alias(c) for c in df.columns]).show()

# If the response variable is null, filter the row
df=df.filter(df.ArrDelay.isNotNull() & (df.ArrDelay != "NA"))

# TO DO: Manage nulls in a smarter way
df=df.filter(df.TailNum.isNotNull() & (df.ArrDelay != "NA"))

for col in df.columns:
  isNumeric = col not in ["TailNum", "UniqueCarrier", "Origin", "Dest"]
  if isNumeric:
    df = df.withColumn(col, df[col].cast(IntegerType()))
  else:
    indexer = StringIndexer(inputCol=col, outputCol=str(col + "_id"))
    df = indexer.fit(df).transform(df)
    df = df.withColumn(str(col + "_id"), df[str(col + "_id")].cast(IntegerType()))

# Drop non-numerical variables
df = df.drop("TailNum", "UniqueCarrier", "Origin", "Dest")

print('Types of the columns: ')
df.printSchema()

"""for col in df.columns:
  df.groupBy(col).count().show()"""

"""df_prueba = df.withColumn("ArrDelay_ID", df["ArrDelay"].cast(IntegerType()))"""

"""df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()"""

"""df.groupBy("ArrDelay").count().sort(col("count").desc()).show(100)"""

df.show(10)

"""df.crosstab('DayOfWeek', 'ArrDelay').show()"""

corr_dow_arrdelay = df.stat.corr('DayOfWeek', 'ArrDelay')
print(corr_dow_arrdelay)

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "features"
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
corr_matrix_df = Correlation.corr(df_vector, vector_col)

# Autocorrelation matrix rounded
for i in range(0, n_corr):
  corr_matrix[:, i] = corr_matrix_df.collect()[0]["pearson({})".format(vector_col)].values[(n_corr*i):(n_corr*(i+1))]

print(type(corr_matrix))
rdd_matrix = sc.parallelize(corr_matrix.tolist())
df_autocorr = sqlContext.createDataFrame(rdd_matrix, schema=df.columns)
for c in df_autocorr.columns:
  df_autocorr = df_autocorr.select("*", round(col(c), 8))

df_autocorr.show()

df.show()

# Create features vector
input_cols = df.columns
input_cols.remove("ArrDelay")

vectorAssembler = VectorAssembler(inputCols = input_cols, outputCol = "features")
df = vectorAssembler.transform(df)
df_features = df.select(["features", "ArrDelay"])

df_features.show()

# Splits data in training/test set
sets = df_features.randomSplit([0.7, 0.3])
train_set = sets[0]
test_set = sets[1]

# Trains a Linear Regression model
lr = LinearRegression(featuresCol = "features", labelCol="ArrDelay")
lr_model = lr.fit(train_set)

print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE (train): ", trainingSummary.rootMeanSquaredError)
print("R2 (train): ", trainingSummary.r2)
print("Adjusted R2 (train): ", trainingSummary.r2adj)

test_result = lr_model.evaluate(test_set)

print("RMSE (test): ", test_result.rootMeanSquaredError)
print("R2 (test): ", test_result.r2)
print("Adjusted R2 (test): ", test_result.r2adj)

# I just wanted to try statistical data analysis stuff
# Let's try an interaction!
df = df.drop("features")
df_interaction = df.withColumn("DepTime_DepDelay", df["DepTime"]*df["DepDelay"])
df_interaction = df.withColumn("DepTime_TaxiOut", df["DepTime"]*df["TaxiOut"])
df_interaction = df.withColumn("TaxiOut_DepDelay", df["TaxiOut"]*df["DepDelay"])
df_interaction.show()

input_cols_interaction = df_interaction.columns
input_cols_interaction.remove("ArrDelay")

# Create features vector
vectorAssembler = VectorAssembler(inputCols = input_cols_interaction, outputCol = "features")
df_interaction = vectorAssembler.transform(df_interaction)
df_features_interaction = df_interaction.select(["features", "ArrDelay"])

# Splits data in training/test set
sets_interaction = df_features_interaction.randomSplit([0.7, 0.3])
train_set_interaction = sets_interaction[0]
test_set_interaction = sets_interaction[1]

# Trains a Linear Regression model
lr_interaction = LinearRegression(featuresCol = "features", labelCol="ArrDelay")
lr_model_interaction = lr_interaction.fit(train_set_interaction)

print("Coefficients: " + str(lr_model_interaction.coefficients))
print("Intercept: " + str(lr_model_interaction.intercept))

trainingSummary_interaction = lr_model_interaction.summary
print("RMSE (train): ", trainingSummary_interaction.rootMeanSquaredError)
print("R2 (train): ", trainingSummary_interaction.r2)
print("Adjusted R2 (train): ", trainingSummary_interaction.r2adj)

test_result_interaction = lr_model_interaction.evaluate(test_set_interaction)

print("RMSE (test): ", test_result_interaction.rootMeanSquaredError)
print("R2 (test): ", test_result_interaction.r2)
print("Adjusted R2 (test): ", test_result_interaction.r2adj)
