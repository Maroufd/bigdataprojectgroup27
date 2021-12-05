#########################
# Data exploration file #
#########################

'''If executed, this file generates the exploratory analysis of the data provided'''

from pyspark.sql.functions import *
from pyspark.mllib.stat import Statistics

def performExploratoryAnalysis(df):
    '''Performs an exploratory data analysis of df
    Input: dataframe'''

    #Exploration of the data base
    print("----------------------------------------------------------")
    print("--------------------DATA EXPLORATION----------------------")
    print("----------------------------------------------------------")
    print("Number of columns: " + str(len(df.columns)))
    print("Number of rows: " + str(df.count()))
    print(df.printSchema())
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    #Exploration of the variables

    #Numerical
    print("----------------------------------------------------------")
    print("-------------SUMMARY OF NUMERICAL VARIABLES---------------")
    print("----------------------------------------------------------")
    #Exploration of the variables
    categorical = ["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "FlightNum", "TailNum", "UniqueCarrier", "Origin", "Dest"]

    #Numerical
    numerical = df.drop(*categorical)
    for col in numerical.columns:
        numerical.describe(col).show()
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    #Categorical
    print("----------------------------------------------------------")
    print("--------FRECUENCY TABLES OF CATEGORICAL VARIABLES---------")
    print("----------------------------------------------------------")
    for col in categorical:
      df.groupBy(col).count().orderBy('count', ascending=False).show()

    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("-------------NUMBER OF NULLS IN THE COLUMNS---------------")
    print("----------------------------------------------------------")
    df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    #Correlation
    print("----------------------------------------------------------")
    print("----------------------CORRELATIONS------------------------")
    print("----------------------------------------------------------")
    for col in numerical.columns:
        print("Correlation to ArrDelay for " + col + ": " + str(numerical.stat.corr("ArrDelay", col)))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
