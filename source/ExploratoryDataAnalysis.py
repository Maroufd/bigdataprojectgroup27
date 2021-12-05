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
    print("DATA EXPLORATION")
    print("Number of columns: " + str(len(df.columns)))
    print("Number of rows: " + str(df.count()))
    print(df.printSchema())

    #Exploration of the variables

    #Numerical
    print("Summary of numerical variables:")
      
    #Exploration of the variables
    categorical = ["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "FlightNum", "TailNum", "UniqueCarrier", "Origin", "Dest"]
    
    #Numerical
    numerical = df.drop(*categorical)
    for col in numerical.columns:
        numerical.describe(col).show()

    #Categorical
    print("Frecuency Tables of categorical variables:")
    for col in categorical:
      df.groupBy(col).count().orderBy('count', ascending=False).show()

    print("Number of nulls in the columns:")
    df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()
    
    #Correlation
    print("Correlations:\r\n")
    for col in numerical.columns:
        print("Correlation to ArrDelay for " + col + ": " + str(numerical.stat.corr("ArrDelay", col)))
