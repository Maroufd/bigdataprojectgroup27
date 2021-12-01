#########################
# Data exploration file #
#########################

'''If executed, this file generates a file called 'data_exploration.txt'
with the exploratory analysis of the data provided'''

from pyspark.sql.functions import *
from pyspark.mllib.stat import Statistics

def write_table(table, name_of_file):
  '''Writes a table in a .txt
  Input: dataframe and string with the name of the file'''

  with open(name_of_file,'a') as f:
    f.write(" | ".join([col for col in table.columns]) + '\r\n')
    rows = table.rdd.collect()
    f.write("\r\n".join([" | ".join(map(str, row)) for row in rows]))
    f.write("\r\n")
  f.close()

def write_correlations(table, variable, name_of_file):
  '''Writes a correlations between variable and the rest of the columns of a dataframe in a .txt
  Input: dataframe, string with the name of the variable and string with the name of the file'''

  with open(name_of_file,'a') as f:
    f.write("Correlations:\r\n')
    for col in table.columns:
      f.write("Correlation to " + variable + " for " + col + ": " + str(table.stat.corr(variable, col)) + '\r\n')
  f.close()

def main(df):

  #Exploration of the data base
  with open('data_exploration.txt','w') as f:
    f.write("DATA EXPLORATION" + '\r\n\r\n')
    f.write("Number of columns: " + str(len(df.columns)) + '\r\n')
    f.write("Number of rows: " + str(df.count()) + '\r\n\r\n')
    f.write("Summary of numerical variables:" + '\r\n')
  f.close()

  #Exploration of the variables

  #Numerical
  numerical = df.drop(*categorical)
  for col in numerical:
    desc = numerical.describe(col)
    write_table(desc, 'data_exploration.txt')

  #Categorical
  with open('data_exploration.txt','a') as f:
    f.write("\r\n"+ "Frecuency Tables of categorical variables:" + '\r\n')
  f.close()

  for col in categorical:
    frec = df.groupBy(col).count().orderBy('count', ascending=False)
    write_table(frec, 'data_exploration.txt')

  #Correlations
  write_correlations(numerical, "ArrDelay", 'data_exploration.txt')


if __name__ == "__main__":
  main(df)
