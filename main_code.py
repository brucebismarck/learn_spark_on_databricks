''' This project is from kaggle competition, I follow Alex logic, showed in hashtag next line'''
# https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# STEP 1: change label 'saleprice' by log function
# STEP 2: For the columns with high skewness, use log function on them to reduce the skewness
# STEP 3: One hot encoding to convert categorical data to dummy variables
# STEP 4: The out put of one hot encoding is a vector with specific type A, and the input 
#         of linear regression is vector with specific type B. So convert data type in this step
# STEP 5: Linear regression


#from pyspark.sql import SQLContext                                               # There is no need to import SQLContext, because we mainly use
                                                                                  # dataframe rather than simple RDD.
from pyspark.sql.functions import *                                               # import all sql functions, discription of sql function of spark 2.0.0 
                                                                                  # could be find here
from pyspark.ml.feature import SQLTransformer, OneHotEncoder, StringIndexer       # These are machine learning data extraction functions
from pyspark.mllib.regression import LabeledPoint                                 # This function
from pyspark.sql import SQLContext                                                # Ok this is used in the later mapping functions....
from pyspark.ml.linalg import VectorUDT                                           # used in STEP 4
from pyspark.mllib.linalg import Vectors as MLLibVectors                          # used in STEP 4
from pyspark.ml.regression import LinearRegression                                # used in STEP 5


#sqlContext = SQLContext(sc)
df = spark.read.options(header = 'true',inferSchema = 'true').\                   # read in data, you can upload data to tables(it is in the sidebar) 
      csv('/FileStore/tables/r31g95231487183714154/train.csv')                    # and change the csv direction.
                                                                                  # without infreSchema, the dataframe reading will give you all string data.
                                                                                  # rather tahn categorical and numeric data
df = df.withColumn('LotFrontage', df.LotFrontage.cast("integer"))                 # these four rows change the numeric data with NA values to integer.
df = df.withColumn('GarageYrBlt', df.GarageYrBlt.cast("integer"))                 # and use withColumn to change the dataframe without get a new dataframe or use join
df = df.withColumn('MasVnrArea', df.MasVnrArea.cast("integer"))
df = df.withColumn('BsmtFinType2', df.BsmtFinType2.cast("integer")) 
df = df.na.fill({'LotFrontage':70.05, 'GarageYrBlt': 1979, 'MasVnrArea':0, 'BsmtFinType2': 0}) # fill na each column with mean or 0. 
                                                                                  # The reason why I dont use column mean is because I dont know how to return
                                                                                  # number...
df = df.withColumn('SalePrice', log(df['SalePrice']))                             # use log on SalePrice because

string_col = []                                                                   # Find string columns and numeric columns
num_col = []
for item in df.dtypes:                                                            # Find all string columns and numeric columns via for loop 
  if item[1] == 'string':                                 
    string_col.append(item[0])
  else:
    num_col.append(item[0])

print df.select(num_col).count()                                                  # Through these two lines, 
print df.select(num_col).dropna().count()                                         # there is no missing value in numerical features

df_1 = df.select(num_col)                                                         # To find the numeric columns with large skewness
skewness_list = []
for item in num_col:
   skewness_list.append(df_1.agg(skewness(df_1[item])).head())
col_list = []                                                                     
skew_list = []
for i in range(len(skewness_list)):
  col_list.append(str(skewness_list[i]).replace('(',')').replace('=',')').split(')')[2])
  skew_list.append(str(skewness_list[i]).replace('(',')').replace('=',')').split(')')[4])
large_skew = []
for item in range(len(skew_list)):
  if float(skew_list[item]) > 0.75 or float(skew_list[item])< -0.75 :
    large_skew.append(col_list[item])
for item in large_skew:
  df = df.withColumn(item, log(df[item]+1))                                      # apply log function on columns with large skewness

df_str = df.select('Id')                                                         # one hot encoding 
for item in string_col:
  stringIndexer = StringIndexer(inputCol=item, outputCol= item + ' index' ).fit(df).transform(df)
  encoder = OneHotEncoder(inputCol= item + ' index', outputCol=item + ' onehot').transform(stringIndexer).select('Id',item + ' onehot')
  df = df.drop(item)
  df_str = df_str.join(encoder,'Id')                                             # the output of one hot encoding is a vector
                                                                                 # unlike r or python, which ask input should be a matrix with 
                                                                                 # many columns. The each line of MLlib features input is a vector.
df_price = df.join(df_str,'Id','inner').select('Id','SalePrice')
df_variable = df.drop('SalePrice')

data = df.rdd.map(lambda r: LabeledPoint(r[-1],[r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15],r[16],r[17],r[18],r[19],r[20],r[21],r[22],r[23],r[24],r[25],r[26],r[27],r[28],r[29],r[30],r[31],r[32],r[33],r[34],r[35],r[36],r[37],r[38]])).toDF()                                                        
# I don't know how to make it here. just use stupid method...

as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDT())          # define function to change datatype input vector.
result = data.withColumn("features", as_ml("features"))

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)             # linear model and parameters

# Fit the model
lrModel = lr.fit(result)                                                         # model fit on data
print("Coefficients: " + str(lrModel.coefficients))                               # print parameters
print("Intercept: " + str(lrModel.intercept))                                    # print intercept
