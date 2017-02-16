#from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.ml.feature import SQLTransformer, OneHotEncoder, StringIndexer

#sqlContext = SQLContext(sc)
df = spark.read.options(header = 'true',inferSchema = 'true').csv('/FileStore/tables/r31g95231487183714154/train.csv')
df = df.withColumn('LotFrontage', df.LotFrontage.cast("integer"))
df = df.withColumn('GarageYrBlt', df.GarageYrBlt.cast("integer"))
df = df.withColumn('MasVnrArea', df.MasVnrArea.cast("integer"))
df = df.withColumn('BsmtFinType2', df.BsmtFinType2.cast("integer"))
df = df.na.fill({'LotFrontage':70.05, 'GarageYrBlt': 1979, 'MasVnrArea':0, 'BsmtFinType2': 0})
df = df.withColumn('SalePrice', log(df['SalePrice']))

string_col = []
num_col = []
for item in df.dtypes:
  if item[1] == 'string':
    string_col.append(item[0])
  else:
    num_col.append(item[0])

print df.select(num_col).count()
print df.select(num_col).dropna().count() # there is no missing value in numerical features

from pyspark.sql.functions import *
df_1 = df.select(num_col)
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
  df = df.withColumn(item, log(df[item]+1))     # this withColumn is nothing but awesome...

df_str = df.select('Id')
for item in string_col:
  stringIndexer = StringIndexer(inputCol=item, outputCol= item + ' index' ).fit(df).transform(df)
  encoder = OneHotEncoder(inputCol= item + ' index', outputCol=item + ' onehot').transform(stringIndexer).select('Id',item + ' onehot')
  df = df.drop(item)
  df_str = df_str.join(encoder,'Id')

df_price = df.join(df_str,'Id','inner').select('Id','SalePrice')
df_variable = df.drop('SalePrice')

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext
def f2Lp(inStr):
    return (inStr[0], Vectors.dense(inStr[1]))

data = df.rdd.map(lambda r: LabeledPoint(r[-1],[r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15],r[16],r[17],r[18],r[19],r[20],r[21],r[22],r[23],r[24],r[25],r[26],r[27],r[28],r[29],r[30],r[31],r[32],r[33],r[34],r[35],r[36],r[37],r[38]])).toDF()

from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors

as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDT())
result = data.withColumn("features", as_ml("features"))

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(result)
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))
