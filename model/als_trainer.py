import findspark 
findspark.init() # Used to properly connect to Spark Master
import pyspark   # Only run after findspark.init()

from pyspark import SparkContext, SparkConf


from pyspark.sql import Row, SQLContext, Window
from pyspark.sql.functions import udf, desc, percent_rank
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler


class ALSTrainer:

  def __init__(self, maxIter=10, regParam=0.1, userCol="user_id", 
               itemCol="item_id", ratingCol="frequency", coldStartStrategy="drop", rank=5):
    #self.sc, self.sqlContext = (sparkContext, sqlContext) if sparkContext is not None else  self.get_spark_context()
    
    # Note cold start strategy to 'drop' to avoid 'nan' during evaluation 
    self.als = ALS(maxIter=maxIter, regParam=regParam, userCol=userCol, 
              itemCol=itemCol, ratingCol=ratingCol, implicitPrefs=True,
              coldStartStrategy=coldStartStrategy, rank=rank)

    self.model = None

  def stop_spark_context(self):
    # If you need to stop the current context first:
    #self.sc.stop()
    pass

  def get_spark_context(self):
    # Create new config
    # Spark might decide to join too many rows of DF to the same thread, 
    # if running in a bigger server add the folloging line
    conf = SparkConf().set("spark.driver.maxResultSize", "20g") 

    sc         = SparkContext(appName="ALSTrainer", sparkHome="/usr/local/spark", conf=conf)    
    sqlContext =  SQLContext(sc)
    return sc, sqlContext

  @staticmethod
  def load(sqlContext, split_train=True):
    schema = StructType([
      StructField("user_id", IntegerType()), StructField("item_id", StringType()),
      StructField("latest_timestamp", IntegerType()), StructField("frequency", IntegerType()),
    ])
    item_hist    = sqlContext.read.csv("data/item_history.tsv",sep="\t", header=True, schema=schema)
    # Split item history in train, test and set validation aside for later
    
    def get_key_value(p):
      # for grouping as key value
      # Drop latest_timestamp from columns and erase the 'I' in front
      # of the item ids for faster evaluation.
      return (int(p[0]),int(p[1].replace("I",""))), float(p[3])
    
    if split_train:
      #ihist_train = df.where("rank <= .7").drop("rank")
      ihist_train = item_hist.orderBy("latest_timestamp").limit(4423962)
      frequencyRDD = ihist_train.rdd.map(get_key_value).reduceByKey(lambda x: sum(list(x)) )
      frequencyRDD = frequencyRDD.map(lambda x: Row(user_id=int(x[0][0]),item_id=int(x[0][1]),frequency=float(x[1])) )
      frequencies = sqlContext.createDataFrame(frequencyRDD)
    else:
      ihist_train_test = item_hist.orderBy("latest_timestamp").limit(4423962 + 1263989)
      frequencyRDD = ihist_train_test.rdd.map(get_key_value).reduceByKey(lambda x: sum(list(x)) )
      frequencyRDD = frequencyRDD.map(lambda x: Row(user_id=int(x[0][0]),item_id=int(x[0][1]),frequency=float(x[1])) )
      frequencies = sqlContext.createDataFrame(frequencyRDD)

    return frequencies
   
  def fit(self, train_df=None):
    self.model = self.als.fit(train_df)
  
  def evaluate(self, test_df=None):
    # Evaluate the model by computing the RMSE on the test data
    predictions = self.model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="frequency", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    #print("Root-mean-square error = " + str(rmse))
    return rmse

  def write_recommomendation_to_csv(self, userRecs, filepath="data/spark_asl.tsv"):
    """
    userRecs is a Dataframe with recommended Items for users. 
    It is easier if it is the return from `model.recommendForAllUsers(k)` or `model.recommendForUserSubset`
    """

    def array_to_string(my_list):
        return ','.join([f"{elem.item_id}" for elem in my_list])

    array_to_string_udf = udf(array_to_string, StringType())

    to_csv = userRecs.withColumn('recommended_items', array_to_string_udf(userRecs["recommendations"])).drop("recommendations")
    to_csv.write.csv(filepath, sep="\t", mode="overwrite")


  def recommend_for_user_subset(self, user_list, numItems):
    """
    Returns a spark dataframe
    """
    return self.model.recommendForUserSubset(user_list, numItems=numItems)

  def recommendForAllUsers(self, numItems):
    return self.model.recommendForAllUsers(numItems)