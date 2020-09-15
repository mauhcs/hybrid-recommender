"""
This file evaluates a recommendation list written by a recommendation model. 
The way it evaluates is by calculating the nDCG@72 of each user/row of the input file, and averaging out. 

Also, while calculating the nDCG@72, the processes going through the dataframe are also computing the entropy of the 
recommendation list.

This function was written using pandas and python's multiprocess library, but it is not as efficient as pySpark would be.

So a natural TODO: Rewrite using pySpark
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, Manager

from utils.metrics import nDCG_at_k

class Var:
  def __init__(self, value, recommends=None, y_test=None):
    self.value = value
    self.y_test = y_test
    self.recommends = recommends
    self.ci = defaultdict(lambda: 0)
    self.bigI = set()

GlobalVar = Var(0)

def work(i, Q):
  row = GlobalVar.recommends.iloc[i]
  user_id = row.user_id
  irec    = row.recommended_items.split(",")
  
  # Parameters for Entropy@72:
  ci = defaultdict(lambda: 0)
  bigI = set(irec)
  for item in irec:
    ci[item] += 1

  y_truth = GlobalVar.y_test[GlobalVar.y_test.user_id == user_id].items_actioned_on.iloc[0].split(",")
  if all([x == "0" for x in  y_truth]):
    Q.put([bigI, dict(ci)]) # Skipping for dcg
    return
  y_pred = GlobalVar.recommends[GlobalVar.recommends.user_id == user_id].recommended_items.iloc[0].split(",")
  
  relevance = []
  for yp in y_pred:
    relevance.append(1 if yp in y_truth else 0)

  Q.put([user_id, nDCG_at_k(relevance), bigI, dict(ci)])


def waiter(Q):
  #print("Starting Waiter")

  
  nDCGs = []
  # Paramters for the entropy@72
  bigI = set()
  ci = defaultdict(lambda: 0)
  
  i = 0
  while 1:
    m = Q.get()
    if m == "kill":
      print("\nStoping Writer")
      break

    elif len(m) == 2: 
      # If the real data had zero items for a given user we count only 
      # the recommendations towards the entropy value
      bigI  = set(bigI)|set(m[0])
      _ci   = m[1] 
      for key, val in _ci.items():
        ci[key] += val
      if i % 1000 == 0:
        print(f"{i:,} / {GlobalVar.value:,} ({i/GlobalVar.value * 100:.2f})%", end="\r", flush=True)
      i+=1

    elif len(m) == 4:
      user_id   = m[0]
      nDCG_at_k = m[1]
      nDCGs.append([user_id, nDCG_at_k])
      
      bigI  = set(bigI)|set(m[2])
      _ci   = m[3] 
      for key, val in _ci.items():
        ci[key] += val

      if i % 1000 == 0:
        print(f"{i:,} / {GlobalVar.value:,} ({i/GlobalVar.value * 100:.2f})%", end="\r", flush=True)
      i+=1

  Q.put([nDCGs, bigI, dict(ci) ])



def spark_filepath_to_pandas(filepath, sqlContext=None):
  """
  If the file given was written with pySpark, we need to instantiate a context and load the file via pySpark.
  This function loads as a pandas dataframe the pySpark file given by `filepath`

  Returns a pandas dataframe 
  """

  if sqlContext is None:
    import findspark
    findspark.init()
    import pyspark # only run after findspark.init()    
    from pyspark.sql import SQLContext
    from pyspark import SparkContext, SparkConf


    conf = SparkConf().set("spark.driver.maxResultSize", "20g")
    sc = SparkContext(appName="RecEvalate", sparkHome="/usr/local/spark", conf=conf)    
    sqlContext = SQLContext(sc)


  from pyspark.sql.types import StructType, StructField
  from pyspark.sql.types import DoubleType, IntegerType, StringType

  schema = StructType([
      StructField("user_id", IntegerType()),
      StructField("recommended_items", StringType()),
  ])
  recommends = sqlContext.read.csv(filepath,sep="\t", schema=schema).toPandas()
  # The following line filters lines that do not have 72 item recommendations.
  recommends = recommends[recommends.recommended_items.apply(lambda r:len(r.split(","))) == 72]
  
  if sqlContext is None:
    sc.stop()
  return recommends


def evaluate(filepath, n_jobs=32, spark_file=False, spark_SQLContext=None):
  """
  This is the main function from the module/file. 

  A common way to use is as follows:

  ```python
  from utils.evaluate import evaluate

  filepath = "data/spark_kmeans_100.tsv"
  nDCGs, entropy, globalVar = evaluate(filepath, spark_file=True)

  metrics = pd.DataFrame(nDCGs, columns=["user_id", "nDCGs"])
  print(f"nDCG avg:{metrics.nDCGs.mean():.4}")
  ```

  And you can plot the results with:

  ```python
  avg = metrics.nDCGs.mean()
  _ = plt.figure(figsize=(12,3))
  _ = plt.hist(metrics.nDCGs, bins=50, label=f"Avg:{avg:>21.4f}\nEntropy@72: {entropy:.4f}")
  plt.title(f"{filepath}")
  plt.xlabel("nDCG@72")
  _ = plt.legend()
  ```

  """

  if not spark_file:
    recommends = pd.read_csv(filepath, sep="\t")
  else:
    recommends = spark_filepath_to_pandas(filepath, spark_SQLContext)

  nDCGs = []
  user_truth = defaultdict(list)

  #def make_Y(user_ids, data, file_name):
  GlobalVar.recommends = recommends
  GlobalVar.y_test = pd.read_csv("data/y_test.tsv", sep="\t")
  GlobalVar.value = len(recommends)

  Q = Manager().Queue()
  with Pool(processes=n_jobs) as pool:
    #put listener to work first
    watcher = pool.apply_async(waiter , (Q,))
    # fire off workers
    jobs = []
    for i in range(GlobalVar.value):
      
      job = pool.apply_async(work, (i, Q))
      jobs.append(job)
    for job in jobs: 
      job.get()
        
    Q.put('kill')
    watcher.get()
    q = Q.get()
  
  nDCGs = q[0]

  # Entropy parameters
  GlobalVar.bigI = q[1]
  GlobalVar.ci   = q[2]
  z = 72 * len(recommends.user_id.unique())
  

  entropy_at_72 = - sum( GlobalVar.ci[i]/z * np.log2(GlobalVar.ci[i]/z) for i in GlobalVar.bigI)
  print("Entropy:", entropy_at_72)
  return nDCGs, entropy_at_72, GlobalVar
  