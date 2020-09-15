from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from pyspark.sql import DataFrame as SparkDataFrame

class KMeansTrainer:

  def __init__(self, k=100):
    self.kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol('features')

  @staticmethod
  def make_features(user_master:SparkDataFrame):
    """
    This method receives the user_master table with features 1 to 6 and returns a copy of the
    dataframe with an extra column called `features` which is a column of sparce vectors with the features 1 to 6 one-hot encoded. 
    """
    df = user_master.select([f'feature{i}' for i in range(1,7) ] + ["user_id"] )
    cols = df.columns

    categoricalColumns = [f'feature{i}' for i in range(1,7)]

    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    #label_stringIdx = StringIndexer(inputCol = 'item_id', outputCol = 'label')
    #stages += [label_stringIdx]


    assemblerInputs = [c + "classVec" for c in categoricalColumns] 
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['features'] + cols
    df = df.select(selectedCols)
    #df.printSchema()

    return df

  def silhouete_score(self, data):
    """
    returns the silhouette score of data.
    """
    predictions = self.model.transform(data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print(f"silhouette score: {silhouette:.4f}")
    return silhouette

  def fit(self, train, silhouette_score=False):
    """
    returns the silhouette score of the fitted data if silhouette_score is True. Default False
    """

    self.model = self.kmeans.fit(train)
    if silhouette_score:
      print("Done Fitting", end="\r")
      return self.silhouete_score(train)


    