from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import time

spark=SparkSession.builder.appName("Yahoo_Recommender").config("spark.executor.memory","16g").getOrCreate()


startTime=time.time()


#READING THE TRAINING DATA FILE
training_path=spark.read.text("spark-examples/train_1.txt").withColumn("value",split("value","\t"))

training=training_path.select(training_path.value.getItem(0).cast("int").alias("userId"),
	training_path.value.getItem(1).cast("int").alias("songId"),training_path.value.getItem(2).cast("float").alias("rating"))
#READING THE TEST DATA FILES
test_path=spark.read.text("spark-examples/test_1.txt").withColumn("value",split("value","\t"))
test=test_path.select(test_path.value.getItem(0).cast("int").alias("userId"),
	test_path.value.getItem(1).cast("int").alias("songId"),test_path.value.getItem(2).cast("float").alias("rating"))


#BUILDING THE MODEL

als= ALS(maxIter=10,regParam=0.05,coldStartStrategy="drop", userCol="userId",itemCol="songId",ratingCol="rating")

model=als.fit(training)

#EVALUATING THE MODEL

predictions=model.transform(test)

evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")

rmse=evaluator.evaluate(predictions)

print("Root-mean squared error = "+str(rmse))

#generating top 10 recommendations for a subset of users

users=training.select(als.getUserCol()).distinct().limit(3)

userSubsetRecs=model.recommendForUserSubset(users,10)

userSubsetRecs.show(truncate=False)

endTime=time.time()

print("Elapsed time = " +str(endTime-startTime))

spark.stop()

