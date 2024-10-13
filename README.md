# Yahoo!-music-recommender
Building a music recommendation system that uses that alternating least squares method to predict how users are likely to rate songs.

## The Data
The data used in this project is initially from Yahoo! and is available at https://webscope.sandbox.yahoo.com/catalog.php?datatype=r 
The dataset has over 700 million song ratings collected from 1.8 million Yahoo! Music between 2002 and 2006. 
For this project, the dataset was already available on a Hadoop cluster at the University of Portsmouth. The dataset was split into 20 chunks (.txt files). I used two of these files, train_1.txt and test_1.txt as our training and test sets. 
## The Tools Used
I wrote a Python script that would run on an Apache Spark cluster. The functions and modules used are:
- SparkSession
- Split
- RegressionEvaluator
- ALS

## Loading the Data
After initializing the spark session, the primary step is to load our training data. I used the spark.read.text method to read into the text files, specifying the tab character "\t" as the separator. This leads to the creation of a spark DataFrame with one column specified as 'value' in the script.
Here is the code:

training_path=spark.read.text("spark-examples/train_1.txt").withColumn("value",split("value","\t"))

The value column where each row holds a list with three items-  the user id, the song id, and the rating. 

training=training_path.select(training_path.value.getItem(0).cast("int").alias("userId"),

Next, I used the getItem method on each row to create a new DataFrame with three columns - userId, songId, and column.

training=training_path.select(training_path.value.getItem(0).cast("int").alias("userId"),
	training_path.value.getItem(1).cast("int").alias("songId"),training_path.value.getItem(2).cast("float").alias("rating"))
 
 Repeat the same steps to load the contents of the test_1.txt file into the test DataFrame. 

## Building the Model
We use the ALS module from MLLib to create the module while specifying 'drop' as the cold start strategy. It is important to do so because the model would otherwise run into difficulties where there are missing user ratings.

als= ALS(maxIter=10,regParam=0.05,coldStartStrategy="drop", userCol="userId",itemCol="songId",ratingCol="rating")


The next step is to fit the model on the training data and the making predictions on the test data.

model=als.fit(training)

Predictions are saved into a valuable called predictions which are used in the next step to evaluate the model.
predictions=model.transform(test)


