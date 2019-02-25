#!/usr/bin/env python
# coding: utf-8

# ## Getting to know PySpark
# 
# ### What is Spark, anyway?
# Spark is a platform for cluster computing. Spark lets you spread data and computations over clusters with multiple nodes (think of each node as a separate computer). Splitting up your data makes it easier to work with very large datasets because each node only works with a small amount of data.
# 
# As each node works on its own subset of the total data, it also carries out a part of the total calculations required, so that both data processing and computation are performed in parallel over the nodes in the cluster. It is a fact that parallel computation can make certain types of programming tasks much faster.
# 
# However, with greater computing power comes greater complexity.
# 
# Deciding whether or not Spark is the best solution for your problem takes some experience, but you can consider questions like:
# 
# Is my data too big to work with on a single machine?
# Can my calculations be easily parallelized?
# 
# ### Using Spark in Python
# The first step in using Spark is connecting to a cluster.
# 
# In practice, the cluster will be hosted on a remote machine that's connected to all other nodes. There will be one computer, called the master that manages splitting up the data and the computations. The master is connected to the rest of the computers in the cluster, which are called slaves. The master sends the slaves data and calculations to run, and they send their results back to the master.
# 
# When you're just getting started with Spark it's simpler to just run a cluster locally. Thus, for this course, instead of connecting to another computer, all computations will be run on DataCamp's servers in a simulated cluster.
# 
# Creating the connection is as simple as creating an instance of the SparkContext class. The class constructor takes a few optional arguments that allow you to specify the attributes of the cluster you're connecting to.
# 
# An object holding all these attributes can be created with the SparkConf() constructor. Take a look at the documentation for all the details!
# 
# For the rest of this course you'll have a SparkContext called sc already available in your workspace.

# In[62]:


import pyspark as sp


# In[63]:


sc = sp.SparkContext.getOrCreate()
print(sc)
print(sc.version)


# ### Using DataFrames
# Spark's core data structure is the Resilient Distributed Dataset (RDD). This is a low level object that lets Spark work its magic by splitting data across multiple nodes in the cluster. However, RDDs are hard to work with directly, so in this course you'll be using the Spark DataFrame abstraction built on top of RDDs.
# 
# The Spark DataFrame was designed to behave a lot like a SQL table (a table with variables in the columns and observations in the rows). Not only are they easier to understand, DataFrames are also more optimized for complicated operations than RDDs.
# 
# When you start modifying and combining columns and rows of data, there are many ways to arrive at the same result, but some often take much longer than others. When using RDDs, it's up to the data scientist to figure out the right way to optimize the query, but the DataFrame implementation has much of this optimization built in!
# 
# To start working with Spark DataFrames, you first have to create a SparkSession object from your SparkContext. You can think of the SparkContext as your connection to the cluster and the SparkSession as your interface with that connection.
# 
# Remember, for the rest of this course you'll have a SparkSession called spark available in your workspace!
# 
# ### Creating a SparkSession
# We've already created a SparkSession for you called spark, but what if you're not sure there already is one? Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the SparkSession.builder.getOrCreate() method. This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary!

# In[64]:


#import SparkSeccion pyspark.sql
from pyspark.sql import SparkSession

#Create my_spark
my_spark = SparkSession.builder.getOrCreate()

#print my_spark
print(my_spark)


# #### Viewing tables
# Once you've created a SparkSession, you can start poking around to see what data is in your cluster!
# 
# Your SparkSession has an attribute called catalog which lists all the data inside the cluster. This attribute has a few methods for extracting different pieces of information.
# 
# One of the most useful is the .listTables() method, which returns the names of all the tables in your cluster as a list.
# ```python
# print(spark.catalog.listTables())
# ```
# ### Are you query-ious?
# One of the advantages of the DataFrame interface is that you can run SQL queries on the tables in your Spark cluster. If you don't have any experience with SQL, don't worry (you can take our Introduction to SQL course!), we'll provide you with queries!
# 
# As you saw in the last exercise, one of the tables in your cluster is the flights table. This table contains a row for every flight that left Portland International Airport (PDX) or Seattle-Tacoma International Airport (SEA) in 2014 and 2015.
# 
# Running a query on this table is as easy as using the .sql() method on your SparkSession. This method takes a string containing the query and returns a DataFrame with the results!
# 
# If you look closely, you'll notice that the table flights is only mentioned in the query, not as an argument to any of the methods. This is because there isn't a local object in your environment that holds that data, so it wouldn't make sense to pass the table as an argument.
# 
# Remember, we've already created a SparkSession called spark in your workspace.
# ```python
# query = "FROM flights SELECT * LIMIT 10"
# 
# # Get the first 10 rows of flights
# flights10 =sql(query)
# 
# # Show the results
# flights10.show()
# ```
# ![Screen%20Shot%202019-01-15%20at%205.11.27%20PM.png](attachment:Screen%20Shot%202019-01-15%20at%205.11.27%20PM.png)

# 
# ### Pandafy a Spark DataFrame
# Suppose you've run a query on your huge dataset and aggregated it down to something a little more manageable.
# 
# Sometimes it makes sense to then take that table and work with it locally using a tool like pandas. Spark DataFrames make that easy with the .toPandas() method. Calling this method on a Spark DataFrame returns the corresponding pandas DataFrame. It's as simple as that!
# 
# This time the query counts the number of flights to each airport from SEA and PDX.
# 
# Remember, there's already a SparkSession called spark in your workspace!
# 
# ```python
# query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
# 
# # Run the query
# flight_counts = spark.sql(query)
# 
# # Convert the results to a pandas DataFrame
# pd_counts = flight_counts.toPandas()
# 
# # Print the head of pd_counts
# print(pd_counts.head())
# 
# ```
# ### Put some Spark in your data
# In the last exercise, you saw how to move data from Spark to pandas. However, maybe you want to go the other direction, and put a pandas DataFrame into a Spark cluster! The SparkSession class has a method for this as well.
# 
# The .createDataFrame() method takes a pandas DataFrame and returns a Spark DataFrame.
# 
# The output of this method is stored locally, not in the SparkSession catalog. This means that you can use all the Spark DataFrame methods on it, but you can't access the data in other contexts.
# 
# For example, a SQL query (using the .sql() method) that references your DataFrame will throw an error. To access the data in this way, you have to save it as a temporary table.
# 
# You can do this using the .createTempView() Spark DataFrame method, which takes as its only argument the name of the temporary table you'd like to register. This method registers the DataFrame as a table in the catalog, but as this table is temporary, it can only be accessed from the specific SparkSession used to create the Spark DataFrame.
# 
# There is also the method .createOrReplaceTempView(). This safely creates a new temporary table if nothing was there before, or updates an existing table if one was already defined. You'll use this method to avoid running into problems with duplicate tables.
# 
# Check out the diagram to see all the different ways your Spark data structures interact with each other.
# 
# ![spark_figure.png](attachment:spark_figure.png)
# 

# In[65]:


import pandas as pd
import numpy as np
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())


# ### Dropping the middle man
# Now you know how to put data into Spark via pandas, but you're probably wondering why deal with pandas at all? Wouldn't it be easier to just read a text file straight into Spark? Of course it would!
# 
# Luckily, your SparkSession has a .read attribute which has several methods for reading different data sources into Spark DataFrames. Using these you can create a DataFrame from a .csv file just like with regular pandas DataFrames!
# 
# The variable file_path is a string with the path to the file airports.csv. This file contains information about different airports all over the world.
# 
# A SparkSession named spark is available in your workspace.
# 

# In[66]:


file_path = 'data/airports.csv'

#Read in the airports path
airports = spark.read.csv(file_path, header=True)

airports.show()


# In[67]:


type(airports)


# In[68]:


my_spark.catalog.listDatabases()


# In[69]:


my_spark.catalog.listTables()


# ## Manipulating data
# ### Creating columns
# In this chapter, you'll learn how to use the methods defined by Spark's DataFrame class to perform common data operations.
# 
# Let's look at performing column-wise operations. In Spark you can do this using the .withColumn() method, which takes two arguments. First, a string with the name of your new column, and second the new column itself.
# 
# The new column must be an object of class Column. Creating one of these is as easy as extracting a column from your DataFrame using df.colName.
# 
# Updating a Spark DataFrame is somewhat different than working in pandas because the Spark DataFrame is immutable. This means that it can't be changed, and so columns can't be updated in place.
# 
# Thus, all these methods return a new DataFrame. To overwrite the original DataFrame you must reassign the returned DataFrame using the method like so:
# ```python
# df = df.withColumn("newCol", df.oldCol + 1)
# ```
# The above code creates a DataFrame with the same columns as df plus a new column, newCol, where every entry is equal to the corresponding entry from oldCol, plus one.
# 
# To overwrite an existing column, just pass the name of the column as the first argument!
# 

# In[70]:


flights = spark.read.csv('data/flights_small.csv', header=True)
flights.show()


# In[71]:


flights.name = flights.createOrReplaceTempView('flights')
my_spark.catalog.listTables()


# In[72]:


# Create the DataFrame flights
flights_df = my_spark.table('flights')
print(flights_df.show())


# In[73]:


#include a new column called duration_hrs
flights = flights.withColumn('duration_hrs', flights.air_time / 60)
flights.show()


# ### Filtering Data
# Let's take a look at the .filter() method. As you might suspect, this is the Spark counterpart of SQL's WHERE clause. The .filter() method takes either a Spark Column of boolean (True/False) values or the WHERE clause of a SQL expression as a string.
# 
# For example, the following two expressions will produce the same output:
# ```python
# flights.filter(flights.air_time > 120).show()
# flights.filter("air_time > 120").show()
# ```

# In[74]:


# Filter flights with a SQL string
long_flights1 = flights.filter('distance > 1000')
# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000 )

long_flights1.show()
long_flights2.show()


# ### Selecting
# The Spark variant of SQL's SELECT is the .select() method. This method takes multiple arguments - one for each column you want to select. These arguments can either be the column name as a string (one for each column) or a column object (using the df.colName syntax). When you pass a column object, you can perform operations like addition or subtraction on the column to change the data contained in it, much like inside .withColumn().
# 
# The difference between .select() and .withColumn() methods is that .select() returns only the columns you specify, while .withColumn() returns all the columns of the DataFrame in addition to the one you defined. It's often a good idea to drop columns you don't need at the beginning of an operation so that you're not dragging around extra data as you're wrangling. In this case, you would use .select() and not .withColumn().

# In[75]:


# Select the first set of columns as a string
selected_1 = flights.select('tailnum', 'origin', 'dest')

# Select the second set of columns usinf df.col_name
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter to only keep flights from SEA to PDX.
FilterA = flights.origin == 'SEA'
FilterB =flights.dest == 'PDX'

# Filter the data, first by filterA then by filterB
selected_2 = temp.filter(FilterA).filter(FilterB)

selected_2.show()


# Similar to SQL, you can also use the .select() method to perform column-wise operations. When you're selecting a column using the df.colName notation, you can perform any column operation and the .select() method will return the transformed column. For example,
# 
# ```python
# flights.select(flights.air_time/60)
# ```
# returns a column of flight durations in hours instead of minutes. You can also use the .alias() method to rename a column you're selecting. So if you wanted to .select() the column duration_hrs (which isn't in your DataFrame) you could do
# 
# ```python
# flights.select((flights.air_time/60).alias("duration_hrs"))
# ```
# The equivalent Spark DataFrame method .selectExpr() takes SQL expressions as a string:
# 
# flights.selectExpr("air_time/60 as duration_hrs")
# with the SQL as keyword being equivalent to the .alias() method. To select multiple columns, you can pass multiple strings.

# In[76]:


#Create a table of the average speed of each flight both ways.
#Calculate average speed by dividing the distance by the air_time (converted to hours).Use the .alias() method name
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
speed_1 = flights.select('origin','dest','tailnum', avg_speed)

#Using the Spark DataFrame method .selectExpr() 
speed_2 =flights.selectExpr('origin','dest','tailnum','distance/(air_time/60) as avg_speed')
speed_2.show()


# ### Aggregating
# All of the common aggregation methods, like `.min()`, `.max()`, and `.count()` are GroupedData methods. These are created by calling the `.groupBy()` DataFrame method. For now, all you have to do to use these functions is call that method on your DataFrame. For example, to find the minimum value of a column, col, in a DataFrame, df, you could do
# ```python
# df.groupBy().min("col").show()
# ```
# This creates a GroupedData object (so you can use the .min() method), then finds the minimum value in col, and returns it as a DataFrame.

# In[77]:


flights.describe()


# In[78]:


#arr_time: string and distance: string, so to find min() and max() we need to convert this float 
flights = flights.withColumn('distance', flights.distance.cast('float'))
flights = flights.withColumn('air_time', flights.air_time.cast('float'))

flights.describe('air_time', 'distance').show()


# In[79]:


#Find the length of the shortest (in terms of distance) flight that left PDX 
flights.filter(flights.origin =='PDX').groupBy().min('distance').show()

#Find the length of the longest (in terms of time) flight that left SEA
flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()


# In[80]:


#get the average air time of Delta Airlines flights  that left SEA. 
flights.filter(flights.carrier == 'DL').filter(flights.origin == 'SEA').groupBy().avg('air_time').show()


# In[81]:


#get the total number of hours all planes in this dataset spent in the air by creating a column called duration_hrs
flights.withColumn('duration_hrs', flights.air_time/60).groupBy().sum('duration_hrs').show()


# ### Grouping and Aggregating I
# Part of what makes aggregating so powerful is the addition of groups. PySpark has a whole class devoted to grouped data frames: `pyspark.sql.GroupedData`, which you saw in the previous exercises.
# 
# You've learned how to create a grouped DataFrame by calling the .groupBy() method on a DataFrame with no arguments.Now you'll see that when you pass the name of one or more columns in your DataFrame to the .groupBy() method, the aggregation methods behave like when you use a GROUP BY statement in a SQL query!

# In[82]:


#Group bu tailnum colum
by_plane = flights.groupBy('tailnum')

#Use the .count() method with no arguments to count the number of flights each plane made
by_plane.count().show()


# In[83]:


#group by origin column
by_origin = flights.groupBy('origin')

#Find the .avg() of the air_time column to find average duration of flights from PDX and SEA
by_origin.avg('air_time').show()


# ### Grouping and Aggregating II
# In addition to the GroupedData methods you've already seen, there is also the `.agg()` method. This method lets you pass an aggregate column expression that uses any of the aggregate functions from the pyspark.sql.functions submodule.
# 
# This submodule contains many useful functions for computing things like standard deviations. All the aggregation functions in this submodule take the name of a column in a GroupedData table.

# In[86]:


# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy('month', 'dest')

#convert to dep_delay to numeric column
flights = flights.withColumn('dep_delay', flights.dep_delay.cast('float'))


# In[87]:


# Average departure delay by month and destination
by_month_dest.avg('dep_delay').show()


# ### Joining II
# In PySpark, joins are performed using the DataFrame method `.join()`. This method takes `three arguments`. The `first` is the `second DataFrame` that you want to join with the first one. The `second argument`, on, is the `name of the key column(s)` as a string. The names of the key column(s) must be the same in each table. The `third argument`, `how,` specifies the kind of join to perform. In this course we'll always use the value `how="leftouter`".
# 
# The flights dataset and a new dataset called airports are already in your workspace.

# In[88]:


airports.show()


# In[89]:


# Rename the faa column
airports = airports.withColumnRenamed('faa','dest')


# In[90]:


# Join the DataFrames
flights_with_airports= flights.join(airports, on='dest', how='leftouter')
flights_with_airports.show()


# ## Getting started with machine learning pipelines
# 
# ### Machine Learning Pipelines
# 
# At the core of the `pyspark.ml` module are the Transformer and Estimator classes. Almost every other class in the module behaves similarly to these two basic classes. Transformer classes have a `.transform()` method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended. For example, you might use the class Bucketizer to create discrete bins from a continuous feature or the class PCA to reduce the dimensionality of your dataset using principal component analysis. Estimator classes all implement a `.fit() ` method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a StringIndexerModel for including categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm for classification or regression.
# 

# In[91]:


planes = my_spark.read.csv('data/planes.csv', header=True)
planes.show()


# In[92]:


# Rename year column on panes to avoid duplicate column name
planes = planes.withColumnRenamed('year', 'plane_year')


# In[93]:


#join the flights and plane table use key as tailnum column
model_data = flights.join(planes, on='tailnum', how='leftouter')
model_data.show()


# ### Data types
# Before you get started modeling, it's important to know that Spark only handles `numeric data`. That means all of the columns in your DataFrame must be either integers or decimals (called 'doubles' in Spark).
# 
# When we imported our data, we let Spark guess what kind of information each column held. Unfortunately, Spark doesn't always guess right and you can see that some of the columns in our DataFrame are strings containing numbers as opposed to actual numeric values.
# 
# To remedy this, you can use the .cast() method in combination with the .withColumn() method. It's important to note that .cast() works on columns, while .withColumn() works on DataFrames.
# 
# The only argument you need to pass to .cast() is the kind of value you want to create, in string form. For example, to create integers, you'll pass the argument "integer" and for decimal numbers you'll use "double".

# In[94]:


model_data.describe()


# In[95]:


model_data = model_data.withColumn('arr_delay', model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn('air_time' , model_data.air_time.cast('integer'))
model_data = model_data.withColumn('month', model_data.month.cast('integer'))
model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))


# In[96]:


model_data.describe('arr_delay', 'air_time','month', 'plane_year').show()


# ### Create a new column
# Create the column `plane_age` using the `.withColumn()` method and subtracting the year of manufacture (column `plane_year`) from the `year` (column year) of the flight.

# In[97]:


model_data =model_data.withColumn('plane_age', model_data.year - model_data.plane_year)


# ### Making a Boolean
# 
# Use the `.withColumn()` method to create the column` is_late`. This column is equal to `model_data.arr_delay > 0`.
# Convert this column to an `integer column` so that you can use it in your model and name it `label` (this is the default name for the response variable in Spark's machine learning routines).
# `Filter out missing values `

# In[98]:


model_data = model_data.withColumn('is_late', model_data.arr_delay >0)

model_data = model_data.withColumn('label', model_data.is_late.cast('integer'))

model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")


# ### Strings and factors
# As you know, Spark requires numeric data for modeling. So far this hasn't been an issue; even boolean columns can easily be converted to integers without any trouble. But you'll also be using the airline and the plane's destination as features in your model. These are coded as strings and there isn't any obvious way to convert them to a numeric data type.
# 
# Fortunately, PySpark has functions for handling this built into the `pyspark.ml.features` submodule. You can create what are called `'one-hot vectors'` to represent the carrier and the destination of each flight. A one-hot vector is a way of representing a categorical feature where every observation has a vector in which all elements are zero except for at most one element, which has a value of one (1).
# 
# Each element in the vector corresponds to a level of the feature, so it's possible to tell what the right level is by seeing which element of the vector is equal to one (1).
# 
# The first step to encoding your categorical feature is to create a `StringIndexer`. Members of this class are Estimators that take a DataFrame with a column of strings and map each unique string to a number. Then, the Estimator returns a Transformer that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric column corresponding to the string column.
# 
# The second step is to encode this numeric column as a one-hot vector using a `OneHotEncoder`. This works exactly the same way as the StringIndexer by creating an Estimator and then a Transformer. The end result is a column that encodes your categorical feature as a vector that's suitable for machine learning routines!
# 

# #### Carrier
# In this exercise you'll create a` StringIndexer` and a `OneHotEncoder` to code the carrier column. To do this, you'll call the class constructors with the arguments inputCol and outputCol. The `inputCol` is the name of the column you want to index or encode, and the `outputCol` is the name of the new column that the Transformer should create.

# In[99]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder


# In[100]:


#Create a StringIndexer
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
#Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')


# #### Destination

# In[101]:


# encode the dest column just like you did above
dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')


# #### Assemble a  Vector
# The last step in the Pipeline is to combine all of the columns containing our features into a single column. pyspark.ml.feature submodule contains a class called VectorAssembler. This Transformer takes all of the columns you specify and combines them into a new vector column.

# In[102]:


from pyspark.ml.feature import  VectorAssembler


# In[103]:


vec_assembler =VectorAssembler(inputCols=['month', 'air_time','carr_fact','dest_fact','plane_age'],
                              outputCol='features')


# #### Create the pipeline
# You're finally ready to create a` Pipeline!` Pipeline is a class in the `pyspark.ml module` that combines all the Estimators and Transformers that you've already created.

# In[104]:


from pyspark.ml import Pipeline


# In[105]:


flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])


# ### Test vs Train
# After you've cleaned your data and gotten it ready for modeling, one of the most important steps is to split the data into a `test set` and a` train set`. After that, `don't touch your test data` until you think you have a good model! As you're building models and forming hypotheses, you can test them on your training data to get an idea of their performance.
# 
# Once you've got your favorite model, you can see how well it predicts the new data in your test set. This never-before-seen data will give you a much more realistic idea of your model's performance in the real world when you're trying to predict or classify new data.
# 
# In Spark it's important to make sure you split the data after all the transformations. This is because operations like StringIndexer don't always produce the same index even when given the same list of strings.

# ### Transform the data

# In[106]:


piped_data =flights_pipe.fit(model_data).transform(model_data)


# In[107]:


piped_data.show()


# In[123]:


training, test = piped_data.randomSplit([.6, .4])


# ## Model tuning and selection
# 
# ### What is logistic regression?
# The model you'll be fitting in this chapter is called a logistic regression. This model is very similar to a linear regression, but instead of predicting a numeric variable, it predicts the probability (between 0 and 1) of an event.
# 
# To use this as a classification algorithm, all you have to do is assign a cutoff point to these probabilities. If the predicted probability is above the cutoff point, you classify that observation as a 'yes' (in this case, the flight being late), if it's below, you classify it as a 'no'!
# 
# You'll tune this model by testing different values for several hyperparameters. A hyperparameter is just a value in the model that's not estimated from the data, but rather is supplied by the user to maximize performance
# 
# 

# In[124]:


from pyspark.ml.classification import LogisticRegression


# In[125]:


lr = LogisticRegression()


# ### Cross validation
# In the next few exercises you'll be tuning your logistic regression model using a procedure called `k-fold cross validation`. This is a method of estimating the model's performance on unseen data (like your test DataFrame).
# 
# It works by splitting the training data into a few different partitions. The exact number is up to you, but in this course you'll be using PySpark's default value of three. Once the data is split up, one of the partitions is set aside, and the model is fit to the others. Then the error is measured against the held out partition. This is repeated for each of the partitions, so that every block of data is held out and used as a test set exactly once. Then the error on each of the partitions is averaged. This is called the cross validation error of the model, and is a good estimate of the actual error on the held out data.
# 
# You'll be using cross validation to choose the hyperparameters by creating a grid of the possible pairs of values for the two hyperparameters, elasticNetParam and regParam, and using the cross validation error to compare all the different models so you can choose the best one!

# #### Create the evaluator
# The first thing you need when doing cross validation for model selection is a way to compare different models. Luckily, the pyspark.ml.evaluation submodule has classes for evaluating different kinds of models. Your model is a binary classification model, so you'll be using the `BinaryClassificationEvaluator` from the `pyspark.ml.evaluation` module. This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number.

# In[126]:


import pyspark.ml.evaluation as evals


# In[127]:


evalator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')


# #### Make a grid
# Next, you need to create a grid of values to search over when looking for the optimal hyperparameters. The submodule `pyspark.ml.tuning` includes a class called `ParamGridBuilder` that does just that (maybe you're starting to notice a pattern here; PySpark has a submodule for just about everything!).
# 
# You'll need to use the `.addGrid()` and `.build()` methods to create a grid that you can use for cross validation. The .addGrid() method takes a model parameter (an attribute of the model Estimator, lr, that you created a few exercises ago) and a list of values that you want to try. The .build() method takes no arguments, it just returns the grid that you'll use later.

# In[128]:


import pyspark.ml.tuning as tune


# In[129]:


# Create the parameter grid
grid = tune.ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0., .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0., 1.])
# Build the grid
grid = grid.build()


# #### Make the validator
# The submodule pyspark.ml.tuning also has a class called CrossValidator for performing cross validation. This Estimator takes the modeler you want to fit, the grid of hyperparameters you created, and the evaluator you want to use to compare your models.
# 
# The submodule `pyspark.ml.tune` has already been imported as `tune`. You'll create the CrossValidator by passing it the logistic regression Estimator `lr`, the parameter `grid`, and the evaluator you created in the previous exercises.

# In[130]:


cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evalator)


# #### Fit the model(s)
# You're finally ready to fit the models and select the best one!

# In[133]:


#models = cv.fit(training)
#when I fit the data I saw a problem on it , Figure it out the problem then continue the last two step.


# ```python
# # Fit cross validation models
# models = cv.fit(training)
# 
# # Extract the best model
# best_lr = models.bestModel
# 
# # Use the model to predict the test set
# test_results = best_lr.transform(test)
# 
# # Evaluate the predictions
# print(evaluator.evaluate(test_results))
# 
# ```

# In[ ]:




