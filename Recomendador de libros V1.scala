
import org.apache.spark.sql.types.{StructType, StructField}
import org.apache.spark.sql.functions.{desc, mean, col}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.types.IntegerType

val books = spark.read.format("csv").option("header", "true").load("./goodbooks-10k/books.csv")
//books = books.withColumnRenamed("id", "books_id")
                        
val book_tags1 = spark.read.format("csv").option("header", "true").load("goodbooks-10k/book_tags.csv")
val book_tags = book_tags1.withColumnRenamed("goodreads_book_id", "book_id")
val ratings1 = spark.read.format("csv").option("header", "true").load("goodbooks-10k/ratings.csv")
val ratings2 = ratings1.withColumn("user_id", $"user_id".cast(IntegerType))
                        .withColumn("book_id", $"book_id".cast(IntegerType))
                        .withColumn("rating", $"rating".cast(IntegerType))
val rawRatings = ratings2.select("user_id", "book_id", "rating")
val tags=spark.read.format("csv").option("header", "true").load("goodbooks-10k/tags.csv")
val to_read = spark.read.format("csv").option("header", "true").load("goodbooks-10k/to_read.csv")

rawRatings.show(1)
book_tags.show(1)
tags.show(1)

def normalizeUserRatings(subject: String, feature: String, df: DataFrame): DataFrame = {
    val featureAvg = s"avg($feature)"
    df.groupBy(subject)
        .agg(mean(feature))
        .join(df, Seq(subject))
        .withColumn(feature, col(feature) - col(featureAvg))
        .drop(featureAvg)
}

val ratings = normalizeUserRatings("user_id", "rating", rawRatings)
ratings.orderBy("user_id", "book_id").show(20)

val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
training.show(1)

val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("user_id")
    .setItemCol("book_id")
    .setRatingCol("rating")
//println(als.explainParams())
val alsModel = als.fit(training)
alsModel.setColdStartStrategy("drop")
val predictions = alsModel.transform(test)
predictions.orderBy("user_id", "book_id").show(20)

val userRecs = alsModel.recommendForAllUsers(30)
    .selectExpr("user_id", "explode(recommendations)")
val itemRecommendations = alsModel.recommendForAllItems(30)
    .selectExpr("book_id", "explode(recommendations)")

userRecs.show(20)
itemRecommendations.show(20)

val userTop = userRecs.withColumn("book_id", $"col.book_id")
    .withColumn("rating", $"col.rating")
    .drop($"col")
userTop.show(20)

val userBookTags = userTop.join(book_tags, Seq("book_id")).drop("count")
userBookTags.show(101)

val topTags = userBookTags.groupBy("user_id", "tag_id")
                        .agg(mean("rating"))
                        .withColumnRenamed("avg(rating)", "tag_rating")
topTags.show(20)

val weightedRecs = userBookTags.join(topTags, Seq("user_id", "tag_id"))
                            .withColumn("rating", $"rating" + $"tag_rating")
                            .drop("tag_rating")
                            .drop("tag_id")
                            .distinct()
weightedRecs.orderBy($"user_id", $"rating".desc).show(20)

val normRecs = normalizeUserRatings("user_id", "rating", weightedRecs)
//                         .groupBy("user_id", "book_id")
//                         .agg(mean("rating"))
//                         .withColumnRenamed("avg(rating)", "rating")
normRecs.orderBy($"user_id", $"rating".desc).show(50)

def makeRecommender(ratings: DataFrame): ALSModel = {
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
    val als = new ALS()
        .setMaxIter(5)
        .setRegParam(0.01)
        .setUserCol("user_id")
        .setItemCol("book_id")
        .setRatingCol("rating")
    val alsModel = als.fit(training)
    
    alsModel.setColdStartStrategy("drop")
}

val recommender = makeRecommender(normRecs)

def getRecommendations(alsModel: ALSModel, numRecs: Int): DataFrame = {
    val userRecs = alsModel.recommendForAllUsers(numRecs)
        .selectExpr("user_id", "explode(recommendations)")
    
    val userTop = userRecs.withColumn("book_id", $"col.book_id")
    .withColumn("rating", $"col.rating")
    .drop($"col")
    
    userTop
}

val recommendations = getRecommendations(recommender, 10)

recommendations.write.format("csv").save("./recommendations_by_tags.csv")

val recs = recommendations.join(books, Seq("book_id"))
                        .select("original_title", "user_id", "book_id", "rating")
                        .orderBy($"user_id", $"rating".desc)
recs.show(40)

import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.sql.functions.{col, expr}

def evaluate(predictions: DataFrame, tag_predictions: DataFrame): Unit = {
    val perUserActual = predictions
        .where("rating > 2.5")
        .groupBy("user_id")
        .agg(expr("collect_set(book_id) as books"))

    val perUserPredictions = predictions
        .orderBy(col("user_id"), col("prediction").desc)
        .groupBy("user_id")
        .agg(expr("collect_list(book_id) as books"))

    val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("user_id"))
        .map(row => (
        row(1).asInstanceOf[Seq[Integer]].toArray,
        row(2).asInstanceOf[Seq[Integer]].toArray.take(15)
        ))
    val ranks = new RankingMetrics(perUserActualvPred.rdd)

    println(s"Mean average Precision = ${ranks.meanAveragePrecision}")
    Seq(1, 3, 5, 8, 10).foreach( k =>
        println(s"Precision at k: $k = ${ranks.precisionAt(k)}")
    )
}

import org.apache.spark.ml.evaluation.RegressionEvaluator
 
def rootMeanSquareError(predictions: DataFrame): Unit = {
    val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
    print("Global average RMSE = 1.1296, netflix grand prize = 0.8563")
}

val tag_predictions = recommender.transform(test)
evaluate(predictions, tag_predictions)

rootMeanSquareError(tag_predictions)
