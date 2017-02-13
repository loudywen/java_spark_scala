package com.devon.demo.scala.main2

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, RegexTokenizer, Word2Vec, StopWordsRemover}

/**
  * Created by Devon on 2/11/2017.
  */


object app3 {

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Devon_Spark_ML_Demo").setMaster("local[10]")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    val training = sparkSession.createDataFrame(Seq(
      ("you@example.com", "hope you are well", 0.0),
      ("raj@example.com", "nice to hear from you", 0.0),
      ("thomas@example.com", "happy holidays", 0.0),
      ("mark@example.com", "see you tomorrow", 0.0),
      ("xyz@example.com", "save money", 1.0),
      ("top10@example.com", "low interest rate", 1.0),
      ("marketing@example.com", "cheap loan", 1.0),
      ("diwen@example.com", "I am diwen", 0.0),
      ("ff@example.com", "free", 1.0)

    )).toDF("email", "message", "label")

    training.show()
    // Configure an Spark machine learning pipeline, consisting of three stages: tokenizer, hashingTF, and lr.

    val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features")

    // LogisticRegression parameter to make lr.fit() use at most 10 iterations and the regularization parameter.
    // When a higher degree polynomial used by the algorithm to fit a set of points in a linear regression model, to prevent overfitting, regularization is used and this parameter is just for that
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to train the model to study the messages
    val model = pipeline.fit(training)

    // Prepare messages for prediction, which are not categorized and leaving upto the algorithm to predict
    val test = sparkSession.createDataFrame(Seq(
      ("you@example.com", "how are you"),
      ("jain@example.com", "hope doing well"),
      ("caren@example.com", "want some money"),
      ("zhou@example.com", "secure loan"),
      ("ted@example.com", "free account")
    )).toDF("email", "message")

    // Make predictions on the new messages
    val prediction = model.transform(test).select("email", "message", "prediction")
    prediction.show()
  }
}
