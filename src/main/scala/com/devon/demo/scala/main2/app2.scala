package com.devon.demo.scala.main2

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.regression.LinearRegressionModel

/**
  * Created by Devon on 2/11/2017.
  */

case class Wine(FixedAcidity: Double, VolatileAcidity: Double, CitricAcid: Double, ResidualSugar: Double,
                Chlorides: Double, FreeSulfurDioxide: Double, TotalSulfurDioxide: Double, Density: Double,
                PH: Double, Sulphates: Double, Alcohol: Double, Quality: Double)

object app2 {

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  val filePath2 = "C:\\Users\\Devon\\IdeaProjects\\java_spark_scala\\winequality-red.csv"

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Devon_Spark_ML_Demo").setMaster("local[10]")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    // Create the the RDD by reading the wine data from the disk
    val wineDataRDD = sc.textFile(filePath2).map(_.split(";")).map(w => Wine(w(0).toDouble, w(1).toDouble, w(2).toDouble, w(3).toDouble, w(4).toDouble,
      w(5).toDouble, w(6).toDouble, w(7).toDouble, w(8).toDouble, w(9).toDouble, w(10).toDouble, w(11).toDouble))


    // Create the data frame containing the training data having two columns. 1) The actual output or label of the data 2) The vector containing the features
    //Vector is a data type with 0 based indices and double-typed values. In that there are two types namely dense and sparse.
    //A dense vector is backed by a double array representing its entry values
    //A sparse vector is backed by two parallel arrays: indices and values
    import sparkSession.implicits._
    val trainingDF = wineDataRDD.map(w => (w.Quality, Vectors.dense(w.FixedAcidity, w.VolatileAcidity, w.CitricAcid,
      w.ResidualSugar, w.Chlorides, w.FreeSulfurDioxide, w.TotalSulfurDioxide, w.Density, w.PH, w.Sulphates, w.Alcohol))).toDF("label", "features")

    trainingDF.show()

    // Create the object of the algorithm which is the Linear Regression
    val lr = new LinearRegression()

    // Linear regression parameter to make lr.fit() use at most 10 iterations
    lr.setMaxIter(10)

    // Create a trained model by fitting the parameters using the training data
    val model = lr.fit(trainingDF)
    //    model.save("wineLRModelPath")
    // Once the model is prepared, to test the model, prepare the test data containing the labels and feature vectors
    val testDF = sparkSession.createDataFrame(Seq(
      (5.0, Vectors.dense(7.4, 0.7, 0.0, 1.9, 0.076, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8)),
      (5.0, Vectors.dense(7.8, 0.88, 0.0, 2.6, 0.098, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4)),
      (7.0, Vectors.dense(7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 18.0, 0.9968, 3.36, 0.57, 9.5)))).toDF("label", "features")

    testDF.show()

    // Do the transformation of the test data using the model and predict the output values or labels.
    // This is to compare the predicted value and the actual label value
    testDF.createOrReplaceTempView("test")

    val tested = model.transform(testDF).select("features", "label", "prediction")

    tested.show()

    // Prepare a dataset without the output/lables to predict the output using the trained model
    val predictDF = sparkSession.sql("SELECT features FROM test")
    predictDF.show()

    // Do the transformation with the predict dataset and display the predictions
    val predicted = model.transform(predictDF).select("features", "prediction")
    predicted.show()

    println("=========================================================")

    /* val newModel = LinearRegressionModel.load("wineLRModelPath")

     val newPredicted = newModel.transform(predictDF).select("features", "prediction")
     newPredicted.show()*/

    /*The preceding code does a lot of things. It performs the following chain of activities in a pipeline:

     1. It reads the wine data from the data file to form a training DataFrame.
     2. Then it creates a LinearRegression object and sets the parameters.
     3. It fits the model with the training data and this completes the estimator pipeline.
     4. It creates a DataFrame containing test data. Typically, the test data will have both features and labels.
        This is to make sure that the model is right and used for comparing the predicted label and actual label.
     5. Using the model created, it does a transformation with the test data, and from the DataFrame produced, extracts the features, input labels,
        and predictions. Note that while doing the transformation using the model, the labels are not required. In other words, the labels will not be used at all.
     6. Using the model created, it does a transformation with the prediction data and from the DataFrame produced, extracts the features and predictions.
        Note that while doing the transformation using the model, the labels are not used. In other words, the labels are not used while doing the predictions.
         This completes a transformer pipeline.*/
  }
}
