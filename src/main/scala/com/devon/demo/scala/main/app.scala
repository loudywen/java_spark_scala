package com.devon.demo.scala.main

import java.util.concurrent.atomic.AtomicInteger

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import com.datastax.spark.connector._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.cassandra._
import org.apache.spark.sql._;

/*
  * Created by diwenlao on 2/8/17.
  */


case class RealTime911(callType: String, datetime: String, address: String, latitude: String, longitude: String, report_location: String, incident_number: String)

object app {
  //  val filePath2 = "/Users/diwenlao/IdeaProjects/java_spark_scala/data.txt"
  //  val filePath = "/Users/diwenlao/IdeaProjects/java_spark_scala/Seattle_Real_Time_Fire_911_Calls_Chrono2.csv"
  val filePath = "C:\\Users\\Devon\\IdeaProjects\\java_spark_scala\\Seattle_Real_Time_Fire_911_Calls_Chrono2.csv"

  val counter = new AtomicInteger()


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("Devon_Spark_Demo").setMaster("local[10]").set("spark.cassandra.connection.host", "172.16.143.138")

    // Home cassandra ip
    // val conf = new SparkConf().setAppName("Devon_Spark_Demo").setMaster("local[10]").set("spark.cassandra.connection.host", "192.168.0.28")

    // Entry point of Spark SQL
    val sparkDF = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()


    // Read CSV to DataFrame
    val df = sparkDF.read.format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load(filePath) //.csv("csv/file/path") //spark 2.0 api

    df.printSchema()

    //
    //    val address1 = sparkDF.sql("SELECT address FROM global_temp WHERE Type LIKE'%Fire%'").rdd
    //    df.printSchema()

//df.show(5000)
/*
    df.foreach(row => {
      if (row.isNullAt(0)) {

       // row.getString(0) == "No Address"
        println("=====" + row.getString(0))
      }

    })
*/


    // Create a global table for later use
    df.createOrReplaceTempView("global_temp")
    val tempView = sparkDF.sql("SELECT * FROM global_temp WHERE address IS NULL")
    tempView.show()


    // This implicits importing is really important!
    import sparkDF.implicits._
    /*  df.map(r => {
          if (r.getString(0) == null)
            r.getString(0) == "No Address"
          RealTime911(r.getString(1), r.getString(2), r.getString(0), r.getString(3), r.getString(4), r.getString(5), r.getString(6))
        })*/

    /*  val temp = df.rdd.foreach(r => {
        if (r.getString(0) == null)
          println("======" + r.getString(0) + " count: " + counter.getAndIncrement())
         r.getString(0)=="No Address"
      })*/

    //      .map(r => RealTime911(r.getString(0), r.getString(1), r.getString(2), r.getString(3), r.getString(4), r.getString(5), r.getString(6))).rdd

    //    temp.saveToCassandra("devon_test", "test1")


  }

}
