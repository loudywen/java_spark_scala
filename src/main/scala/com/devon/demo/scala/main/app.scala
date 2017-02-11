package com.devon.demo.scala.main

import java.util.concurrent.atomic.AtomicInteger

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import com.datastax.spark.connector._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.cassandra._
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession};

/*
  * Created by diwenlao on 2/8/17.
  */


case class RealTime911(val address: String, callType: String, datetime: String, latitude: String, longitude: String, report_location: String, incident_number: String)

object app {

  val filePath = "/Users/diwenlao/IdeaProjects/java_spark_scala/Seattle_Real_Time_Fire_911_Calls_Chrono2.csv"
  //  val filePath2 = "/Users/diwenlao/IdeaProjects/java_spark_scala/data.txt"

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.WARN);

    val conf = new SparkConf().setAppName("Devon_Spark_Demo").setMaster("local[10]").set("spark.cassandra.connection.host", "172.16.143.138")

    /*val sc = new SparkContext(conf)

    val collection = sc.parallelize(Seq(("cat", 31), ("fox", 40)))
    println(collection)
    collection.saveToCassandra("devon_test", "words", SomeColumns("word", "count"))
*/

    // Entry point of Spark SQL
    val sparkDF = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()


    // Read CSV to DataFrame
    val df = sparkDF.read.format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
      .option("mode", "PERMISSIVE")
      .load(filePath) //.csv("csv/file/path") //spark 2.0 api

    //    df.printSchema()
    // Create a global table for later use
    //    df.createOrReplaceTempView("global_temp")
    //
    //    val address1 = sparkDF.sql("SELECT address FROM global_temp WHERE Type LIKE'%Fire%'").rdd
    //    df.printSchema()

    // This implicits importing is really important!
    import sparkDF.implicits._


    /*   val temp:RDD[RealTime911]= df.map(r => {

         RealTime911(r.getString(0), r.getString(1), r.getString(2), r.getString(3), r.getString(4), r.getString(5), r.getString(6))

       }).rdd*/

    val counter = new AtomicInteger()
    val temp = df.rdd.foreach(r =>{
      if (r.getString(0) == null) println("======" + r.getString(0) + " count: "+ counter.getAndIncrement())
      // r.getString(0) != null

    })

    //      .map(r => RealTime911(r.getString(0), r.getString(1), r.getString(2), r.getString(3), r.getString(4), r.getString(5), r.getString(6))).rdd

    //    temp.saveToCassandra("devon_test", "test1")

    //temp.rdd.saveToCassandra("devon_test", "test1", SomeColumns("address", "callType", "datetime", "latitude", "longitude", "report_location", "incident_number"))
    //
    //    val dataset2: Dataset[address] = dataset.map( r => new address(r.getString(0)))
    //
    //
    //    println(dataset2)


    //    df.select("address").show()

    //    df.select("address").write
    //      .format("org.apache.spark.sql.cassandra")
    //      .options(Map( "table" -> "test", "keyspace" -> "devon_test" ))
    //      .save()

    //    val collect = df.select("address").rdd

    //    println(dataset)

    //    dataset.write
    //      .format("org.apache.spark.sql.cassandra")
    //      .options(Map("table" -> "test", "keyspace" -> "devon_test")).mode(SaveMode.Append)
    //      .save()

    //    collect.foreach(row => println(row))
    //    collect.saveToCassandra("devon_test","test",SomeColumns("address"))

    //    collect.saveAsTextFile("/Users/diwenlao/IdeaProjects/java_spark_scala/test.txt")


    //typeView.show()
    //    val rows: RDD[Row] = typeView.rdd
    //
    //
    //   val strs:RDD[String] = rows.map(rdd => rdd.toString())


    //
    //    strs.foreach( str => println(str))
    //    strs.saveToCassandra("devon_test", "test")
    //        typeView.write.format("org.apache.spark.sql.cassandra").options(Map( "table" -> "test", "keyspace" -> "devon_test")).save()


  }

}
