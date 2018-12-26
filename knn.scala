import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vectors => mlv}
import scala.math._

object knn {
  System.currentTimeMillis()
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  
  //find nearest k distance function
  def nearest[T](n:Int, iter:Iterable[T])(implicit ord: Ordering[T]): Iterable[T] = {
    def partitionMin(acc: Iterable[T], it: Iterable[T]): Iterable[T]  = {
      val min = it.min(ord)
      val (nextElems, rest) = it.partition(ord.lteq(_, min))
      val minElems = acc ++ nextElems
      if (minElems.size >= n || rest.isEmpty) minElems.take(n)
      else partitionMin(minElems, rest)
    }
    partitionMin(iter.take(0), iter)
  }
  //main
  def main(args: Array[String]): Unit = {
    val path = args(0)
    val k = args(1).toInt
    val sc = SparkSession.builder
      .appName("knn")
      .master("local[*]")
      .getOrCreate()
    
    println("Reading csv file")
 
    val small = sc.read
      .format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(path)
 
  small.take(5)
    println("transforming data")
    val outsmall0 = new VectorAssembler()
      .setInputCols(small.columns.drop(1).dropRight(1))
      .setOutputCol("features")
      .transform(small)
    
     val outsmall = outsmall0.select("class","features").rdd
    
    println("random Split data into test and train")
    val Array(tr, ts) = outsmall.randomSplit(Array(0.70, 0.30))
    
    //tr.saveAsTextFile("src/main/resources/train")
    //ts.saveAsTextFile("src/main/resources/test")
  
    println("mapping processing knn")
    val start = System.nanoTime()
    val scomb = ts.cartesian(tr).persist()
    val s = scomb.map{
      case(r ,t)=> (r.getAs(1).toString,math.sqrt(mlv.sqdist(r.getAs(1),t.getAs(1))),t.getAs(0).toString)
    }.persist()
    val sss = s.groupBy(_._1).map(x => (x._1,nearest(k,x._2.map(x=>(x._2,x._3))),nearest(k,x._2.map(x=>(x._2,x._3))).map(x =>x._2).groupBy(identity).maxBy(_._2.size)._1))
    
    //sss.saveAsTextFile("src/main/resources/results")
    val elapsedTime = (System.nanoTime() - start)/1000000000.000000
    println("elapsed " + elapsedTime +" seconds")
    
    sc.stop()
  }
}
