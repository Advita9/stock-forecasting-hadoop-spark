from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("StockAnalysis") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs://localhost:9000/stock-data/aapl.csv",
    header=True,
    inferSchema=True
)

df.show(5)
