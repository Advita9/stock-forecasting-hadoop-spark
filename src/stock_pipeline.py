from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lead, abs, mean, stddev
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------
# 1. Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("Stock Forecasting + Anomaly Detection") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# -----------------------------
# 2. Load Data
# -----------------------------
df = spark.read.csv(
    "hdfs://localhost:9000/stock-data/aapl.csv",
    header=True,
    inferSchema=True
)

# Fix column name
df = df.withColumnRenamed("Price", "Date")

# Remove junk rows
df = df.filter(col("Date") != "Ticker")
df = df.filter(col("Date") != "Date")
df = df.filter(col("Close").isNotNull())

# Cast numeric
for c in ["Close", "Open", "High", "Low", "Volume"]:
    df = df.withColumn(c, col(c).cast("double"))

df = df.dropna()

# -----------------------------
# 3. Feature Engineering
# -----------------------------
window = Window.orderBy("Date")

df = df.withColumn("prev_close", lag("Close", 1).over(window))

# 🔥 KEY FIX → next day prediction
df = df.withColumn("next_close", lead("Close", 1).over(window))

df = df.dropna()

# -----------------------------
# 4. Prepare ML Data
# -----------------------------
assembler = VectorAssembler(
    inputCols=["Open", "High", "Low", "Volume", "prev_close"],
    outputCol="features"
)

data = assembler.transform(df).select(
    "Date", "features", col("next_close").alias("label")
)

# -----------------------------
# 5. TIME-BASED SPLIT (IMPORTANT)
# -----------------------------
train = data.orderBy("Date").limit(int(data.count() * 0.8))
test = data.orderBy("Date").subtract(train)

# -----------------------------
# 6. Train Model
# -----------------------------
lr = LinearRegression()
model = lr.fit(train)

# -----------------------------
# 7. Predictions
# -----------------------------
predictions = model.transform(test)

predictions.select("Date", "label", "prediction").show(10)

# -----------------------------
# 8. Evaluation Metrics
# -----------------------------
evaluator_rmse = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2"
)

print("RMSE:", evaluator_rmse.evaluate(predictions))
print("MAE:", evaluator_mae.evaluate(predictions))
print("R2:", evaluator_r2.evaluate(predictions))

# -----------------------------
# 9. Anomaly Detection
# -----------------------------
predictions = predictions.withColumn(
    "error", abs(col("label") - col("prediction"))
)

stats = predictions.select(mean("error"), stddev("error")).collect()
threshold = stats[0][0] + 1.5 * stats[0][1]

predictions = predictions.withColumn(
    "is_anomaly",
    (col("error") > threshold).cast("int")
)

predictions.select("label", "prediction", "error", "is_anomaly").show(20)

# -----------------------------
# 10. Save Output (FIXED)
# -----------------------------

# Drop vector column (IMPORTANT)
output_df = predictions.drop("features")

# Save locally
output_df.toPandas().to_csv("output.csv", index=False)

print("Local results saved!")

# Save to HDFS
output_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("/output/stock_results")

print("HDFS results saved!")

# -----------------------------
# 11. Stop
# -----------------------------
spark.stop()
