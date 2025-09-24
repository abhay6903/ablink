from pyspark.sql import SparkSession
from splink.backends.spark import similarity_jar_location

# Build Spark with Splink similarity JAR
jar_path = similarity_jar_location()
print(f"Using Splink similarity JAR at: {jar_path}")

spark = (
    SparkSession.builder
    .appName("CheckSplinkUDFs")
    .config("spark.jars", jar_path)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# List all functions Spark currently knows
print("\n=== Registered functions (showing only those with similarity) ===")
all_funcs = spark.sql("SHOW FUNCTIONS").toPandas()

# Filter only Splink ones (case-insensitive search)
splink_funcs = all_funcs[all_funcs["function"].str.contains("similarity|levenshtein|soundex|jaccard|qgram", case=False)]
print(splink_funcs)

# Quick test: try calling jaro_winkler_similarity directly
try:
    spark.sql("SELECT jaro_winkler_similarity('martha', 'marhta') as jw").show()
except Exception as e:
    print(f"⚠️ Test failed: {e}")

spark.stop()
