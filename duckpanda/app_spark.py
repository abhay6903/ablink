from __future__ import annotations
import os
import threading
import time
import uuid
import json
import random
import traceback
from typing import Dict, Any, List, Tuple
from functools import reduce
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask import Response
import glob
import shutil
try:
    from rapidfuzz import fuzz as rf_fuzz
except Exception:
    rf_fuzz = None

# Spark Integration
try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import monotonically_increasing_id, col, lower, regexp_replace, trim
    from pyspark.sql.types import StringType, DoubleType, ArrayType
    from pyspark.sql.window import Window
    import pyspark.sql.functions as F
except ImportError:
    SparkSession = None
    DataFrame = None

# External deps expected: trino, splink
try:
    import trino
except Exception:
    trino = None

from splink import Linker, SparkAPI
from splink.exploratory import profile_columns
try:
    import altair as alt
except Exception:
    alt = None
import plotly.io as pio

import auto_blocking as ab

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ---------------- In-memory job store for progress and artifacts ----------------
jobs: Dict[str, Dict[str, Any]] = {}
current_session_id: str | None = None

def _get_trino_connection(host: str, port: int, user: str, catalog: str, http_scheme: str):
    if trino is None:
        raise RuntimeError("Python package 'trino' not installed")
    return trino.dbapi.connect(
        host=host,
        port=port,
        user=user,
        catalog=catalog,
        http_scheme=http_scheme,
    )

def _purge_outputs_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    if not os.path.isdir(out_dir):
        return
    for path in glob.glob(os.path.join(out_dir, "*")):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: could not remove path {path}. Reason: {e}")

# Global Spark session
_global_spark_session = None

def _get_or_create_spark_session() -> SparkSession:
    """Get or create a global Spark session with UDF registration"""
    global _global_spark_session
    
    if SparkSession is None:
        raise RuntimeError("PySpark is not installed")
    
    if _global_spark_session is not None:
        return _global_spark_session
    
    # Path to custom JAR file
    CUSTOM_JAR_PATH = os.path.join(os.path.dirname(__file__), "..", "scala-udf-similarity-0.1.1-shaded.jar")
    
    # Create Spark session exactly like in the notebook
    conf = SparkConf()
    conf.set("spark.driver.memory", "12g")
    conf.set("spark.default.parallelism", "8")
    conf.set("spark.sql.codegen.wholeStage", "false")
    conf.set("spark.jars", CUSTOM_JAR_PATH)
    
    sc = SparkContext.getOrCreate(conf=conf)
    _global_spark_session = SparkSession(sc)
    _global_spark_session.sparkContext.setCheckpointDir("./tmp_checkpoints")
    
    # Register UDFs
    _global_spark_session.udf.registerJavaFunction("accent_remove", "uk.gov.moj.dash.linkage.AccentRemover", StringType())
    _global_spark_session.udf.registerJavaFunction("double_metaphone", "uk.gov.moj.dash.linkage.DoubleMetaphone", StringType())
    _global_spark_session.udf.registerJavaFunction("double_metaphone_alt", "uk.gov.moj.dash.linkage.DoubleMetaphoneAlt", StringType())
    _global_spark_session.udf.registerJavaFunction("cosine_distance", "uk.gov.moj.dash.linkage.CosineDistance", DoubleType())
    _global_spark_session.udf.registerJavaFunction("jaccard_similarity", "uk.gov.moj.dash.linkage.JaccardSimilarity", DoubleType())
    _global_spark_session.udf.registerJavaFunction("jaro_similarity", "uk.gov.moj.dash.linkage.JaroSimilarity", DoubleType())
    _global_spark_session.udf.registerJavaFunction("jaro_winkler_similarity", "uk.gov.moj.dash.linkage.JaroWinklerSimilarity", DoubleType())
    _global_spark_session.udf.registerJavaFunction("lev_damerau_distance", "uk.gov.moj.dash.linkage.LevDamerauDistance", DoubleType())
    
    return _global_spark_session

def _initialize_spark_session(job_id: str) -> SparkSession:
    """Initialize Spark session with custom JAR and UDF registration"""
    return _get_or_create_spark_session()

@app.post("/reset")
def reset_server_state():
    """Clear outputs folder and reset in-memory job/session state."""
    try:
        _purge_outputs_dir()
        jobs.clear()
        global current_session_id
        current_session_id = None
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def validate_schema_and_tables(host: str, port: int, user: str, catalog: str, http_scheme: str,
                               schema: str, tables: List[str]) -> bool:
    try:
        conn = _get_trino_connection(host, port, user, catalog, http_scheme)
        cur = conn.cursor()
        cur.execute(f"SHOW SCHEMAS IN {catalog}")
        schemas = [r[0] for r in cur.fetchall()]
        if schema not in schemas:
            return False
        cur.execute(f"SHOW TABLES IN {catalog}.{schema}")
        available = [r[0] for r in cur.fetchall()]
        invalid = [t for t in tables if t not in available]
        return len(invalid) == 0
    except Exception:
        return False

def preprocess_data_spark(df: DataFrame) -> DataFrame:
    """Notebook parity: minimal preprocessing (drop rows with missing values) using Spark."""
    if DataFrame is None:
        raise RuntimeError("PySpark is not installed, cannot perform Spark operations.")
    return df.dropna()

def train_and_save_model_spark(path: str, df: DataFrame, settings_dict: dict, roles: dict,
                              sample_size: int = 20000, max_pairs_for_sampling: int = 2_000_000) -> Tuple[Linker, dict, list]:
    db_api = SparkAPI(spark_session=df.sql_ctx.sparkSession)
    
    # Sample data for training
    total_count = df.count()
    n = min(total_count, sample_size)
    if total_count > n:
        training_df = df.sample(fraction=n/total_count, seed=42)
    else:
        training_df = df

    train_linker = Linker(training_df, settings_dict, db_api=db_api)
    
    random.seed(42)
    np.random.seed(42)
    
    blocking_rules = settings_dict.get("blocking_rules_to_generate_predictions", []) or []
    def _estimate_with_recall(recall_value: float, use_rules: bool = True):
        rules_for_est = [r.get("blocking_rule") for r in blocking_rules if isinstance(r, dict) and r.get("blocking_rule")]
        if use_rules and rules_for_est:
            train_linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=rules_for_est,
                recall=recall_value,
            )
        else:
            train_linker.training.estimate_probability_two_random_records_match(
                recall=recall_value,
            )

    try:
        _estimate_with_recall(0.95, use_rules=True)
    except Exception as e:
        msg = str(e).lower()
        try:
            _estimate_with_recall(1.0, use_rules=True)
        except Exception:
            _estimate_with_recall(1.0, use_rules=False)
    
    train_linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs_for_sampling)
    for br in settings_dict.get("blocking_rules_to_generate_predictions", []):
        try:
            train_linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule=br.get("blocking_rule"))
        except Exception:
            try:
                train_linker.training.estimate_parameters_using_expectation_maximisation()
            except Exception:
                pass

    trained_settings_json = train_linker.misc.save_model_to_json(path, overwrite=True)
    
    with open(path, "r", encoding="utf-8") as fh:
        trained_settings = json.load(fh)
    full_linker = Linker(df, trained_settings, db_api=db_api)

    diagnostics: list = []
    return full_linker, roles, diagnostics

def generate_predictions_spark(linker, original_df: DataFrame, prediction_path: str, cluster_path: str, threshold: float):
    df_predictions = linker.inference.predict()
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predictions, threshold_match_probability=threshold
    )
    
    # Save predictions as CSV
    df_predictions.write.mode("overwrite").option("header", True).csv(prediction_path)
    
    # Save clusters as CSV
    clusters.write.mode("overwrite").option("header", True).csv(cluster_path)
    
    return df_predictions, clusters

def create_cluster_report_spark(df: DataFrame, preds: DataFrame, report_path: str) -> DataFrame:
    """Create cluster report using Spark operations - exactly like notebook"""
    # --- Step 1: Edges
    edges = preds.filter(F.col("match_probability") >= 0.9) \
                 .select(F.col("unique_id_l").alias("src"),
                         F.col("unique_id_r").alias("dst"))

    # --- Step 2: Components
    vertices = edges.select("src").union(edges.select("dst")).distinct() \
                   .withColumnRenamed("src", "id") \
                   .withColumn("component", F.col("id"))
    components = vertices
    changed = True
    while changed:
        updated = edges.join(components, edges.src == components.id, "inner") \
                       .select(edges.dst.alias("id"), components.component) \
                       .union(components.select("id", "component")) \
                       .groupBy("id").agg(F.min("component").alias("component"))
        changed = updated.join(components, ["id"], "left") \
                         .filter(updated.component != components.component).count() > 0
        components = updated

    # --- Step 3: Sequential cluster_id
    distinct_clusters = components.select("component").distinct() \
                                  .withColumn("cluster_id",
                                              F.row_number().over(Window.orderBy("component")))
    components = components.join(distinct_clusters, "component", "left")

    # --- Step 4: Join back with original df
    df_with_clusters = df.join(components, df.unique_id == components.id, "left") \
                         .drop("id", "component")

    # --- Step 5: Partition group
    window_spec = Window.partitionBy("cluster_id").orderBy("unique_id")
    df_with_clusters = df_with_clusters.withColumn(
        "partition_group", F.row_number().over(window_spec)
    )

    # --- Step 6: Save
    df_with_clusters.write.mode("overwrite").parquet(report_path)
    
    return df_with_clusters

def _run_dedupe_job_spark(job_id: str, creds: Dict[str, Any], schema: str, table: str):
    """Main deduplication job using Spark backend"""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 5
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    spark = None
    try:
        spark = _initialize_spark_session(job_id)
        jobs[job_id]["progress"] = 10

        # Load data from Trino
        jdbc_url = f"jdbc:trino://{creds['host']}:{creds['port']}/{creds['catalog']}"
        query = f"SELECT * FROM {creds['catalog']}.{schema}.{table}"
        
        df_spark = spark.read.format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .option("user", creds["user"]) \
            .option("driver", "io.trino.jdbc.TrinoDriver") \
            .load()
        jobs[job_id]["progress"] = 20

        processed_spark_df = preprocess_data_spark(df_spark)
        processed_spark_df = processed_spark_df.withColumn("unique_id", monotonically_increasing_id().cast("string"))
        jobs[job_id]["progress"] = 30

        # Use auto_blocking with SparkAPI
        db_api = SparkAPI(spark_session=spark)
        settings, roles, diagnostics, df_enhanced, deterministic_rules = ab.auto_generate_settings(processed_spark_df, db_api)
        jobs[job_id]["progress"] = 40

        def _parse_columns_from_rules(rules: List[Any]) -> List[str]:
            import re
            all_cols = set()
            for rule in rules:
                rule_sql_str = ""
                if isinstance(rule, dict):
                    rule_sql_str = rule.get("blocking_rule", "")
                elif isinstance(rule, str):
                    rule_sql_str = rule
                if rule_sql_str:
                    matches = re.findall(r'"([^"]+)"', rule_sql_str)
                    for col in matches:
                        all_cols.add(col)
            return sorted(list(all_cols))

        blocking_rules_list = settings.get("blocking_rules_to_generate_predictions", [])
        auto_grouping_cols = _parse_columns_from_rules(blocking_rules_list)
        jobs[job_id]["auto_grouping_cols"] = auto_grouping_cols

        roles_path = os.path.join(out_dir, f"roles_{job_id}.json")
        with open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f)

        # Ensure unique_id is properly set
        df_enhanced = df_enhanced.withColumn("unique_id", monotonically_increasing_id().cast("string"))
        
        full_data_path = os.path.join(out_dir, f"full_data_{job_id}.parquet")
        df_enhanced.write.mode("overwrite").parquet(full_data_path)
        jobs[job_id]["progress"] = 50

        # Sample for training
        total_count = df_enhanced.count()
        sample_size = min(5000, total_count)
        if total_count > sample_size:
            sample_df = df_enhanced.sample(fraction=sample_size/total_count, seed=42)
        else:
            sample_df = df_enhanced
        
        sample_path = os.path.join(out_dir, f"sample_{job_id}.parquet")
        sample_df.write.mode("overwrite").parquet(sample_path)
        
        model_path = os.path.join(out_dir, f"trained_model_{job_id}.json")
        linker, _, _ = train_and_save_model_spark(model_path, df_enhanced, settings, roles)
        jobs[job_id]["progress"] = 70

        preds_path = os.path.join(out_dir, f"splink_predictions_{job_id}.csv")
        clusters_path = os.path.join(out_dir, f"splink_clusters_{job_id}.csv")
        cluster_threshold = 0.99
        try:
            num_rows = df_enhanced.count()
            if num_rows < 1000: cluster_threshold = 0.85
            elif num_rows < 5000: cluster_threshold = 0.9
        except Exception: pass
        
        df_preds, clusters_df = generate_predictions_spark(linker, df_enhanced, preds_path, clusters_path, threshold=cluster_threshold)
        jobs[job_id]["progress"] = 85

        # Generate cluster report
        report_path = os.path.join(out_dir, f"reports_{job_id}.parquet")
        report_df = create_cluster_report_spark(df_enhanced, df_preds, report_path)
        jobs[job_id]["report_path"] = report_path

        # Create deduped representatives
        deduped = report_df.groupBy("cluster_id").agg(F.first("unique_id").alias("rep_id"))
        deduped_out = os.path.join(out_dir, f"deduped_{job_id}.csv")
        deduped.write.mode("overwrite").option("header", True).csv(deduped_out)

        jobs[job_id].update({
            "status": "completed", "progress": 100,
            "full_data_path": full_data_path, "sample_path": sample_path,
            "model_path": model_path, "roles_path": roles_path,
            "clusters_path": clusters_path, "report_path": report_path
        })
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 100
    finally:
        # Don't stop the global session
        pass

def _run_dedupe_job_csv(job_id: str, csv_path: str):
    """Main deduplication job using CSV upload with Spark backend"""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 5
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    spark = None
    try:
        spark = _initialize_spark_session(job_id)
        jobs[job_id]["progress"] = 10

        # Load CSV data
        df_spark = spark.read.csv(csv_path, header=True, inferSchema=True)
        jobs[job_id]["progress"] = 20

        processed_spark_df = preprocess_data_spark(df_spark)
        processed_spark_df = processed_spark_df.withColumn("unique_id", monotonically_increasing_id().cast("string"))
        jobs[job_id]["progress"] = 30

        # Use auto_blocking with SparkAPI
        db_api = SparkAPI(spark_session=spark)
        settings, roles, diagnostics, df_enhanced, deterministic_rules = ab.auto_generate_settings(processed_spark_df, db_api)
        jobs[job_id]["progress"] = 40

        def _parse_columns_from_rules(rules: List[Any]) -> List[str]:
            import re
            all_cols = set()
            for rule in rules:
                rule_sql_str = ""
                if isinstance(rule, dict):
                    rule_sql_str = rule.get("blocking_rule", "")
                elif isinstance(rule, str):
                    rule_sql_str = rule
                if rule_sql_str:
                    matches = re.findall(r'"([^"]+)"', rule_sql_str)
                    for col in matches:
                        all_cols.add(col)
            return sorted(list(all_cols))

        blocking_rules_list = settings.get("blocking_rules_to_generate_predictions", [])
        auto_grouping_cols = _parse_columns_from_rules(blocking_rules_list)
        jobs[job_id]["auto_grouping_cols"] = auto_grouping_cols

        roles_path = os.path.join(out_dir, f"roles_{job_id}.json")
        with open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f)

        # Ensure unique_id is properly set
        df_enhanced = df_enhanced.withColumn("unique_id", monotonically_increasing_id().cast("string"))
        
        full_data_path = os.path.join(out_dir, f"full_data_{job_id}.parquet")
        df_enhanced.write.mode("overwrite").parquet(full_data_path)
        jobs[job_id]["progress"] = 50

        # Sample for training
        total_count = df_enhanced.count()
        sample_size = min(5000, total_count)
        if total_count > sample_size:
            sample_df = df_enhanced.sample(fraction=sample_size/total_count, seed=42)
        else:
            sample_df = df_enhanced
        
        sample_path = os.path.join(out_dir, f"sample_{job_id}.parquet")
        sample_df.write.mode("overwrite").parquet(sample_path)
        
        model_path = os.path.join(out_dir, f"trained_model_{job_id}.json")
        linker, _, _ = train_and_save_model_spark(model_path, df_enhanced, settings, roles)
        jobs[job_id]["progress"] = 70

        preds_path = os.path.join(out_dir, f"splink_predictions_{job_id}.csv")
        clusters_path = os.path.join(out_dir, f"splink_clusters_{job_id}.csv")
        cluster_threshold = 0.99
        try:
            num_rows = df_enhanced.count()
            if num_rows < 1000: cluster_threshold = 0.85
            elif num_rows < 5000: cluster_threshold = 0.9
        except Exception: pass
        
        df_preds, clusters_df = generate_predictions_spark(linker, df_enhanced, preds_path, clusters_path, threshold=cluster_threshold)
        jobs[job_id]["progress"] = 85

        # Generate cluster report
        report_path = os.path.join(out_dir, f"reports_{job_id}.parquet")
        report_df = create_cluster_report_spark(df_enhanced, df_preds, report_path)
        jobs[job_id]["report_path"] = report_path

        # Create deduped representatives
        deduped = report_df.groupBy("cluster_id").agg(F.first("unique_id").alias("rep_id"))
        deduped_out = os.path.join(out_dir, f"deduped_{job_id}.csv")
        deduped.write.mode("overwrite").option("header", True).csv(deduped_out)

        jobs[job_id].update({
            "status": "completed", "progress": 100,
            "full_data_path": full_data_path, "sample_path": sample_path,
            "model_path": model_path, "roles_path": roles_path,
            "clusters_path": clusters_path, "report_path": report_path
        })
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 100
    finally:
        # Don't stop the global session
        pass

def _check_record_against_clusters(job_id: str, new_record: Dict[str, Any], threshold: float = 0.95) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise ValueError("Job not found or not completed.")

    paths = ["full_data_path", "model_path", "roles_path", "report_path"]
    if not all(p in job and os.path.exists(job[p]) for p in paths):
        raise ValueError("One or more required artifacts for checking are missing.")

    # Load data using Spark
    spark = _get_or_create_spark_session()
    try:
        base_df_spark = spark.read.parquet(job["full_data_path"])
        report_df_spark = spark.read.parquet(job["report_path"])
        
        with open(job["model_path"], "r", encoding="utf-8") as f:
            settings = json.load(f)
        with open(job["roles_path"], "r", encoding="utf-8") as f:
            roles = json.load(f)

        db_api = SparkAPI(spark_session=spark)
        linker = Linker(base_df_spark, settings, db_api=db_api)

        record_copy = dict(new_record)
        record_copy["unique_id"] = "new_record_to_check"
        new_df_spark = spark.createDataFrame([record_copy])

        # Ensure all columns exist
        for col_name in base_df_spark.columns:
            if col_name not in new_df_spark.columns:
                new_df_spark = new_df_spark.withColumn(col_name, F.lit(None))

        new_df_spark = ab.ensure_derived_columns_enhanced(new_df_spark, roles)

        matches = linker.inference.find_matches_to_new_records(new_df_spark)
        
        if matches.count() == 0:
            return {"result": "unique"}

        # Get the best match
        best_match = matches.orderBy(F.desc("match_probability")).limit(1).collect()[0]
        match_prob = best_match["match_probability"]
        
        adaptive_threshold = threshold
        try:
            num_rows = base_df_spark.count()
            if num_rows < 1000 and adaptive_threshold > 0.85: adaptive_threshold = 0.85
            elif num_rows < 5000 and adaptive_threshold > 0.9: adaptive_threshold = 0.9
        except Exception: pass
        
        if match_prob >= adaptive_threshold:
            result_type = "duplicate"
        elif match_prob >= 0.75:
            result_type = "potential_duplicate"
        else:
            return {"result": "unique"}
            
        best_match_id = str(best_match["unique_id_l"])
        match_info = report_df_spark.filter(F.col("unique_id") == best_match_id).collect()
        
        if not match_info:
            return {"result": result_type, "cluster_id": "N/A", "partition_group": "N/A", "match_probability": float(match_prob)}

        match_row = match_info[0]
        return {
            "result": result_type,
            "cluster_id": str(match_row["cluster_id"]),
            "partition_group": str(match_row["partition_group"]),
            "match_probability": float(match_prob),
        }
    finally:
        # Don't stop the global session
        pass

@app.get("/")
def index():
    defaults = {
        "TRINO_HOST": os.getenv("TRINO_HOST", "3.108.199.0"),
        "TRINO_PORT": int(os.getenv("TRINO_PORT", "32092")),
        "TRINO_USER": os.getenv("TRINO_USER", "root"),
        "TRINO_CATALOG": os.getenv("TRINO_CATALOG", "hive"),
        "TRINO_HTTP_SCHEME": os.getenv("TRINO_HTTP_SCHEME", "http"),
    }
    return render_template("index_spark.html", defaults=defaults)

@app.get("/session")
def get_session():
    sid = uuid.uuid4().hex[:16]
    return jsonify({"ok": True, "session_id": sid})

@app.post("/connect")
def connect_trino():
    data = request.get_json(force=True)
    host, port, user, catalog, http_scheme = (
        data.get("host"), int(data.get("port")), data.get("user"), 
        data.get("catalog"), data.get("http_scheme")
    )
    try:
        conn = _get_trino_connection(host, port, user, catalog, http_scheme)
        cur = conn.cursor()
        cur.execute("SHOW SCHEMAS")
        schemas = sorted([row[0] for row in cur.fetchall()])
        return jsonify({"ok": True, "schemas": schemas})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/tables")
def list_tables():
    data = request.get_json(force=True)
    host, port, user, catalog, http_scheme, schema = (
        data.get("host"), int(data.get("port")), data.get("user"),
        data.get("catalog"), data.get("http_scheme"), data.get("schema")
    )
    try:
        conn = _get_trino_connection(host, port, user, catalog, http_scheme)
        cur = conn.cursor()
        cur.execute(f"SHOW TABLES FROM {schema}")
        tables = sorted([row[0] for row in cur.fetchall()])
        return jsonify({"ok": True, "tables": tables})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/columns")
def list_columns():
    data = request.get_json(force=True)
    host, port, user, catalog, http_scheme, schema, table = (
        data.get("host"), int(data.get("port")), data.get("user"),
        data.get("catalog"), data.get("http_scheme"), data.get("schema"), data.get("table")
    )
    try:
        conn = _get_trino_connection(host, port, user, catalog, http_scheme)
        cur = conn.cursor()
        cur.execute(f"DESCRIBE {catalog}.{schema}.{table}")
        rows = cur.fetchall()
        cols = [{"name": r[0], "type": r[1]} for r in rows if r and r[0]]
        return jsonify({"ok": True, "columns": cols})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/upload_csv")
def upload_csv():
    """Handle CSV file upload"""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"ok": False, "error": "File must be a CSV"}), 400
    
    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    filename = f"{uuid.uuid4().hex[:12]}_{file.filename}"
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    
    return jsonify({"ok": True, "file_path": file_path, "filename": filename})

@app.post("/get_csv_columns")
def get_csv_columns():
    """Get column information from uploaded CSV file"""
    data = request.get_json(force=True)
    file_path = data.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"ok": False, "error": "File not found"}), 400
    
    try:
        # Use Spark to read CSV and get schema
        spark = _get_or_create_spark_session()
        try:
            df = spark.read.csv(file_path, header=True, inferSchema=True)
            columns = []
            for field in df.schema.fields:
                columns.append({
                    "name": field.name,
                    "type": str(field.dataType)
                })
            return jsonify({"ok": True, "columns": columns})
        finally:
            spark.stop()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/check_record")
def check_record():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    new_record = data.get("record") or {}
    threshold = float(data.get("threshold", 0.95))
    try:
        res = _check_record_against_clusters(job_id, new_record, threshold)
        return jsonify({"ok": True, **res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/run")
def run_job():
    data = request.get_json(force=True)
    creds = {k: data.get(k) for k in ["host", "port", "user", "catalog", "http_scheme"]}
    schema, table, session_id = data.get("schema"), data.get("table"), data.get("session_id")

    global current_session_id
    if session_id and session_id != current_session_id:
        _purge_outputs_dir()
        current_session_id = session_id

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0}

    t = threading.Thread(target=_run_dedupe_job_spark, args=(job_id, creds, schema, table), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})

@app.post("/run_csv")
def run_job_csv():
    data = request.get_json(force=True)
    file_path = data.get("file_path")
    session_id = data.get("session_id")

    global current_session_id
    if session_id and session_id != current_session_id:
        _purge_outputs_dir()
        current_session_id = session_id

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "progress": 0}

    t = threading.Thread(target=_run_dedupe_job_csv, args=(job_id, file_path), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})

@app.get("/progress/<job_id>")
def job_progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    return jsonify({"ok": True, **{k: v for k, v in job.items() if k in ("status", "progress", "error")}})

@app.get("/download/<job_id>")
def download_clusters(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("clusters_path")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "clusters file not available"}), 404
    return send_file(path, as_attachment=True, download_name="clusters.csv")

@app.get("/report/<job_id>")
def download_report(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"ok": False, "error": "job not completed"}), 400
    report_path = job.get("report_path")
    if not report_path or not os.path.exists(report_path):
        return jsonify({"ok": False, "error": "report file not available"}), 404
    # Convert Parquet to CSV for download
    temp_csv_path = report_path.replace(".parquet", ".csv")
    spark = _get_or_create_spark_session()
    try:
        df = spark.read.parquet(report_path)
        df.write.mode("overwrite").option("header", True).csv(temp_csv_path)
        return send_file(temp_csv_path, as_attachment=True, download_name="reports.csv")
    finally:
        # Don't stop the global session
        pass

@app.get("/profile/<job_id>")
def get_profile(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    profile_path = os.path.join(out_dir, f"value_distribution_{job_id}.html")
    
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as fh:
            return jsonify({"ok": True, "html": fh.read()})
    
    sample_path = job.get("sample_path")
    if not sample_path or not os.path.exists(sample_path):
        return jsonify({"ok": False, "error": "profile sample not available"}), 404
        
    try:
        # Load sample data using Spark
        spark = _get_or_create_spark_session()
        try:
            sample_df_spark = spark.read.parquet(sample_path)
            db_api = SparkAPI(spark_session=spark)
            figs = profile_columns(sample_df_spark, db_api=db_api)
            
            def _fig_to_html(fig):
                try: return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                except Exception:
                    try: return fig.to_html() if alt else ""
                    except Exception: return ""
            
            html = "\n".join(_fig_to_html(f) for f in figs) if isinstance(figs, (list, tuple)) else _fig_to_html(figs)
            
            with open(profile_path, 'w', encoding='utf-8') as fh:
                fh.write(html)
            job["profile_path"] = profile_path
            return jsonify({"ok": True, "html": html})
        finally:
            spark.stop()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/profile_html/<job_id>")
def get_profile_html(job_id: str):
    """Return the profile HTML directly for iframe embedding."""
    job = jobs.get(job_id)
    if not job:
        return Response("Job not found", status=404)
        
    profile_response = get_profile(job_id)
    if profile_response.status_code != 200:
        error_data = profile_response.get_json()
        return Response(error_data.get("error", "Could not generate profile"), status=profile_response.status_code)

    html = profile_response.get_json().get("html", "")
    return Response(html, status=200, mimetype='text/html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)