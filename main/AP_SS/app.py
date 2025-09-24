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
import pandas as pd
import glob
import shutil
from pathlib import Path
from datasketch import MinHash, MinHashLSH
try:
    from rapidfuzz import fuzz as rf_fuzz
except Exception:
    rf_fuzz = None

# --- Spark Integration ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import monotonically_increasing_id, col, lit, when
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

# Splink imports
from splink import Linker, SparkAPI
from splink.exploratory import profile_columns

try:
    import altair as alt  # For Altair chart handling from Splink explorers
except Exception:  # pragma: no cover
    alt = None
import plotly.io as pio

import auto_blocking as ab
from metaphone import doublemetaphone
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


def sanitize_settings_for_backend(settings_dict: dict) -> dict:
    """
    Clean up Splink settings for Spark backend:
      - Deduplicate comparisons based on output_column_name.
      - Validate that all comparison levels are dictionaries, providing a fallback if not.
      - Replace/strip unsupported SQL functions (e.g., Levenshtein) for Spark with a fallback.
      - Guarantee an exact match level for columns ending in '_norm'.
    """
    comparisons = settings_dict.get("comparisons", [])
    cleaned_comparisons = []
    seen_columns = set()

    print(f"Debug: Original comparisons: {comparisons}")

    for comp in comparisons:
        # Deduplicate comparisons based on output column name
        col_name = comp.get("output_column_name")
        if not col_name or col_name in seen_columns:
            continue
        seen_columns.add(col_name)

        original_levels = comp.get("comparison_levels", []) or []
        sanitized_levels = []
        has_exact_match = False

        # Single loop to both validate and sanitize levels
        for level in original_levels:
            # 1. Validate that the level is a dictionary. If not, replace with a valid fallback.
            if not isinstance(level, dict):
                print(f"Warning: Invalid comparison level in {col_name}: {level} (type: {type(level)}). Converting to exact match.")
                sanitized_levels.append({
                    "sql_condition": f"{col_name}_l = {col_name}_r",
                    "label_for_charts": "Exact match (fallback)",
                })
                continue

            # 2. Sanitize SQL conditions for Spark compatibility
            sql_cond = level.get("sql_condition", "")
            if not sql_cond:
                sanitized_levels.append(level)
                continue
            
            # Replace unsupported UDFs with a fallback to avoid errors
            if any(udf in sql_cond.upper() for udf in ["LEVENSHTEIN", "DAMERAU", "SOUNDEX"]):
                sanitized_levels.append({
                    "sql_condition": f"{col_name}_l = {col_name}_r",
                    "label_for_charts": "Exact (fallback)",
                })
                continue

            # Fix Trino-specific functions for Spark
            if "TRY_STRPTIME" in sql_cond:
                level["sql_condition"] = sql_cond.replace("TRY_STRPTIME", "to_date")
                level["label_for_charts"] = level.get("label_for_charts", "") + " (spark-safe)"

            sanitized_levels.append(level)

        # 3. Ensure normalized columns have at least one exact match level for model stability
        for lvl in sanitized_levels:
            sql_cond = lvl.get("sql_condition", "")
            label = lvl.get("label_for_charts", "").lower()
            if ("=" in sql_cond) or ("exact" in label):
                has_exact_match = True
                break
        
        if not has_exact_match and col_name.endswith("_norm"):
            sanitized_levels.insert(0, {
                "sql_condition": f"{col_name}_l = {col_name}_r",
                "label_for_charts": "Exact match",
                "is_model_blocking_level": True,
            })

        comp["comparison_levels"] = sanitized_levels
        cleaned_comparisons.append(comp)

    settings_dict["comparisons"] = cleaned_comparisons
    print(f"Debug: Sanitized comparisons: {cleaned_comparisons}")
    return settings_dict

def _register_similarity_udfs(spark):
    """
    Explicitly register Splink's Scala similarity UDFs for Spark.
    """
    udfs_to_register = {
        "jaro_winkler_similarity": ("uk.gov.moj.dash.linkage.JaroWinklerSimilarity", "double"),
        "damerau_levenshtein": ("uk.gov.moj.dash.linkage.DamerauLevenshtein", "int"),
        "levenshtein": ("uk.gov.moj.dash.linkage.Levenshtein", "int"),
        "soundex": ("uk.gov.moj.dash.linkage.Soundex", "string"),
    }
    for name, (cls, return_type) in udfs_to_register.items():
        try:
            spark.udf.registerJavaFunction(name, cls, return_type)
            print(f"✅ Registered UDF: {name} -> {cls}")
        except Exception as e:
            print(f"⚠️ Could not register {name}: {e}")

# --- Metaphone and Unidecode UDFs for Spark ---
def metaphone_py(s: str) -> str:
    if not s:
        return ""
    return doublemetaphone(s)[0]

metaphone_udf = udf(metaphone_py, StringType())

def unidecode_py(s: str) -> str:
    if not s or not unidecode:
        return s
    return unidecode(s)

unidecode_udf = udf(unidecode_py, StringType())


app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- In-memory job store for progress and artifacts ----------------
jobs: Dict[str, Dict[str, Any]] = {}
current_session_id: str | None = None

def _get_trino_connection(host: str, port: int, user: str, catalog: str, http_scheme: str):
    import trino
    return trino.dbapi.connect(
        host=host,
        port=port,
        user=user,
        catalog=catalog,
        http_scheme=http_scheme,
    )

def _purge_outputs_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "tmp_checkpoints")
    for d in [out_dir, checkpoint_dir]:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*")):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: could not remove path {path}. Reason: {e}")

def preprocess_data_spark(df: DataFrame) -> DataFrame:
    """Minimal preprocessing: drop rows with missing values using Spark."""
    return df.dropna()

def _build_spark_session(app_name: str) -> SparkSession:
    """
    Builds and returns a SparkSession with Splink's similarity JAR loaded,
    and registers all custom UDFs.
    """
    jar_path = r"C:\Users\AbhayPandey\AppData\Local\Programs\Python\Python311\Lib\site-packages\splink\internals\files\spark_jars\scala-udf-similarity-0.1.2_spark3.x.jar"
    print(f"Loading Splink similarity JAR from: {jar_path}")

    if not os.path.exists(jar_path):
        raise FileNotFoundError(f"Splink JAR not found at path: {jar_path}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars", jar_path)
        .config("spark.jars.packages", "io.trino:trino-jdbc:460")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Register similarity functions
    _register_similarity_udfs(spark)

    print(f"✅ Spark Session created with Splink JAR: {jar_path}")
    return spark

def _run_dedupe_job(job_id: str, creds: Dict[str, Any], schema: str, table: str):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 5
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    spark = None
    try:
        spark = _build_spark_session(f"SplinkDedupeJob-{job_id}")

        checkpoint_dir = os.path.join(os.path.dirname(__file__), "tmp_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        spark.sparkContext.setCheckpointDir(checkpoint_dir)

        # Load data from Trino
        jdbc_url = f"jdbc:trino://{creds['host']}:{creds['port']}/{creds['catalog']}"
        query = f"SELECT * FROM {creds['catalog']}.{schema}.{table}"

        df_spark = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", query)
            .option("user", creds["user"])
            .option("driver", "io.trino.jdbc.TrinoDriver")
            .load()
        )
        jobs[job_id]["progress"] = 15

        # Preprocess and add unique_id
        processed_spark_df = preprocess_data_spark(df_spark)
        processed_spark_df = processed_spark_df.withColumn(
            "unique_id", monotonically_increasing_id().cast("string")
        )

        # Ensure derived columns
        processed_spark_df = ab.ensure_derived_columns_enhanced_spark(processed_spark_df, {})

        jobs[job_id]["progress"] = 25

        db_api = SparkAPI(spark_session=spark)

        # Generate settings
        settings_dict, roles, diagnostics, deterministic_rules, df_enhanced_spark = (
            ab.auto_generate_settings(processed_spark_df, db_api=db_api)
        )

        # Sanitize settings for Spark
        settings_dict = sanitize_settings_for_backend(settings_dict)

        jobs[job_id]["progress"] = 35

        # Save artifacts
        roles_path = os.path.join(out_dir, f"roles_{job_id}.json")
        with open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f)

        full_data_path = os.path.join(out_dir, f"full_data_{job_id}.parquet")
        df_enhanced_spark.write.mode("overwrite").parquet(full_data_path)
        jobs[job_id]["progress"] = 45

        row_count = df_enhanced_spark.count()
        sample_df_spark = df_enhanced_spark.sample(
            fraction=min(1.0, 5000 / row_count if row_count > 0 else 1.0), seed=42
        )
        sample_path = os.path.join(out_dir, f"sample_{job_id}.parquet")
        sample_df_spark.write.mode("overwrite").parquet(sample_path)

        print("Starting Splink model training...")
        linker = Linker(df_enhanced_spark, settings_dict, db_api=db_api)

        random.seed(42)
        np.random.seed(42)

        try:
            linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000)
        except Exception as e:
            print(f"Warning: estimate_u_using_random_sampling failed: {e}")

        model_path = os.path.join(out_dir, f"trained_model_{job_id}.json")
        linker.misc.save_model_to_json(model_path, overwrite=True)
        jobs[job_id]["progress"] = 70

        # Inference
        df_predictions = linker.inference.predict()
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            df_predictions, threshold_match_probability=0.9
        )

        cluster_mapping = clusters.select("unique_id", "cluster_id")
        full_df_with_clusters_spark = df_enhanced_spark.join(
            cluster_mapping, on="unique_id", how="left"
        ).withColumn(
            "cluster_id",
            when(col("cluster_id").isNull(), col("unique_id")).otherwise(col("cluster_id")),
        )

        clusters_path = os.path.join(out_dir, f"splink_clusters_{job_id}.parquet")
        full_df_with_clusters_spark.write.mode("overwrite").parquet(clusters_path)

        preds_path = os.path.join(out_dir, f"splink_predictions_{job_id}.parquet")
        df_predictions.write.mode("overwrite").parquet(preds_path)

        jobs[job_id]["progress"] = 85
        report_path = os.path.join(out_dir, f"reports_{job_id}.parquet")
        full_df_with_clusters_spark.toPandas().to_parquet(report_path, index=False)
        jobs[job_id]["report_path"] = report_path

        jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "full_data_path": full_data_path,
                "sample_path": sample_path,
                "model_path": model_path,
                "roles_path": roles_path,
                "clusters_path": clusters_path,
                "report_path": report_path,
            }
        )
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 100
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()


def _check_record_against_clusters(
    job_id: str, new_record: Dict[str, Any], threshold: float = 0.95
) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise ValueError("Job not found or not completed.")

    paths = ["full_data_path", "model_path", "roles_path", "report_path"]
    if not all(p in job and os.path.exists(job[p]) for p in paths):
        raise ValueError("One or more required artifacts for checking are missing.")

    spark = None
    try:
        spark = _build_spark_session(f"SplinkCheckRecord-{job_id}")

        base_df_spark = spark.read.parquet(job["full_data_path"])

        # Ensure derived columns
        base_df_spark = ab.ensure_derived_columns_enhanced_spark(base_df_spark, {})

        with open(job["model_path"], "r", encoding="utf-8") as f:
            settings = json.load(f)

        # Sanitize settings for Spark
        settings = sanitize_settings_for_backend(settings)

        linker = Linker(base_df_spark, settings, db_api=SparkAPI(spark_session=spark))

        record_copy = dict(new_record)
        record_copy["unique_id"] = "new_record_to_check"
        new_df_pd = pd.DataFrame([record_copy])
        new_df = spark.createDataFrame(new_df_pd)

        # Align schema with base_df_spark
        for col_name in base_df_spark.columns:
            if col_name not in new_df.columns:
                new_df = new_df.withColumn(col_name, lit(None).cast(base_df_spark.schema[col_name].dataType))

        # Ensure derived columns
        new_df = ab.ensure_derived_columns_enhanced_spark(new_df, {})

        matches = linker.inference.find_matches_to_new_records(new_df)
        matches_pd = matches.toPandas()

        if matches_pd.empty:
            return {"result": "unique"}

        best_match = matches_pd.sort_values(by="match_probability", ascending=False).iloc[0]
        return {
            "result": "duplicate"
            if best_match["match_probability"] >= threshold
            else "potential_duplicate",
            "match_probability": float(best_match["match_probability"]),
        }
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()


# --- Flask Routes ---
@app.post("/reset")
def reset_server_state():
    try:
        _purge_outputs_dir()
        jobs.clear()
        global current_session_id
        current_session_id = None
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/")
def index():
    defaults = {
        "TRINO_HOST": os.getenv("TRINO_HOST", "3.108.199.0"),
        "TRINO_PORT": int(os.getenv("TRINO_PORT", "32092")),
        "TRINO_USER": os.getenv("TRINO_USER", "root"),
        "TRINO_CATALOG": os.getenv("TRINO_CATALOG", "hive"),
        "TRINO_HTTP_SCHEME": os.getenv("TRINO_HTTP_SCHEME", "http"),
    }
    return render_template("index.html", defaults=defaults)

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

    t = threading.Thread(target=_run_dedupe_job, args=(job_id, creds, schema, table), daemon=True)
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
    
    temp_csv_path = path.replace(".parquet", ".csv")
    pd.read_parquet(path).to_csv(temp_csv_path, index=False)
    return send_file(temp_csv_path, as_attachment=True, download_name="clusters.csv")

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
    
    temp_csv_path = report_path.replace(".parquet", ".csv")
    pd.read_parquet(report_path).to_csv(temp_csv_path, index=False)
    return send_file(temp_csv_path, as_attachment=True, download_name="reports.csv")

@app.get("/profile/<job_id>")
def get_profile(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    
    sample_path = job.get("sample_path")
    if not sample_path or not os.path.exists(sample_path):
        return jsonify({"ok": False, "error": "profile sample not available"}), 404
        
    try:
        spark = _build_spark_session(f"SplinkProfile-{job_id}")
        sample_df_spark = spark.read.parquet(sample_path)
        db_api = SparkAPI(spark_session=spark)
        figs = profile_columns(sample_df_spark, db_api=db_api)
        
        def _fig_to_html(fig):
            try: return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
            except Exception:
                try: return fig.to_html() if alt else ""
                except Exception: return ""
        
        html = "\n".join(_fig_to_html(f) for f in figs) if isinstance(figs, (list, tuple)) else _fig_to_html(figs)
        
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
        profile_path = os.path.join(out_dir, f"profile_{job_id}.html")
        with open(profile_path, 'w', encoding='utf-8') as fh:
            fh.write(html)
        job["profile_path"] = profile_path
        return jsonify({"ok": True, "html_path": f"/profile_html/{job_id}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if spark:
            print("Stopping Spark Session.")
            spark.stop()

@app.get("/profile_html/<job_id>")
def get_profile_html(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return Response("Job not found", status=404)
        
    profile_path = job.get("profile_path")
    if not profile_path or not os.path.exists(profile_path):
        return Response("Profile not generated or found.", status=404)

    with open(profile_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return Response(html_content, mimetype='text/html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)