import os
import io
import json
import uuid
import time
import shutil
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from flask import Flask, request, jsonify, send_file, render_template
import trino

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import col, lower, trim, regexp_replace, lit, concat_ws, coalesce, monotonically_increasing_id
from pyspark.sql.types import StringType


from rapidfuzz.fuzz import token_set_ratio

# Splink (Spark)
from splink import Linker
from splink import SparkAPI
from splink.backends.spark import similarity_jar_location

# Auto-blocking module from local file
import auto_blocking2 as ab


# ----------------------------
# App and global stats
# ----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Local path to your Scala UDF similarity jar from notes.ipynb
CUSTOM_JAR_PATH = os.path.join(BASE_DIR, "scala-udf-similarity-0.1.1-shaded.jar")

# In-memory job tracking
jobs: Dict[str, Dict[str, Any]] = {}

# Connection/session state
spark: Optional[SparkSession] = None
connection: Dict[str, Any] = {
    "connected": False,
    "trino": None,  # dict of host, port, catalog, schema, user
}

# Simple session id (frontend expects one)
SESSION_ID = uuid.uuid4().hex[:16]
current_session_id = SESSION_ID


# ----------------------------
# Spark helpers
# ----------------------------

def get_or_create_spark() -> SparkSession:
    global spark
    if spark is not None:
        return spark

    # Create SparkSession with Trino JDBC
    spark = (
        SparkSession.builder.appName("SplinkSparkApp")
        # Robust settings inspired by notes.ipynb
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "8g")
        .config("spark.python.worker.memory", "4g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.codegen.wholeStage", "true")
        .config("spark.ui.port", "4040")
        # Jars and packages: Trino + GraphFrames
        .config("spark.jars.packages", "io.trino:trino-jdbc:460,graphframes:graphframes:0.8.2-spark3.2-s_2.12")
        .config("spark.jars", f"{similarity_jar_location()},{CUSTOM_JAR_PATH}")
        .getOrCreate()
    )

    # Optional: set checkpoint directory
    try:
        spark.sparkContext.setCheckpointDir(os.path.join(BASE_DIR, "tmp_checkpoints"))
    except Exception:
        pass

    # Register Scala UDF similarity and helper functions from notes.ipynb
    try:
        from pyspark.sql.types import StringType, DoubleType, ArrayType
        spark.udf.registerJavaFunction("accent_remove", "uk.gov.moj.dash.linkage.AccentRemover", StringType())
        spark.udf.registerJavaFunction("double_metaphone", "uk.gov.moj.dash.linkage.DoubleMetaphone", StringType())
        spark.udf.registerJavaFunction("double_metaphone_alt", "uk.gov.moj.dash.linkage.DoubleMetaphoneAlt", StringType())

        spark.udf.registerJavaFunction("cosine_distance_custom", "uk.gov.moj.dash.linkage.CosineDistance", DoubleType())
        spark.udf.registerJavaFunction("jaccard_similarity", "uk.gov.moj.dash.linkage.JaccardSimilarity", DoubleType())
        spark.udf.registerJavaFunction("jaro_similarity", "uk.gov.moj.dash.linkage.JaroSimilarity", DoubleType())
        spark.udf.registerJavaFunction("jaro_winkler_similarity", "uk.gov.moj.dash.linkage.JaroWinklerSimilarity", DoubleType())
        spark.udf.registerJavaFunction("lev_damerau_distance", "uk.gov.moj.dash.linkage.LevDamerauDistance", DoubleType())

        spark.udf.registerJavaFunction("qgram_tokeniser", "uk.gov.moj.dash.linkage.QgramTokeniser", StringType())
        spark.udf.registerJavaFunction("q2gram_tokeniser", "uk.gov.moj.dash.linkage.Q2gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q3gram_tokeniser", "uk.gov.moj.dash.linkage.Q3gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q4gram_tokeniser", "uk.gov.moj.dash.linkage.Q4gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q5gram_tokeniser", "uk.gov.moj.dash.linkage.Q5gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q6gram_tokeniser", "uk.gov.moj.dash.linkage.Q6gramTokeniser", StringType())

        spark.udf.registerJavaFunction("dual_array_explode", "uk.gov.moj.dash.linkage.DualArrayExplode", ArrayType(StringType()))
        spark.udf.registerJavaFunction("latlong_explode", "uk.gov.moj.dash.linkage.latlongexplode", ArrayType(StringType()))

        spark.udf.registerJavaFunction("sql_escape", "uk.gov.moj.dash.linkage.sqlEscape", StringType())
    except Exception:
        # If the JAR is missing or class names mismatch, continue without raising here
        pass
    return spark


def trino_jdbc_url(cfg: Dict[str, Any]) -> str:
    host = cfg.get("host", "localhost")
    port = str(cfg.get("port", 8080))
    catalog = cfg.get("catalog", "system")
    # Remove schema from JDBC URL; dbtable will be fully qualified
    return f"jdbc:trino://{host}:{port}/{catalog}"


def _get_trino_connection(cfg: Dict[str, Any]):
    """Create a Trino DB-API connection for metadata queries."""
    return trino.dbapi.connect(
        host=cfg.get("host"),
        port=int(cfg.get("port", 8080)),
        user=str(cfg.get("user", "spark")),
        catalog=cfg.get("catalog"),
        schema=cfg.get("schema") or None,
    )


def read_trino_table_as_spark_df(cfg: Dict[str, Any], table: str) -> SparkDataFrame:
    sp = get_or_create_spark()
    url = trino_jdbc_url(cfg)
    props = {
        "driver": "io.trino.jdbc.TrinoDriver",
        "user": cfg.get("user", "spark"),
    }
    # Build fully qualified name catalog.schema.table
    parts = table.split(".")
    if len(parts) == 1:
        # Require schema in cfg when only table provided
        if not cfg.get("schema"):
            raise RuntimeError(
                "Schema not set. Provide table as 'schema.table' or 'catalog.schema.table', "
                "or call /run with {'schema': '...', 'table': '...'}"
            )
        fq_table = f"{cfg['catalog']}.{cfg['schema']}.{parts[0]}"
    elif len(parts) == 2:
        fq_table = f"{cfg['catalog']}.{parts[0]}.{parts[1]}"
    else:
        fq_table = table  # already fully qualified

    return (
        sp.read.format("jdbc")
        .option("url", url)
        .option("dbtable", fq_table)
        .options(**props)
        .load()
    )


def query_trino_as_df(cfg: Dict[str, Any], sql_text: str) -> SparkDataFrame:
    sp = get_or_create_spark()
    url = trino_jdbc_url(cfg)
    props = {
        "driver": "io.trino.jdbc.TrinoDriver",
        "user": cfg.get("user", "spark"),
    }
    wrapped = f"( {sql_text} ) t"
    return (
        sp.read.format("jdbc")
        .option("url", url)
        .option("dbtable", wrapped)
        .options(**props)
        .load()
    )


def write_single_csv(df: SparkDataFrame, out_path: str) -> None:
    temp_dir = out_path + "__tmp__"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    # Move the single part file to the final path
    part_file = None
    for name in os.listdir(temp_dir):
        if name.startswith("part-") and name.endswith(".csv"):
            part_file = os.path.join(temp_dir, name)
            break
    if not part_file:
        # Some Spark versions write .csv without suffix, fallback to any part-
        for name in os.listdir(temp_dir):
            if name.startswith("part-"):
                part_file = os.path.join(temp_dir, name)
                break
    if not part_file:
        raise RuntimeError("Could not find Spark CSV part file to move.")
    shutil.move(part_file, out_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _purge_outputs_dir():
    """Remove files for current session to keep outputs clean."""
    if not os.path.isdir(OUTPUTS_DIR):
        return
    for name in os.listdir(OUTPUTS_DIR):
        path = os.path.join(OUTPUTS_DIR, name)
        try:
            if os.path.isfile(path) and current_session_id in name:
                os.remove(path)
            elif os.path.isdir(path) and current_session_id in name:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


# ----------------------------
# Helper functions similar to app1.py
# ----------------------------

def ensure_first_last_from_name(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    name = (rec.get("name") or rec.get("full_name") or "").strip()
    if name and (not rec.get("first_name") or not rec.get("last_name")):
        parts = [p for p in name.split(" ") if p]
        if len(parts) >= 2:
            rec.setdefault("first_name", parts[0])
            rec.setdefault("last_name", parts[-1])
    return rec


def ensure_name_aliases(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    if rec.get("first_name") and rec.get("last_name") and not rec.get("full_name"):
        rec["full_name"] = f"{rec['first_name']} {rec['last_name']}"
    if rec.get("name") and not rec.get("full_name"):
        rec["full_name"] = rec["name"]
    return rec


def _pick_cols(df: SparkDataFrame, cols: List[str]) -> SparkDataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    return df.select(*existing)


# ----------------------------
# Fuzzy rescoring against clusters
# ----------------------------

def _normalize_text(v: Optional[str]) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _rf_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        return float(token_set_ratio(a, b)) / 100.0
    except Exception:
        return 0.0


def _check_record_against_clusters(record: Dict[str, Any], model_settings: Dict[str, Any], df_src: SparkDataFrame) -> Dict[str, Any]:
    """Use Splink to find matches to a new record and then apply fuzzy boost."""
    sp = get_or_create_spark()
    db_api = SparkAPI(spark_session=sp)
    linker = Linker(df_src, model_settings, db_api=db_api)

    # Build a single-row Spark DataFrame from the record with available columns
    r = ensure_name_aliases(ensure_first_last_from_name(record))
    cols = [c for c in df_src.columns if c in r]
    if not cols:
        return {"status": "unique", "match_probability": 0.0}
    row_df = sp.createDataFrame([ {c: r[c] for c in cols} ])

    preds = linker.inference.find_matches_to_new_records(row_df)
    rows = [x.asDict(recursive=True) for x in preds.limit(5000).collect()]
    if not rows:
        return {"status": "unique", "match_probability": 0.0}

    # Fuzzy rescoring boost on top of Splink scores
    target = {
        "first_name_norm": _normalize_text(r.get("first_name")),
        "last_name_norm": _normalize_text(r.get("last_name")),
        "full_name": _normalize_text(r.get("full_name")),
        "address_norm": _normalize_text(r.get("address")),
        "city_norm": _normalize_text(r.get("city")),
        "state_norm": _normalize_text(r.get("state")),
    }

    best = {"score": 0.0, "cluster_id": None, "partition_group": None}
    for row in rows:
        base_prob = float(row.get("match_probability", 0.0))
        name_score = max(
            _rf_ratio(target.get("full_name", ""), _normalize_text(row.get("full_name"))),
            _rf_ratio(target.get("first_name_norm", ""), _normalize_text(row.get("first_name_norm"))),
        )
        last_score = _rf_ratio(target.get("last_name_norm", ""), _normalize_text(row.get("last_name_norm")))
        addr_score = _rf_ratio(target.get("address_norm", ""), _normalize_text(row.get("address_norm")))
        city_score = _rf_ratio(target.get("city_norm", ""), _normalize_text(row.get("city_norm")))
        state_score = _rf_ratio(target.get("state_norm", ""), _normalize_text(row.get("state_norm")))

        composite = (
            0.45 * name_score + 0.25 * last_score + 0.15 * addr_score + 0.10 * city_score + 0.05 * state_score
        )
        if name_score > 0.9 and last_score > 0.9:
            composite = min(1.0, composite + 0.05)

        boosted = min(1.0, 0.7 * base_prob + 0.3 * composite)
        if boosted > best["score"]:
            best = {
                "score": boosted,
                "cluster_id": row.get("cluster_id"),
                "partition_group": row.get("partition_group"),
            }

    n = len(rows)
    base_thresh = 0.72 if n < 5000 else 0.75 if n < 20000 else 0.78
    potential_thresh = base_thresh - 0.05
    result_type = "unique"
    if best["score"] >= base_thresh:
        result_type = "duplicate"
    elif best["score"] >= potential_thresh:
        result_type = "potential_duplicate"

    return {
        "status": result_type,
        "cluster_id": best["cluster_id"],
        "partition_group": best["partition_group"],
        "match_probability": round(best["score"], 4),
    }


# ----------------------------
# Job orchestration
# ----------------------------

def _update_progress(job_id: str, stage: str, pct: float) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["progress"] = max(0.0, min(1.0, pct))
    job["stage"] = stage
    job["updated_at"] = datetime.utcnow().isoformat()


# ----------------------------
# Job worker
# ----------------------------
# ----------------------------
# Job worker
# ----------------------------
# def _run_dedupe_job(job_id: str, params: Dict[str, Any]) -> None:
#     """
#     Background worker to run Splink dedupe.

#     Examples:
#       # Run job with fully-qualified table
#       curl -s -X POST http://localhost:5000/run \
#         -H "Content-Type: application/json" \
#         -d '{"source":{"table":"hive.default.my_table"}}'

#       # Or with schema-qualified table
#       curl -s -X POST http://localhost:5000/run \
#         -H "Content-Type: application/json" \
#         -d '{"source":{"table":"default.my_table"}}'
#     """
#     try:
#         _update_progress(job_id, "starting", 0.02)
#         sp = get_or_create_spark()
#         db_api = SparkAPI(spark_session=sp)

#         # Validations with clear, separate messages:
#         if not connection.get("connected"):
#             raise RuntimeError("Trino is not connected. Please POST to /connect first.")

#         trino_cfg = connection.get("trino")
#         if not trino_cfg:
#             raise RuntimeError("Trino configuration missing. Did /connect succeed?")

#         # Support both nested {"source":{"table":...}} and flat {"schema":..., "table":...}
#         src = params.get("source") or {}
#         table = src.get("table") or params.get("table")
#         schema = params.get("schema")
#         # If schema chosen later, set into trino_cfg copy for this run
#         if schema and not trino_cfg.get("schema"):
#             trino_cfg = {**trino_cfg, "schema": schema}

#         if not table and schema:
#             # if schema + table provided separately, build fq_table
#             t = params.get("table")
#             if t:
#                 table = f"{schema}.{t}"

#         if not table:
#             raise RuntimeError(
#                 "No source table specified. "
#                 "Pass either {'source': {'table': 'catalog.schema.table'}} "
#                 "or flat {'schema': '...', 'table': '...'} to /run."
#             )


#         # Load source (supports table, schema.table, or catalog.schema.table)
#         df_src = read_trino_table_as_spark_df(trino_cfg, table)

#         # Ensure an unique_id exists; if not, create a synthetic
#         if "unique_id" not in df_src.columns:
#             df_src = df_src.withColumn("unique_id", concat_ws("-", *[col(c) for c in df_src.columns[:3]]))

#         _update_progress(job_id, "auto_blocking", 0.10)
#         settings, roles, diagnostics, df_enhanced = ab.auto_generate_settings(
#             df_src, db_api=db_api, spark=sp
#         )
#         df_enhanced = df_enhanced.unpersist()  # Clear any prior cache
#         # Persist enhanced DF to prevent recomputation of Python UDFs
#         from pyspark.storagelevel import StorageLevel
#         df_enhanced.persist(StorageLevel.MEMORY_AND_DISK)
#         # Materialize cache
#         _ = df_enhanced.count()

#         # Create a sampled training set (40%) to reduce Python UDF pressure, as per notes.ipynb
#         _update_progress(job_id, "linker_init", 0.22)
#         training_df = df_enhanced.sample(0.4, seed=42).cache()
#         _ = training_df.count()

#         # Training on the sampled DF using deterministic rules selected by auto_blocking2
#         training_linker = Linker(training_df, settings, db_api=db_api)
#         # Collect deterministic rules (kept=True) from diagnostics
#         deterministic_rules = [d.get("rule") for d in diagnostics if d.get("kept") and d.get("rule") is not None]

#         _update_progress(job_id, "training_prob", 0.28)
#         try:
#             training_linker.training.estimate_probability_two_random_records_match(
#                 deterministic_matching_rules=deterministic_rules,
#                 recall=0.95,
#             )
#         except Exception:
#             training_linker.training.estimate_probability_two_random_records_match(
#                 deterministic_matching_rules=deterministic_rules,
#                 recall=1.0,
#             )

#         _update_progress(job_id, "training_u", 0.32)
#         # Cap pairs to reduce load while keeping robustness
#         training_linker.training.estimate_u_using_random_sampling(max_pairs=2e6)

#         _update_progress(job_id, "training_em", 0.40)
#         # EM on deterministic rules from diagnostics (not blocking rules to generate predictions)
#         if deterministic_rules:
#             training_linker.training.estimate_parameters_using_expectation_maximisation(deterministic_rules)
#         else:
#             training_linker.training.estimate_parameters_using_expectation_maximisation()

#         # Save trained model
#         model_path = os.path.join(OUTPUTS_DIR, f"trained_model_{job_id}.json")
#         try:
#             training_linker.save_model_to_json(model_path)
#         except Exception:
#             with io.open(model_path, "w", encoding="utf-8") as f:
#                 json.dump(training_linker._settings_obj.as_dict(), f)

#         # Free training cache
#         try:
#             training_df.unpersist()
#         except Exception:
#             pass

#         # Reload trained model against full enhanced DF for inference
#         _update_progress(job_id, "predict", 0.55)
#         with io.open(model_path, "r", encoding="utf-8") as f:
#             trained_settings = json.load(f)
#         inference_linker = Linker(df_enhanced, trained_settings, db_api=db_api)
#         df_predictions = inference_linker.inference.predict()

#         # Adaptive threshold based on dataset size
#         total_rows = df_src.count()
#         if total_rows < 5000:
#             default_thresh = 0.45
#         elif total_rows < 20000:
#             default_thresh = 0.55
#         else:
#             default_thresh = 0.65
#         threshold = float(params.get("threshold", default_thresh))

#         _update_progress(job_id, "cluster", 0.70)
#         # Generate clusters and map back to the full original data using unique_id
#         clusters = inference_linker.clustering.cluster_pairwise_predictions_at_threshold(
#             df_predictions, threshold_match_probability=threshold
#         )
#         df_clusters = clusters

#         # Partition-group level report (cluster sizes and mean scores by partition if available)
#         _update_progress(job_id, "report", 0.78)
#         group_cols = [c for c in ["partition_group", "cluster_id"] if c in df_clusters.columns]
#         if group_cols:
#             # include optional mean score if available
#             score_col = "match_weight" if "match_weight" in df_clusters.columns else None
#             if score_col:
#                 from pyspark.sql.functions import avg
#                 report_df = df_clusters.groupBy(*group_cols).agg(
#                     avg(score_col).alias("mean_match_weight")
#                 )
#             else:
#                 report_df = df_clusters.groupBy(*group_cols).count()
#         else:
#             report_df = df_clusters.groupBy("cluster_id").count()

#         preds_path = os.path.join(OUTPUTS_DIR, f"splink_predictions_{job_id}.csv")
#         clus_path = os.path.join(OUTPUTS_DIR, f"splink_clusters_{job_id}.csv")
#         report_path = os.path.join(OUTPUTS_DIR, f"reports_{job_id}.csv")

#         _update_progress(job_id, "write_preds", 0.82)
#         write_single_csv(df_predictions, preds_path)

#         _update_progress(job_id, "write_clusters", 0.86)
#         write_single_csv(df_clusters, clus_path)

#         _update_progress(job_id, "write_report", 0.90)
#         write_single_csv(report_df, report_path)

#         # Save roles and parquet outputs
#         roles_path = os.path.join(OUTPUTS_DIR, f"roles_{job_id}.json")
#         try:
#             with io.open(roles_path, "w", encoding="utf-8") as f:
#                 json.dump(roles, f, indent=2)
#         except Exception:
#             pass
#         try:
#             df_predictions.write.mode("overwrite").parquet(os.path.join(OUTPUTS_DIR, f"predictions_{job_id}.parquet"))
#             df_clusters.write.mode("overwrite").parquet(os.path.join(OUTPUTS_DIR, f"clusters_{job_id}.parquet"))
#         except Exception:
#             pass

#         # Save deduped id mapping by joining clusters back to df_enhanced
#         try:
#             mapping = df_enhanced
#             if "unique_id" in mapping.columns:
#                 # ensure cluster id exists; if missing, use unique_id
#                 cluster_map = df_clusters.select("unique_id", "cluster_id") if set(["unique_id","cluster_id"]).issubset(set(df_clusters.columns)) else None
#                 if cluster_map is not None:
#                     full_map = mapping.join(cluster_map, on="unique_id", how="left")
#                     full_map = full_map.withColumn("cluster_id", coalesce(col("cluster_id"), col("unique_id")))
#                     # choose an id column to output
#                     id_col = "unique_id"
#                     mapping_path = os.path.join(OUTPUTS_DIR, f"deduped_{job_id}.csv")
#                     write_single_csv(full_map.select(id_col, "cluster_id").distinct(), mapping_path)
#         except Exception:
#             pass

#         # Profiling using Splink profile_columns
#         _update_progress(job_id, "profile", 0.93)
#         profile_html_path = os.path.join(OUTPUTS_DIR, f"profile_{job_id}.html")
#         try:
#             from splink.internals.profile_data import profile_columns
#             # choose a subset of columns if very wide
#             prof_cols = df_src.columns[:25]
#             html = profile_columns(Linker(df_src, settings, db_api=db_api), prof_cols)
#             with io.open(profile_html_path, "w", encoding="utf-8") as f:
#                 f.write(html if isinstance(html, str) else str(html))
#         except Exception:
#             with io.open(profile_html_path, "w", encoding="utf-8") as f:
#                 f.write("<html><body><p>Profile unavailable.</p></body></html>")

#         jobs[job_id].update({
#             "status": "completed",
#             "progress": 1.0,
#             "stage": "done",
#             "outputs": {
#                 "predictions": preds_path,
#                 "clusters": clus_path,
#                 "report": report_path,
#                 "profile_html": profile_html_path,
#                 "model": model_path,
#             },
#             "completed_at": datetime.utcnow().isoformat(),
#         })

#     except Exception as e:
#         err_txt = traceback.format_exc()
#         error_path = os.path.join(OUTPUTS_DIR, f"error_{job_id}.txt")
#         try:
#             with io.open(error_path, "w", encoding="utf-8") as f:
#                 f.write(err_txt)
#         except Exception:
#             error_path = None
#         jobs[job_id]["status"] = "failed"
#         jobs[job_id]["error"] = str(e)
#         jobs[job_id]["error_file"] = error_path
#         jobs[job_id]["stage"] = "error"
#         jobs[job_id]["progress"] = 1.0

def _run_dedupe_job(job_id: str, params: Dict[str, Any]) -> None:
    """
    Background worker to run Splink dedupe.

    Examples:
      # Run job with fully-qualified table
      curl -s -X POST http://localhost:5000/run \
        -H "Content-Type: application/json" \
        -d '{"source":{"table":"hive.default.my_table"}}'

      # Or with schema-qualified table
      curl -s -X POST http://localhost:5000/run \
        -H "Content-Type: application/json" \
        -d '{"source":{"table":"default.my_table"}}'
    """
    try:
        _update_progress(job_id, "starting", 0.02)
        sp = get_or_create_spark()
        db_api = SparkAPI(spark_session=sp)

        # Validations with clear, separate messages:
        if not connection.get("connected"):
            raise RuntimeError("Trino is not connected. Please POST to /connect first.")

        trino_cfg = connection.get("trino")
        if not trino_cfg:
            raise RuntimeError("Trino configuration missing. Did /connect succeed?")

        # Support both nested {"source":{"table":...}} and flat {"schema":..., "table":...}
        src = params.get("source") or {}
        table = src.get("table") or params.get("table")
        schema = params.get("schema")
        # If schema chosen later, set into trino_cfg copy for this run
        if schema and not trino_cfg.get("schema"):
            trino_cfg = {**trino_cfg, "schema": schema}

        if not table and schema:
            # if schema + table provided separately, build fq_table
            t = params.get("table")
            if t:
                table = f"{schema}.{t}"

        if not table:
            raise RuntimeError(
                "No source table specified. "
                "Pass either {'source': {'table': 'catalog.schema.table'}} "
                "or flat {'schema': '...', 'table': '...'} to /run."
            )

        # Load source (supports table, schema.table, or catalog.schema.table)
        df_src = read_trino_table_as_spark_df(trino_cfg, table)

        # Ensure a unique_id exists; if not, create a synthetic one
        if "unique_id" not in df_src.columns:
            df_src = df_src.withColumn("unique_id", monotonically_increasing_id().cast(StringType()))

        _update_progress(job_id, "auto_blocking", 0.10)
        settings, roles, diagnostics, df_enhanced = ab.auto_generate_settings(
            df_src, db_api=db_api, spark=sp
        )
        
        # Persist enhanced DF to prevent recomputation of Python UDFs
        from pyspark.storagelevel import StorageLevel
        df_enhanced.persist(StorageLevel.MEMORY_AND_DISK)
        # Materialize cache by calling an action
        _ = df_enhanced.count()

        # Create a sampled training set (40%) to reduce Python UDF pressure
        _update_progress(job_id, "linker_init", 0.22)
        training_df = df_enhanced.sample(0.4, seed=42).cache()
        _ = training_df.count()

        # Training on the sampled DF using deterministic rules selected by auto_blocking2
        training_linker = Linker(training_df, settings, db_api=db_api)
        deterministic_rules = [d.get("rule") for d in diagnostics if d.get("kept") and d.get("rule") is not None]

        _update_progress(job_id, "training_prob", 0.28)
        try:
            training_linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=deterministic_rules,
                recall=0.95,
            )
        except Exception:
            training_linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=deterministic_rules,
                recall=1.0,
            )

        _update_progress(job_id, "training_u", 0.32)
        training_linker.training.estimate_u_using_random_sampling(max_pairs=2e6)

        _update_progress(job_id, "training_em", 0.40)
        # **FIX 1:** Train on the FULL list of rules for a more robust model.
        # This matches the best practice from your notebook.
        # Ensure your Flask environment's Splink version matches your notebook's.
        if deterministic_rules:
            training_linker.training.estimate_parameters_using_expectation_maximisation(deterministic_rules)
        else:
            training_linker.training.estimate_parameters_using_expectation_maximisation()

        # Save trained model
        model_path = os.path.join(OUTPUTS_DIR, f"trained_model_{job_id}.json")
        training_linker.save_model_to_json(model_path, overwrite=True)

        # Free training cache
        training_df.unpersist()

        # Reload trained model against full enhanced DF for inference
        _update_progress(job_id, "predict", 0.55)
        # No need to load from JSON, we can just create a new Linker with the full dataframe
        inference_linker = Linker(df_enhanced, training_linker.settings.as_dict(), db_api=db_api)
        predictions_splink = inference_linker.inference.predict()

        # Adaptive threshold based on dataset size
        total_rows = df_src.count()
        threshold = 0.65
        if total_rows < 5000:
            threshold = 0.45
        elif total_rows < 20000:
            threshold = 0.55
        threshold = float(params.get("threshold", threshold))

        _update_progress(job_id, "cluster", 0.70)
        # Generate clusters (returns a Splink DataFrame wrapper)
        clusters_splink = inference_linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions_splink, threshold_match_probability=threshold
        )
        
        # **FIX 2:** Convert Splink objects to native PySpark DataFrames for all further processing.
        # This solves the 'no attribute groupBy' error.
        df_predictions = predictions_splink.as_spark_dataframe()
        df_clusters = clusters_splink.as_spark_dataframe()

        # Partition-group level report (now runs on a native Spark DataFrame)
        _update_progress(job_id, "report", 0.78)
        group_cols = [c for c in ["partition_group", "cluster_id"] if c in df_clusters.columns]
        if group_cols:
            score_col = "match_weight" if "match_weight" in df_clusters.columns else None
            if score_col:
                from pyspark.sql.functions import avg
                report_df = df_clusters.groupBy(*group_cols).agg(avg(score_col).alias("mean_match_weight"))
            else:
                report_df = df_clusters.groupBy(*group_cols).count()
        else:
            report_df = df_clusters.groupBy("cluster_id").count()

        preds_path = os.path.join(OUTPUTS_DIR, f"splink_predictions_{job_id}.csv")
        clus_path = os.path.join(OUTPUTS_DIR, f"splink_clusters_{job_id}.csv")
        report_path = os.path.join(OUTPUTS_DIR, f"reports_{job_id}.csv")

        _update_progress(job_id, "write_preds", 0.82)
        write_single_csv(df_predictions, preds_path)

        _update_progress(job_id, "write_clusters", 0.86)
        write_single_csv(df_clusters, clus_path)

        _update_progress(job_id, "write_report", 0.90)
        write_single_csv(report_df, report_path)

        # Save roles and parquet outputs (using native Spark DataFrames)
        roles_path = os.path.join(OUTPUTS_DIR, f"roles_{job_id}.json")
        with io.open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f, indent=2)

        df_predictions.write.mode("overwrite").parquet(os.path.join(OUTPUTS_DIR, f"predictions_{job_id}.parquet"))
        df_clusters.write.mode("overwrite").parquet(os.path.join(OUTPUTS_DIR, f"clusters_{job_id}.parquet"))

        # Save deduped id mapping by joining clusters back to df_enhanced (using native Spark DataFrames)
        if "unique_id" in df_enhanced.columns and "cluster_id" in df_clusters.columns:
            cluster_map = df_clusters.select("unique_id", "cluster_id")
            full_map = df_enhanced.join(cluster_map, on="unique_id", how="left")
            full_map = full_map.withColumn("cluster_id", coalesce(col("cluster_id"), col("unique_id")))
            mapping_path = os.path.join(OUTPUTS_DIR, f"deduped_{job_id}.csv")
            write_single_csv(full_map.select("unique_id", "cluster_id").distinct(), mapping_path)

        # Profiling using Splink profile_columns
        _update_progress(job_id, "profile", 0.93)
        profile_html_path = os.path.join(OUTPUTS_DIR, f"profile_{job_id}.html")
        try:
            from splink.internals.profile_data import profile_columns
            prof_cols = df_src.columns[:25]
            # Create a temporary Linker for profiling
            profiling_linker = Linker(df_src, settings, db_api=db_api)
            html = profile_columns(profiling_linker, prof_cols)
            with io.open(profile_html_path, "w", encoding="utf-8") as f:
                f.write(html if isinstance(html, str) else str(html))
        except Exception:
            with io.open(profile_html_path, "w", encoding="utf-8") as f:
                f.write("<html><body><p>Profile unavailable.</p></body></html>")

        # Clean up the main cache
        df_enhanced.unpersist()

        jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "stage": "done",
            "outputs": {
                "predictions": preds_path,
                "clusters": clus_path,
                "report": report_path,
                "profile_html": profile_html_path,
                "model": model_path,
            },
            "completed_at": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        err_txt = traceback.format_exc()
        error_path = os.path.join(OUTPUTS_DIR, f"error_{job_id}.txt")
        try:
            with io.open(error_path, "w", encoding="utf-8") as f:
                f.write(err_txt)
        except Exception:
            error_path = None
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["error_file"] = error_path
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["progress"] = 1.0

# ----------------------------
# Routes
# ----------------------------
def _default_ui_config() -> Dict[str, Any]:
    return {
        "TRINO_HOST": os.environ.get("TRINO_HOST", ""),
        "TRINO_PORT": os.environ.get("TRINO_PORT", ""),
        "TRINO_USER": os.environ.get("TRINO_USER", ""),
        "TRINO_CATALOG": os.environ.get("TRINO_CATALOG", ""),
    }


@app.route("/")
def index():
    return render_template("index.html", defaults=_default_ui_config())


@app.route("/session", methods=["GET"])
def get_session():
    return jsonify({
        "ok": True,
        "session_id": SESSION_ID,
        "connected": connection.get("connected", False),
        "trino": connection.get("trino"),
    })


# ----------------------------
# /connect endpoint
# ----------------------------
# ----------------------------
# /connect endpoint
# ----------------------------
@app.route("/connect", methods=["POST"])
def connect_trino():
    """
    Connect to Trino and return available schemas.

    Examples:
      # Get session info
      curl -s http://localhost:5000/session

      # Connect to Trino (schema NOT required here; user picks later)
      curl -s -X POST http://localhost:5000/connect \
        -H "Content-Type: application/json" \
        -d '{"host":"localhost","port":8080,"catalog":"hive","user":"admin"}'

      # List tables for a chosen schema later:
      curl -s -X POST http://localhost:5000/tables \
        -H "Content-Type: application/json" \
        -d '{"schema":"default"}'

      # Run job (table may be schema.table or catalog.schema.table)
      curl -s -X POST http://localhost:5000/run \
        -H "Content-Type: application/json" \
        -d '{"source":{"table":"hive.default.my_table"}}'
    """
    data = request.get_json(force=True)

    # Schema is OPTIONAL at connect time; it's selected later via dropdown
    required = ["host", "port", "catalog", "user"]
    for k in required:
        if k not in data or data[k] in (None, ""):
            return jsonify({"ok": False, "error": f"Missing '{k}'"}), 400

    cfg = {
        "host": data["host"],
        "port": data["port"],
        "catalog": data["catalog"],
        "schema": data.get("schema"),  # optional here
        "user": data["user"],
    }

    # Test connection and fetch schemas using Trino DB-API
    try:
        with _get_trino_connection({**cfg, "schema": None}) as conn:
            cur = conn.cursor()
            cur.execute("SHOW SCHEMAS")
            schemas = [row[0] for row in cur.fetchall()]
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to connect to Trino: {str(e)}"}), 400

    connection["connected"] = True
    connection["trino"] = cfg  # keep structure; schema may be None here
    get_or_create_spark()  # ensure spark up
    return jsonify({"ok": True, "schemas": schemas})


@app.route("/tables", methods=["POST"])
def list_tables():
    if not connection.get("connected"):
        return jsonify({"ok": False, "error": "Not connected"}), 400
    cfg = connection["trino"].copy()
    data = request.get_json(force=True) if request.data else {}
    schema = data.get("schema") or cfg.get("schema")
    if not schema:
        return jsonify({"ok": False, "error": "Schema is required"}), 400
    try:
        with _get_trino_connection({**cfg, "schema": schema}) as conn:
            cur = conn.cursor()
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
        return jsonify({"ok": True, "tables": tables})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/columns", methods=["POST"])
def list_columns():
    if not connection.get("connected"):
        return jsonify({"ok": False, "error": "Not connected"}), 400
    cfg = connection["trino"]
    data = request.get_json(force=True) if request.data else {}
    table = data.get("table")
    schema = data.get("schema") or cfg.get("schema")
    if not table:
        return jsonify({"ok": False, "error": "Missing table"}), 400

    try:
        # Parse table into schema and name, accepting 1-3 part identifiers
        parts = table.split(".")
        if len(parts) == 1:
            schema, tname = schema, parts[0]
        elif len(parts) == 2:
            schema, tname = parts[0], parts[1]
        else:
            # catalog.schema.table
            _, schema, tname = parts[-3], parts[-2], parts[-1]
        with _get_trino_connection({**cfg, "schema": schema}) as conn:
            cur = conn.cursor()
            cur.execute(f"DESCRIBE {tname}")
            rows = [{"name": r[0], "type": r[1]} for r in cur.fetchall()]
        return jsonify({"ok": True, "columns": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/reset", methods=["POST"])
def reset_state():
    global jobs, current_session_id
    jobs = {}
    current_session_id = uuid.uuid4().hex[:16]
    _purge_outputs_dir()
    return jsonify({"ok": True, "session_id": current_session_id})


@app.route("/run", methods=["POST"])
def run_job():
    if not connection.get("connected"):
        return jsonify({"error": "Not connected"}), 400

    params = request.get_json(force=True) if request.data else {}
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "progress": 0.0,
        "stage": "created",
        "created_at": datetime.utcnow().isoformat(),
        "outputs": {},
        "params": params,
    }

    t = threading.Thread(target=_run_dedupe_job, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    # frontend expects percentage 0-100
    p = int(round((job.get("progress") or 0) * 100))
    status = job.get("status")
    if status == "failed":
        status_out = "error"
    elif status == "completed":
        status_out = "completed"
    else:
        status_out = "running"
    return jsonify({
        "ok": True,
        "id": job_id,
        "status": status_out,
        "progress": p,
        "stage": job.get("stage"),
        "outputs": job.get("outputs", {}),
        "error": job.get("error"),
        "error_file": job.get("error_file"),
    })


@app.route("/download/<job_id>")
def download_clusters(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("outputs", {}).get("clusters")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "clusters not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route("/report/<job_id>")
def download_report(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("outputs", {}).get("splink_clusters")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "report not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route("/profile/<job_id>")
def profile_csv(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    # Provide predictions CSV for profiling purposes
    path = job.get("outputs", {}).get("predictions")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "profile csv not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route("/profile_html/<job_id>")
def profile_html(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("outputs", {}).get("profile_html")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "profile html not found"}), 404
    return send_file(path, as_attachment=False)


@app.route("/check_record", methods=["POST"])
def check_record():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    record = data.get("record") or {}
    if not job_id or job_id not in jobs:
        return jsonify({"ok": False, "error": "invalid job_id"}), 400
    job = jobs[job_id]
    outputs = job.get("outputs", {})
    model_path = outputs.get("model")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"ok": False, "error": "trained model not available for this job"}), 400

    # Load model settings for find_matches_to_new_records
    try:
        with io.open(model_path, "r", encoding="utf-8") as f:
            model_settings = json.load(f)
    except Exception as e:
        return jsonify({"ok": False, "error": f"could not load model settings: {str(e)}"}), 400

    # Recreate df_src from source for this job
    params = job.get("params", {})
    src = params.get("source", {})
    table = src.get("table") or params.get("table")
    schema = params.get("schema")
    if not table and schema:
        t = params.get("table")
        if t:
            table = f"{schema}.{t}"

    trino_cfg = connection.get("trino")
    if not table or not trino_cfg:
        return jsonify({"ok": False, "error": "source not available to re-evaluate record"}), 400
    df_src = read_trino_table_as_spark_df(trino_cfg, table)
    result = _check_record_against_clusters(record, model_settings, df_src)
    # adapt to frontend expectation keys
    return jsonify({
        "ok": True,
        "result": result.get("status"),
        "cluster_id": result.get("cluster_id"),
        "partition_group": result.get("partition_group"),
        "match_probability": result.get("match_probability"),
    })


if __name__ == "__main__":
    # Ensure Spark session starts fast on app run
    get_or_create_spark()
    app.run(host="0.0.0.0", port=5000, debug=False)
