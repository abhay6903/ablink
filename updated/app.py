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
from difflib import SequenceMatcher
try:
    from rapidfuzz import fuzz as rf_fuzz
except Exception:
    rf_fuzz = None

# --- Spark Integration: Add PySpark imports ---
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import monotonically_increasing_id
except ImportError:
    SparkSession = None
    DataFrame = None
# --- End Spark Integration ---

# External deps expected: trino, duckdb, splink
try:
    import trino
except Exception:  # pragma: no cover
    trino = None

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None

from splink import Linker, DuckDBAPI
from splink.exploratory import profile_columns
try:
    import altair as alt  # For Altair chart handling from Splink explorers
except Exception:  # pragma: no cover
    alt = None
import plotly.io as pio

import auto_blocking as ab


app = Flask(__name__, template_folder="templates", static_folder="static")


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


# This function is no longer used by the main job but kept for utility/validation
def _fetch_table_full(conn, schema: str, table: str) -> pd.DataFrame:
    query = f"SELECT * FROM {schema}.{table}"
    return pd.read_sql(query, conn)


def _purge_outputs_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    if not os.path.isdir(out_dir):
        return
    # More comprehensive purge to include all job artifacts
    for path in glob.glob(os.path.join(out_dir, "*")):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: could not remove path {path}. Reason: {e}")


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


# =====================
# Notebook parity utils
# =====================

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

# --- Spark Integration: New preprocessing function for Spark DataFrames ---
def preprocess_data_spark(df: DataFrame) -> DataFrame:
    """Notebook parity: minimal preprocessing (drop rows with missing values) using Spark."""
    if DataFrame is None:
        raise RuntimeError("PySpark is not installed, cannot perform Spark operations.")
    return df.dropna()
# --- End Spark Integration ---

def train_and_save_model(path: str, df: pd.DataFrame, settings_dict: dict, roles: dict,
                         sample_size: int = 20000, max_pairs_for_sampling: int = 2_000_000) -> Tuple[Linker, dict, list]:
    db_api = DuckDBAPI()
    
    # Ensure derived columns are present before training
    df = ab.ensure_derived_columns_enhanced(df, roles)

    n = min(len(df), sample_size)
    training_df = df.sample(n=n, random_state=42) if len(df) > n else df.copy()

    train_linker = Linker(training_df, settings_dict, db_api=db_api)
    
    # Set seeds for deterministic training
    random.seed(42)
    np.random.seed(42)
    
    # Robust prior estimation: adapt recall to avoid failure across datasets
    blocking_rules = settings_dict.get("blocking_rules_to_generate_predictions", []) or []
    def _estimate_with_recall(recall_value: float, use_rules: bool = True):
        # Correctly handle that blocking_rules is a list of dicts
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
        # Start with a generous recall, then escalate if needed
        _estimate_with_recall(0.95, use_rules=True)
    except Exception as e:
        msg = str(e).lower()
        try:
            # Retry at maximum recall with rules
            _estimate_with_recall(1.0, use_rules=True)
        except Exception:
            # Fall back to estimating without deterministic rules
            _estimate_with_recall(1.0, use_rules=False)
    train_linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs_for_sampling)
    # Parameter estimation loop with safeguards: if a blocking rule fails, continue with others
    for br in settings_dict.get("blocking_rules_to_generate_predictions", []):
        try:
            # Pass the dictionary directly as expected by the function
            train_linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule=br.get("blocking_rule"))
        except Exception:
            # Fallback: try EM without blocking rule context
            try:
                train_linker.training.estimate_parameters_using_expectation_maximisation()
            except Exception:
                # Skip if still fails; continue with remaining rules
                pass

    # Save the trained model settings
    trained_settings_json = train_linker.misc.save_model_to_json(path, overwrite=True)
    
    # Re-create linker with the full dataset and the trained model
    with open(path, "r", encoding="utf-8") as fh:
        trained_settings = json.load(fh)
    full_linker = Linker(df, trained_settings, db_api=db_api)

    diagnostics: list = []
    return full_linker, roles, diagnostics


def generate_predictions(linker, original_df: pd.DataFrame, prediction_path: str, cluster_path: str, threshold: float):
    # --- Step 1: Get predictions and clusters as before ---
    df_predictions = linker.inference.predict()
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predictions, threshold_match_probability=threshold
    )
    # This dataframe only contains records that were part of a cluster
    clusters_pd = clusters.as_pandas_dataframe()

    # --- Step 2: NEW LOGIC - Merge clusters with the full original dataset ---
    # We select only the essential columns for the merge: the unique identifier and the cluster ID.
    if not clusters_pd.empty:
        cluster_mapping = clusters_pd[['unique_id', 'cluster_id']].copy()
        # Ensure data types are consistent for merging
        cluster_mapping['unique_id'] = cluster_mapping['unique_id'].astype(str)
        original_df['unique_id'] = original_df['unique_id'].astype(str)

        # Perform a LEFT merge to bring cluster IDs to every original record
        # All records from original_df will be kept.
        full_df_with_clusters = original_df.merge(
            cluster_mapping, on="unique_id", how="left"
        )
    else:
        # If no clusters were found at all, create the column manually
        full_df_with_clusters = original_df.copy()
        full_df_with_clusters['cluster_id'] = pd.NA


    # --- Step 3: NEW LOGIC - Assign a unique cluster_id to singleton records ---
    # Records that were not in a cluster will have a null (NaT/NaN) value for 'cluster_id'.
    # We fill these nulls with the record's own 'unique_id', making them a cluster of one.
    full_df_with_clusters['cluster_id'] = full_df_with_clusters['cluster_id'].fillna(
        full_df_with_clusters['unique_id']
    )

    # --- Step 4: Save the complete, partitioned DataFrame ---
    # The final CSV now contains ALL original records.
    full_df_with_clusters.to_csv(cluster_path, index=False)

    # For consistency, we can return the full dataframe. The raw predictions are less critical now.
    df_predictions_pd = df_predictions.as_pandas_dataframe()
    df_predictions_pd.to_csv(prediction_path, index=False)
    
    return df_predictions_pd, full_df_with_clusters


def get_deduped_id_mapping(df: pd.DataFrame) -> Dict[str, str]:
    # Ensure IDs are strings for consistency
    df['unique_id_l'] = df['unique_id_l'].astype(str)
    df['unique_id_r'] = df['unique_id_r'].astype(str)
    
    deduped_ids = df.groupby('unique_id_r')['unique_id_l'].first().reset_index()
    id_mapping = pd.Series(deduped_ids['unique_id_l'].values, index=deduped_ids['unique_id_r']).to_dict()
    return id_mapping


def deduplicate_by_mapped_ids(df: pd.DataFrame, column_name: str, id_mapping: Dict[str, str], output_path: str) -> pd.DataFrame:
    df = df.copy()
    # Ensure ID column is string before mapping
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].replace(id_mapping)
    df = df.drop_duplicates(subset=column_name, keep='last')
    df.to_csv(output_path, index=False)
    return df


def ensure_name_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both last_name and surname views exist, including their normalized/metaphone variants."""
    df = df.copy()
    # Base column aliasing
    if "last_name" in df.columns and "surname" not in df.columns:
        df["surname"] = df["last_name"]
    if "surname" in df.columns and "last_name" not in df.columns:
        df["last_name"] = df["surname"]
    # Derived variants
    pairs = [
        ("last_name_norm", "surname_norm"),
        ("last_name_metaphone", "surname_metaphone"),
    ]
    for col1, col2 in pairs:
        if col1 in df.columns and col2 not in df.columns:
            df[col2] = df[col1]
        if col2 in df.columns and col1 not in df.columns:
            df[col1] = df[col2]
    return df


def ensure_first_last_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """Create `first_name`, `last_name`, and `full_name` from a single `name` column if needed.
    This helps satisfy blocking rules like `first_name_first_char` when datasets only have `name`.
    """
    df = df.copy()
    if "name" in df.columns:
        name_series = df["name"].astype("string").fillna("")
        if "first_name" not in df.columns:
            df["first_name"] = name_series.str.split().str[0].fillna("")
        if "last_name" not in df.columns:
            # last token if available, else empty
            parts = name_series.str.split()
            df["last_name"] = parts.apply(lambda xs: xs[-1] if isinstance(xs, list) and len(xs) > 1 else "")
        if "full_name" not in df.columns:
            df["full_name"] = name_series
    return df


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
        # Use fully qualified name to avoid catalog ambiguity
        cur.execute(f"DESCRIBE {catalog}.{schema}.{table}")
        rows = cur.fetchall()
        cols = [{"name": r[0], "type": r[1]} for r in rows if r and r[0]]
        return jsonify({"ok": True, "columns": cols})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ================== CORRECTED FUNCTION ==================
def _run_dedupe_job(job_id: str, creds: Dict[str, Any], schema: str, table: str):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 5
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Spark Integration: Initialize SparkSession ---
    spark = None
    try:
        if SparkSession is None:
            raise RuntimeError("PySpark is not installed.")
        spark = SparkSession.builder \
            .appName(f"SplinkDedupeJob-{job_id}") \
            .config("spark.jars.packages", "io.trino:trino-jdbc:460") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
        jdbc_url = f"jdbc:trino://{creds['host']}:{creds['port']}/{creds['catalog']}"
        query = f"SELECT * FROM {creds['catalog']}.{schema}.{table}"
        
        df_spark = spark.read.format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .option("user", creds["user"]) \
            .option("driver", "io.trino.jdbc.TrinoDriver") \
            .load()
        jobs[job_id]["progress"] = 15

        processed_spark_df = preprocess_data_spark(df_spark)
        processed_spark_df = processed_spark_df.withColumn("unique_id", monotonically_increasing_id().cast("string"))

        df = processed_spark_df.toPandas()
        jobs[job_id]["progress"] = 25

        db_api = DuckDBAPI()
        df = ensure_first_last_from_name(df)
        settings_dict, roles, diagnostics = ab.auto_generate_settings(df, db_api)
        jobs[job_id]["progress"] = 35

        # --- BUG FIX IS HERE ---
        def _parse_columns_from_rules(rules: List[Any]) -> List[str]:
            import re
            all_cols = set()
            # Iterate through the list of rules
            for rule in rules:
                rule_sql_str = ""
                # Check if the rule is a dictionary and get the SQL string
                if isinstance(rule, dict):
                    rule_sql_str = rule.get("blocking_rule", "")
                # Check if it's already a string (for safety)
                elif isinstance(rule, str):
                    rule_sql_str = rule
                
                # Only proceed if we have a valid string to parse
                if rule_sql_str:
                    # This regex finds all column names enclosed in double quotes
                    matches = re.findall(r'"([^"]+)"', rule_sql_str)
                    for col in matches:
                        all_cols.add(col)
            return sorted(list(all_cols))

        # The variable from settings can contain dictionaries, not just strings
        blocking_rules_list = settings_dict.get("blocking_rules_to_generate_predictions", [])
        auto_grouping_cols = _parse_columns_from_rules(blocking_rules_list)
        jobs[job_id]["auto_grouping_cols"] = auto_grouping_cols
        # --- END OF BUG FIX ---

        roles_path = os.path.join(out_dir, f"roles_{job_id}.json")
        with open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f)

        df = ab.ensure_derived_columns_enhanced(df, roles)
        df = ensure_name_aliases(df)
        full_data_path = os.path.join(out_dir, f"full_data_{job_id}.parquet")
        df.to_parquet(full_data_path, index=False)
        jobs[job_id]["progress"] = 45

        sample_df = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df.copy()
        sample_path = os.path.join(out_dir, f"sample_{job_id}.parquet")
        sample_df.to_parquet(sample_path, index=False)
        
        model_path = os.path.join(out_dir, f"trained_model_{job_id}.json")
        linker, _, _ = train_and_save_model(model_path, df, settings_dict, roles)
        jobs[job_id]["progress"] = 70

        preds_path = os.path.join(out_dir, f"splink_predictions_{job_id}.csv")
        clusters_path = os.path.join(out_dir, f"splink_clusters_{job_id}.csv")
        cluster_threshold = 0.99
        try:
            num_rows = len(df)
            if num_rows < 1000: cluster_threshold = 0.85
            elif num_rows < 5000: cluster_threshold = 0.9
        except Exception: pass
        
        df_preds, _ = generate_predictions(linker, df, preds_path, clusters_path, threshold=cluster_threshold)
        jobs[job_id]["progress"] = 85

        id_map = get_deduped_id_mapping(df_preds)
        deduped_out = os.path.join(out_dir, f"deduped_{job_id}.csv")
        deduplicate_by_mapped_ids(df.copy(), "unique_id", id_map, deduped_out)

        jobs[job_id].update({
            "status": "completed", "progress": 100,
            "full_data_path": full_data_path, "sample_path": sample_path,
            "model_path": model_path, "roles_path": roles_path,
            "clusters_path": clusters_path
        })
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 100
    finally:
        if spark:
            spark.stop()


def _check_record_against_clusters(job_id: str, new_record: Dict[str, Any], threshold: float = 0.95) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise ValueError("Job not found or not completed.")

    # Retrieve all necessary artifact paths from the job store
    paths = ["full_data_path", "model_path", "roles_path", "clusters_path"]
    if not all(p in job and os.path.exists(job[p]) for p in paths):
        raise ValueError("One or more required artifacts for checking are missing.")

    # 1. Load all required artifacts
    base_df = pd.read_parquet(job["full_data_path"])
    clusters_df = pd.read_csv(job["clusters_path"], dtype={"unique_id": str})
    with open(job["model_path"], "r", encoding="utf-8") as f:
        settings = json.load(f)
    with open(job["roles_path"], "r", encoding="utf-8") as f:
        roles = json.load(f)

    # 2. Re-initialize the Linker with the FULL dataset and trained model
    db_api = DuckDBAPI()
    linker = Linker(base_df, settings, db_api=db_api)

    # 3. Prepare the new record using the same preprocessing pipeline
    record_copy = dict(new_record)
    record_copy["unique_id"] = "new_record_to_check"
    new_df = pd.DataFrame([record_copy])

    # Ensure all columns the model expects are present, even if null
    for col in base_df.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA

    # Coerce column dtypes in the incoming record to match the training/base dataframe.
    # This prevents silent schema drift (e.g., numbers sent as strings) from hurting match quality.
    try:
        coerced = {}
        for col, dtype in base_df.dtypes.items():
            if col not in new_df.columns:
                continue
            series = new_df[col]
            # Skip coercion for fully null columns
            if series.isna().all():
                coerced[col] = series
                continue
            kind = getattr(dtype, "kind", None)
            if kind in ("i", "u", "f"):
                coerced[col] = pd.to_numeric(series, errors="coerce")
            elif kind == "b":
                # Map common textual booleans
                coerced[col] = series.map(lambda v: True if str(v).strip().lower() in {"true", "1", "t", "yes", "y"}
                                          else False if str(v).strip().lower() in {"false", "0", "f", "no", "n"}
                                          else pd.NA)
            else:
                # Datetime/timedelta
                if str(dtype).startswith("datetime64") or str(dtype).startswith("datetime"):
                    # Try both integer epoch/day counts and ISO strings
                    s = series
                    try:
                        # If looks like integer, try numeric then origin='unix' or '1899-12-30' excel days not assumed here
                        s_num = pd.to_numeric(s, errors="coerce")
                        # If many non-nulls, treat as unix seconds
                        if s_num.notna().any():
                            coerced[col] = pd.to_datetime(s_num, unit="s", errors="coerce")
                        else:
                            coerced[col] = pd.to_datetime(s, errors="coerce")
                    except Exception:
                        coerced[col] = pd.to_datetime(series, errors="coerce")
                else:
                    # Strings/objects: normalize whitespace
                    coerced[col] = series.astype("string").str.strip()
        # Assign back coerced columns
        for c, s in coerced.items():
            new_df[c] = s
    except Exception:
        # Best effort; continue without failing hard
        pass

    # Apply same feature engineering as the original data
    new_df = ensure_first_last_from_name(new_df)
    new_df = ab.ensure_derived_columns_enhanced(new_df, roles)
    new_df = ensure_name_aliases(new_df)

    # 4. Find matches
    matches = linker.inference.find_matches_to_new_records(new_df)
    matches_pd = matches.as_pandas_dataframe()
    # Adaptive match threshold: when the training dataset is small, allow slightly lower threshold
    adaptive_threshold = threshold
    try:
        num_rows = len(base_df)
        if num_rows < 1000 and adaptive_threshold > 0.85:
            adaptive_threshold = 0.85
        elif num_rows < 5000 and adaptive_threshold > 0.9:
            adaptive_threshold = 0.9
    except Exception:
        pass
    strong = matches_pd[matches_pd["match_probability"] >= adaptive_threshold].copy()

    if strong.empty:
        return {"result": "unique"}

    # 5. Identify best match and its cluster
    best = strong.sort_values(by="match_probability", ascending=False).iloc[0]
    # Splink can emit either 'unique_id_l' (pairwise) or 'unique_id' depending on context/version
    best_id = best.get("unique_id_l") if "unique_id_l" in best else best.get("unique_id")
    best_id = str(best_id)
    
    cluster_row = clusters_df[clusters_df["unique_id"] == best_id]
    cluster_id = str(cluster_row["cluster_id"].iloc[0]) if not cluster_row.empty else None

    # --- CHANGE IS HERE: Removed the logic that finds and formats the similar record ---
    # The 'similar_record' key is no longer included in the response.

    # 6. Return minimal duplicate info without cluster representative details
    return {
        "result": "duplicate",
        "cluster_id": cluster_id,
        "match_probability": float(best.get("match_probability", 0.0)),
    }


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
    return send_file(path, as_attachment=True, download_name="clusters.csv")

@app.get("/report/<job_id>")
def download_report(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"ok": False, "error": "job not completed"}), 400

    clusters_path = job.get("clusters_path")
    if not clusters_path or not os.path.exists(clusters_path):
        return jsonify({"ok": False, "error": "clusters file not available"}), 404

    try:
        clusters_df = pd.read_csv(clusters_path, dtype={"unique_id": str, "cluster_id": str})

        # Load roles to decide which fields compose the fingerprint
        roles_path = job.get("roles_path")
        roles = {}
        if roles_path and os.path.exists(roles_path):
            with open(roles_path, "r", encoding="utf-8") as f:
                roles = json.load(f) or {}

        # Build role-based fingerprint using normalized/derived columns when available
        def _pick_cols(df: pd.DataFrame, roles: Dict[str, str]) -> List[str]:
            preferred = []
            # Name fields
            for col in [
                "full_name", "full_name_norm",
                roles.get("full_name", ""),
                "first_name_norm", "last_name_norm",
                roles.get("first_name", ""), roles.get("last_name", "")
            ]:
                if col and col in df.columns:
                    preferred.append(col)
            # Address and geo
            for col in [
                "address_norm", roles.get("address", ""),
                "city_norm", roles.get("city", ""),
                "state_norm", roles.get("state", ""),
                "zip_norm", roles.get("zip", "")
            ]:
                if col and col in df.columns:
                    preferred.append(col)
            # Contact
            for col in ["email_norm", roles.get("email", ""), "phone_digits", roles.get("phone", "")]:
                if col and col in df.columns:
                    preferred.append(col)
            # Fallback to all object/string columns if nothing selected
            if not preferred:
                other = [c for c in df.columns if c not in ["unique_id", "cluster_id"]]
                preferred = df[other].select_dtypes(include=["object", "string"]).columns.tolist()
            # Ensure uniqueness/order preserved
            seen = set()
            ordered = []
            for c in preferred:
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            return ordered

        fp_cols = _pick_cols(clusters_df, roles)
        if not fp_cols:
            # Degenerate case: just group by original cluster
            clusters_df['partition_group'] = 'group_' + (pd.factorize(clusters_df['cluster_id'])[0] + 1).astype(str)
        else:
            # Create normalized fingerprint
            def _norm_series(s: pd.Series) -> pd.Series:
                try:
                    return s.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
                except Exception:
                    return s.astype(str)

            fp_df = clusters_df[fp_cols].fillna("")
            for c in fp_cols:
                fp_df[c] = _norm_series(fp_df[c])
            clusters_df['fingerprint'] = fp_df.agg(' '.join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

            # Representatives per initial cluster
            # Use the longest non-empty fingerprint in each cluster as the representative for stability
            rep_series = clusters_df.loc[clusters_df['fingerprint'].ne('')].sort_values(
                by=['cluster_id', clusters_df['fingerprint'].str.len()], ascending=[True, False]
            ).drop_duplicates(subset=['cluster_id'])
            cluster_representatives = pd.Series(
                rep_series['fingerprint'].values, index=rep_series['cluster_id'].values
            ).to_dict()
            unique_clusters = list(cluster_representatives.keys())

            # DSU for merging similar clusters
            parent = {cid: cid for cid in unique_clusters}
            def find(c_id):
                if parent[c_id] == c_id:
                    return c_id
                parent[c_id] = find(parent[c_id])
                return parent[c_id]
            def union(c1_id, c2_id):
                r1, r2 = find(c1_id), find(c2_id)
                if r1 != r2:
                    parent[r2] = r1

            # Similarity threshold (token_set_ratio ~ 0..100). Use 80 as requested.
            SIM_THRESHOLD = 80

            for i in range(len(unique_clusters)):
                for j in range(i + 1, len(unique_clusters)):
                    c1, c2 = unique_clusters[i], unique_clusters[j]
                    k1, k2 = cluster_representatives.get(c1, ''), cluster_representatives.get(c2, '')
                    if not k1 or not k2:
                        continue
                    if rf_fuzz is not None:
                        score = rf_fuzz.token_set_ratio(k1, k2)
                        similar = score >= SIM_THRESHOLD
                    else:
                        # Fallback to SequenceMatcher ratio 0..1 scaled to 0..100
                        score = SequenceMatcher(None, k1, k2).ratio() * 100.0
                        similar = score >= SIM_THRESHOLD
                    if similar:
                        union(c1, c2)

            # Map each original cluster_id to its partition root
            cluster_to_partition_map = {c: find(c) for c in unique_clusters}
            clusters_df['partition_root'] = clusters_df['cluster_id'].map(cluster_to_partition_map).fillna(clusters_df['cluster_id'])

            # Stable, sequential group labels by sorting roots on their representative fingerprint then by id
            roots = pd.Series(list(set(cluster_to_partition_map.values())))
            root_to_rep = {r: cluster_representatives.get(r, '') for r in roots}
            ordered_roots = sorted(roots, key=lambda r: (root_to_rep.get(r, ''), str(r)))
            root_to_groupnum = {r: i + 1 for i, r in enumerate(ordered_roots)}
            clusters_df['partition_group'] = 'group_' + clusters_df['partition_root'].map(root_to_groupnum).astype(int).astype(str)
            clusters_df = clusters_df.drop(columns=['fingerprint', 'partition_root'], errors='ignore')

        # Reorder columns to place 'partition_group' right after 'cluster_id'
        cols = clusters_df.columns.tolist()
        if 'partition_group' in cols and 'cluster_id' in cols:
            cols.remove('partition_group')
            cols.insert(cols.index('cluster_id') + 1, 'partition_group')
            clusters_df = clusters_df[cols]
        
        # Sort the entire dataframe by the new partition group to group similar records together
        clusters_df = clusters_df.sort_values(by=['partition_group', 'cluster_id', 'unique_id'])

        # Save and send the final report
        report_path = os.path.join(os.path.dirname(__file__), "outputs", f"reports_{job_id}.csv")
        clusters_df.to_csv(report_path, index=False)
        return send_file(report_path, as_attachment=True, download_name="reports.csv")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/profile/<job_id>")
def get_profile(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    profile_path = os.path.join(out_dir, f"value_distribution_{job_id}.html")
    
    # If pre-generated file exists, return it
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as fh:
            return jsonify({"ok": True, "html": fh.read()})
    
    # Else, generate on-demand from saved sample
    sample_path = job.get("sample_path")
    if not sample_path or not os.path.exists(sample_path):
        return jsonify({"ok": False, "error": "profile sample not available"}), 404
        
    try:
        sample_df = pd.read_parquet(sample_path)
        db_api = DuckDBAPI()
        figs = profile_columns(sample_df, db_api=db_api)
        
        def _fig_to_html(fig):
            try: return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
            except Exception:
                try: return fig.to_html() if alt else ""
                except Exception: return ""
        
        html = "\n".join(_fig_to_html(f) for f in figs) if isinstance(figs, (list, tuple)) else _fig_to_html(figs)
        
        # Cache it for subsequent requests
        with open(profile_path, 'w', encoding='utf-8') as fh:
            fh.write(html)
        job["profile_path"] = profile_path
        return jsonify({"ok": True, "html": html})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/profile_html/<job_id>")
def get_profile_html(job_id: str):
    """Return the profile HTML directly for iframe embedding."""
    job = jobs.get(job_id)
    if not job:
        return Response("Job not found", status=404)
        
    # Reuse JSON endpoint logic to generate if it doesn't exist
    profile_response = get_profile(job_id)
    if profile_response.status_code != 200:
        error_data = profile_response.get_json()
        return Response(error_data.get("error", "Could not generate profile"), status=profile_response.status_code)

    html = profile_response.get_json().get("html", "")
    return Response(html, status=200, mimetype='text/html')


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)