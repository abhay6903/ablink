from __future__ import annotations
import re
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures

from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import (
#     col, lower, upper, trim, regexp_replace, substring, split,
#     element_at, expr, when, length, sha2, lit, concat,
# )
from pyspark.sql.functions import (
    col, lower, upper, trim, regexp_replace, substring, split,
    element_at, expr, when, length, sha2, lit, concat, regexp_extract
)
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.functions import udf

# Gracefully import unidecode for accent normalization
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

# Splink imports
import splink.comparison_library as cl
from splink import SettingsCreator, block_on, SparkAPI
from splink.blocking_analysis import count_comparisons_from_blocking_rule

# --- UDFs for Spark ---
# --- UDFs for Spark ---
# --- MODIFICATION START: Added robust error handling to prevent crashes ---
if unidecode:
    @udf(StringType())
    def accent_remove_udf(s):
        try:
            # Handle nulls or non-string types gracefully
            if s is None or not isinstance(s, str):
                return None
            return unidecode(s)
        except Exception:
            return None # Return None if any unexpected error occurs
else:
    # Fallback if unidecode is not available
    @udf(StringType())
    def accent_remove_udf(s):
        return s

@udf(StringType())
def double_metaphone_udf(s):
    from metaphone import doublemetaphone
    try:
        if s is None or not isinstance(s, str):
            return "" # Metaphone should return an empty string for non-strings
        return doublemetaphone(s)[0]
    except Exception:
        return "" # Return a safe empty string on any error

@udf(StringType())
def hashlib_sha256_udf(s):
    try:
        if s is None:
            return None
        # Safely convert any type to a string before encoding
        return hashlib.sha256(str(s).encode('utf-8')).hexdigest()
    except Exception:
        return None # Return None on any error
# --- MODIFICATION END ---

# --- helper functions ---

def _nonnull_share(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    """Calculate the share of non-null values in a Series."""
    if total_rows == 0:
        return 0.0
    return df.filter(col(col_name).isNotNull()).count() / total_rows

def _cardinality_ratio(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    """Calculate the ratio of unique values to total values in a Series."""
    if total_rows == 0:
        return 0.0
    unique_count = df.select(col_name).distinct().count()
    return unique_count / total_rows

# --- role inference / derived columns / rule generation ---

def create_enhanced_column_mapper() -> Dict[str, List[str]]:
    """Expanded dictionary mapping roles to common column name aliases."""
    return {
        "first_name": ["first_name", "firstname", "given_name", "fname", "forename"],
        "last_name": ["last_name", "lastname", "surname", "lname", "family_name"],
        "full_name": ["full_name", "customer_name", "name", "person_name", "complete_name"],
        "email": ["email", "email_address", "e_mail", "contact_email"],
        "phone": ["phone", "mobile", "contact_no", "phone_number", "telephone"],
        "zip": ["zip", "zipcode", "postal_code", "pincode", "postcode"],
        "city": ["city", "town", "municipality"],
        "state": ["state", "province", "region", "county"],
        "address": ["address", "street", "street_address", "addr", "location"],
        "date": ["date", "dob", "birth_date", "order_date", "timestamp", "created_at"],
        "url": ["url", "website", "web_address"],
        "geo_lat": ["lat", "latitude", "geo_lat"],
        "geo_lon": ["lon", "long", "longitude", "geo_lon"],
        "currency": ["price", "amount", "cost", "revenue", "salary"],
        "numeric_id": ["id", "key", "account_number", "record_id", "customer_id", "user_id", "person_id", "orderkey"],
    }

def create_semantic_patterns() -> Dict[str, List[str]]:
    """Expanded dictionary of regex patterns to infer column roles."""
    return {
        "email": [r".*mail.*", r".*@.*"],
        "phone": [r".*(phone|mobile|tel|contact).*"],
        "zip": [r".*(zip|postal|pin).*code.*"],
        "date": [r".*date.*", r".*_at$", r".*timestamp.*", r"^\d{4}[/-]\d{2}[/-]\d{2}"], # Added date regex for YYYY-MM-DD
        "url": [r".*(url|website|http|www).*"],
        "geo_lat": [r".*lat(itude)?.*"],
        "geo_lon": [r".*lon(gitude)?.*"],
        "numeric_id": [r".*(id|key|number|no)$"],
        "name": [r".*name.*"], # Catch-all for names if more specific first/last not found
        "address": [r".*(addr|address|street).*"],
        "geo": [r".*(city|state|province).*"]
    }

def infer_roles_enhanced(input_df: SparkDataFrame) -> Dict[str, str]:
    """
    Infer column roles using a multi-pass approach: direct aliasing, regex patterns,
    and data type/distribution analysis.
    """
    if input_df.isEmpty():
        return {}

    total_rows = input_df.count()
    if total_rows == 0:
        return {}
        
    lowercase_to_original = {c.lower(): c for c in input_df.columns}
    roles: Dict[str, str] = {}
    
    # Take a sample for efficient data-based inference
    sample_df_count = min(total_rows, 100000)
    sample_df = input_df.sample(False, float(sample_df_count) / total_rows, seed=42) if total_rows > 100000 else input_df

    # Pass 1: Direct alias matching
    for role, aliases in create_enhanced_column_mapper().items():
        if role in roles: continue
        for alias in aliases:
            if alias in lowercase_to_original:
                col_original = lowercase_to_original[alias]
                # Avoid assigning 'unique_id' as a general role if it's the actual unique ID column
                if "unique_id" not in col_original.lower() and col_original not in roles.values():
                    roles[role] = col_original
                    break

    # Pass 2: Pattern-based matching (using column names)
    for role, patterns in create_semantic_patterns().items():
        if role in roles: continue
        for pattern in patterns:
            for col_lower, col_original in lowercase_to_original.items():
                if re.search(pattern, col_lower, re.IGNORECASE) and \
                   "unique_id" not in col_lower and \
                   col_original not in roles.values():
                    roles[role] = col_original
                    break
            if role in roles: break

    # Pass 3: Data type and distribution-based inference
    # This pass is more expensive as it requires data scans.
    # It's better suited for remaining unassigned columns.
    for col_name in input_df.columns: # Use input_df for accurate counts
        if col_name in roles.values() or "unique_id" in col_name.lower():
            continue
        
        # Check non-null share before expensive checks
        non_null_share = _nonnull_share(input_df, col_name, total_rows)
        if non_null_share < 0.1: # Skip columns with too many nulls
            continue

        # Infer dtype from Spark's schema, and potentially refine with sample data
        spark_dtype = input_df.schema[col_name].dataType
        
        # Numeric ID inference
        if str(spark_dtype).startswith(("Integer", "Long", "Float", "Double")) and "numeric_id" not in roles:
            card_ratio = _cardinality_ratio(sample_df, col_name, sample_df_count)
            # High cardinality + high non-null implies it could be an ID
            if card_ratio >= 0.85 and non_null_share >= 0.9:
                roles["numeric_id"] = col_name
        
        # Date inference (if not already found by name/pattern)
        elif str(spark_dtype).startswith("TimestampType") and "date" not in roles:
            roles["date"] = col_name
        
        # String/Text inference
        elif str(spark_dtype).startswith("StringType"):
            card_ratio = _cardinality_ratio(sample_df, col_name, sample_df_count)
            if card_ratio < 0.1 and card_ratio > 0 and "category_enum" not in roles and non_null_share > 0.2:
                roles["category_enum"] = col_name
            elif card_ratio > 0.5 and "text_freeform" not in roles and non_null_share > 0.2:
                roles["text_freeform"] = col_name
                
    return roles

def ensure_derived_columns_enhanced(input_df: SparkDataFrame, roles: Dict[str, str]) -> SparkDataFrame:
    """
    Safely create derived columns for normalization, phonetic matching, hashing, etc.
    """
    df_transformed = input_df.repartition(8)

    total_rows = input_df.count()
    if total_rows == 0:
        return df_transformed # Return empty if input is empty

    # Split full_name into first_name and last_name if full_name is present
    if "full_name" in roles and roles["full_name"] in input_df.columns:
        full_name_col = roles["full_name"]
        if _nonnull_share(input_df, full_name_col, total_rows) > 0.2:
            df_transformed = df_transformed.withColumn(
                "temp_name_parts", split(col(full_name_col).cast("string"), " ", 2)
            )
            if "first_name" not in roles:
                df_transformed = df_transformed.withColumn(
                    "first_name_derived", element_at(col("temp_name_parts"), 1)
                )
                roles["first_name"] = "first_name_derived"
            if "last_name" not in roles:
                df_transformed = df_transformed.withColumn(
                    "last_name_derived", element_at(col("temp_name_parts"), 2)
                )
                roles["last_name"] = "last_name_derived"
            df_transformed = df_transformed.drop("temp_name_parts")

    # Name normalization and metaphone
    for name_type in ["first_name", "last_name"]:
        if name_type in roles and roles[name_type] in df_transformed.columns:
            base_col = roles[name_type]
            if _nonnull_share(df_transformed, base_col, total_rows) > 0.2:
                try:
                    df_transformed = df_transformed.withColumn(
                        f"{name_type}_norm",
                        lower(trim(accent_remove_udf(col(base_col).cast("string"))))
                    )
                    df_transformed = df_transformed.withColumn(
                        f"{name_type}_metaphone",
                        double_metaphone_udf(col(f"{name_type}_norm"))
                    )
                except Exception as e:
                    print(f"Warning: Could not create derived name columns for '{name_type}': {e}")

    # Other roles' derived columns
    for role, col_name in roles.items():
        if col_name not in df_transformed.columns or _nonnull_share(df_transformed, col_name, total_rows) <= 0.2:
            continue
        
        try:
            if role == "email":
                df_transformed = df_transformed.withColumn(
                    "email_norm", lower(trim(col(col_name).cast("string")))
                )
            elif role == "phone":
                df_transformed = df_transformed.withColumn(
                    "phone_digits", regexp_replace(col(col_name).cast("string"), r"\D", "")
                )
            elif role == "zip":
                df_transformed = df_transformed.withColumn(
                    "zip_norm", upper(regexp_replace(col(col_name).cast("string"), r"\s", ""))
                )
            elif role in ("city", "state", "address"):
                df_transformed = df_transformed.withColumn(
                    f"{role}_norm", lower(trim(accent_remove_udf(col(col_name).cast("string"))))
                )
            elif role == "date":
                # Attempt to parse to date and format. Coalesce handles mixed types.
                # Spark's to_date function is more robust with multiple formats
                df_transformed = df_transformed.withColumn(
                    "date_norm",
                    when(col(col_name).isNotNull(),
                         expr(f"date_format(to_date(CAST(`{col_name}` AS STRING), 'yyyy-MM-dd'), 'yyyy-MM-dd')"))
                    .otherwise(None)
                )
            elif role == "url":
                # Extract domain from URL
                df_transformed = df_transformed.withColumn(
                    "url_domain",
                    lower(regexp_extract(col(col_name).cast("string"), r'https?://(?:www\.)?([^/]+)', 1))
                )
            elif role == "numeric_id":
                # Create a hash of the numeric ID
                df_transformed = df_transformed.withColumn(
                    f"{col_name}_hash",
                    hashlib_sha256_udf(col(col_name).cast("string"))
                    # Can also use Spark's built-in sha2 for hexadecimal output
                    # sha2(col(col_name).cast("string"), 256)
                )
        except Exception as e:
            print(f"Warning: Could not create derived column for role '{role}' on column '{col_name}': {e}")
            
    # First character of first name
    if "first_name" in roles and roles["first_name"] in df_transformed.columns:
        if _nonnull_share(df_transformed, roles["first_name"], total_rows) > 0.2:
            df_transformed = df_transformed.withColumn(
                "first_name_first_char",
                substring(col(roles["first_name"]).cast("string"), 1, 1).alias("first_name_first_char").cast(StringType())
            )

    df_transformed.cache() # Checkpoint for performance
    return df_transformed
def generate_robust_blocking_rules(df: SparkDataFrame, roles: Dict[str, str]) -> List[Tuple[str, object]]:
    """
    Generate a diverse list of candidate blocking rules based on available columns.
    """
    rules = []
    total_rows = df.count() # Get total rows once

    def add_rule_if_valid(name, *cols):
        min_non_null_share = 0.3
        # Check if columns exist AND have sufficient non-null values
        if all(c in df.columns and _nonnull_share(df, c, total_rows) > min_non_null_share for c in cols):
            rules.append((name, block_on(*cols)))

    if "numeric_id" in roles:
         add_rule_if_valid(f"exact_{roles['numeric_id']}", roles['numeric_id'])
         # Modulo blocking for numeric IDs
         numeric_id_col = roles['numeric_id']
         if numeric_id_col in df.columns and str(df.schema[numeric_id_col].dataType).startswith(("Integer", "Long")):
            df = df.withColumn(f"{numeric_id_col}_mod_1000", col(numeric_id_col) % 1000)
            add_rule_if_valid(f"{numeric_id_col}_mod_1000", f"{numeric_id_col}_mod_1000")
         
         # Hash of numeric ID
         numeric_id_hash_col = f"{roles['numeric_id']}_hash"
         if numeric_id_hash_col in df.columns:
             add_rule_if_valid(f"hash_{roles['numeric_id']}", numeric_id_hash_col)

    add_rule_if_valid("exact_email", "email_norm")
    add_rule_if_valid("exact_phone", "phone_digits")
    add_rule_if_valid("exact_zip", "zip_norm")
    add_rule_if_valid("exact_date", "date_norm") # Use normalized date
    
    add_rule_if_valid("exact_first_last_name", "first_name_norm", "last_name_norm")
    add_rule_if_valid("metaphone_full_name", "first_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("exact_last_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("zip_lastname_meta", "zip_norm", "last_name_metaphone")
    add_rule_if_valid("city_firstname", "city_norm", "first_name_norm")
    add_rule_if_valid("first_name_first_char", "first_name_first_char")
    
    if "category_enum" in roles:
        add_rule_if_valid(f"exact_{roles['category_enum']}", roles['category_enum'])
    
    if "text_freeform" in roles:
        # For free text, we could hash or use first few characters if appropriate
        text_col = roles['text_freeform']
        if text_col in df.columns:
            # Hash of the text_freeform content
            df = df.withColumn(f"{text_col}_hash", hashlib_sha256_udf(col(text_col).cast("string")))
            add_rule_if_valid(f"hash_{text_col}", f"{text_col}_hash")

    if "url" in roles:
        add_rule_if_valid("url_domain_exact", "url_domain")

    return rules


def build_settings_enhanced(df: SparkDataFrame, roles: Dict[str, str], blocking_rules: List[object]) -> SettingsCreator:
    comparisons = []
    total_rows = df.count()

    def add_comparison_if_valid(comparison_func, col_name: str, *args, **kwargs):
        min_non_null_share = 0.2
        if col_name in df.columns and _nonnull_share(df, col_name, total_rows) > min_non_null_share:
            comparisons.append(comparison_func(col_name, *args, **kwargs))
        else:
            print(f"Skipping comparison for '{col_name}' due to missing column or too many nulls.")

    if "first_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "first_name_norm", [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "first_name_metaphone")
    if "last_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "last_name_norm", [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "last_name_metaphone")
    if "email" in roles:
        add_comparison_if_valid(cl.EmailComparison, "email_norm")
    if "phone" in roles:
        add_comparison_if_valid(cl.LevenshteinAtThresholds, "phone_digits", [2, 4])
    if "address" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "address_norm", [0.7, 0.9])
    if "city" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "city_norm", [0.8, 0.95])  # Removed term_frequency_adjustments
    if "state" in roles:
        add_comparison_if_valid(cl.ExactMatch, "state_norm")
    if "zip" in roles:
        add_comparison_if_valid(cl.PostcodeComparison, "zip_norm")
    if "date" in roles:
        add_comparison_if_valid(cl.DateOfBirthComparison, "date_norm", input_is_string=True)
    if "full_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["full_name"], [0.8, 0.95])
    if "text_freeform" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["text_freeform"], [0.8, 0.95])
    if "url" in roles:
        add_comparison_if_valid(cl.ExactMatch, "url_domain")
    if not comparisons:
        if "numeric_id" in roles:
            numeric_id_hash_col = f"{roles['numeric_id']}_hash"
            if numeric_id_hash_col in df.columns and _nonnull_share(df, numeric_id_hash_col, total_rows) > 0.2:
                add_comparison_if_valid(cl.ExactMatch, numeric_id_hash_col)
        if not comparisons:
            raise ValueError(
                "Splink model cannot be trained because no valid comparison columns were generated. "
                f"The table may lack suitable fields for deduplication. Detected roles: {list(roles.keys())}"
            )

    return SettingsCreator(
        link_type="dedupe_only",
        em_convergence=0.001,
        max_iterations=25,
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=False,
    )

def select_optimal_blocking_rules(
    df: SparkDataFrame,
    candidate_rules: List[Tuple[str, object]],
    db_api: SparkAPI,
    max_rules: Optional[int] = 5,
    max_comparisons: int = 20_000_000,
) -> Tuple[List[Tuple[str, object]], List[Dict]]:
    """
    Selects an optimal set of blocking rules by evaluating the number of comparisons in parallel.
    Selects rules from the middle layer of comparison counts, excluding those with too few or too many pairs.
    """
    diagnostics = []
    scored = []
    min_comparisons = 100  # Minimum comparisons to avoid overly restrictive rules
    ideal_comparisons = 1_000_000  # Target for "middle layer" rules

    def compute_count(name, rule):
        """Helper function to compute comparison count for a rule."""
        try:
            result = count_comparisons_from_blocking_rule(
                table_or_tables=df, blocking_rule=rule, link_type="dedupe_only", db_api=db_api
            )
            cnt = int(result.get("number_of_comparisons_to_be_scored_post_filter_conditions", float('inf')))
            return name, rule, cnt
        except Exception as e:
            return name, rule, -1

    # Parallel evaluation of blocking rules
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_count, name, rule) for name, rule in candidate_rules]
        for future in concurrent.futures.as_completed(futures):
            name, rule, cnt = future.result()
            reason = "not_evaluated"
            if cnt == float('inf'):
                reason = "not_evaluated"
            elif cnt == 0:
                reason = "zero_comparisons"
            elif cnt < min_comparisons:
                reason = "too_few_comparisons"
            elif cnt > max_comparisons:
                reason = "too_many_comparisons"
            else:
                reason = "selected"
            if reason == "selected":
                scored.append((name, rule, cnt))
                diagnostics.append({"name": name, "comparisons": cnt, "kept": True, "reason": reason, "rule": rule})
            else:
                diagnostics.append({"name": name, "comparisons": (f"{cnt:,}" if isinstance(cnt, int) and cnt >= 0 else str(cnt)), "kept": False, "reason": reason, "rule": rule})

    # Sort by comparison count and select middle layer rules
    scored.sort(key=lambda x: x[2])  # Sort by number of comparisons
    if len(scored) > max_rules:
        # Select middle max_rules rules
        start = (len(scored) - max_rules) // 2
        scored = scored[start:start + max_rules]
    selected_rules_with_names = [(name, rule) for name, rule, cnt in scored]

    # Fallback: Select rule closest to ideal_comparisons if no rules are selected
    if not selected_rules_with_names and candidate_rules:
        fallback_rule_info = None
        min_diff = float('inf')
        for diag in diagnostics:
            if isinstance(diag['comparisons'], int) and diag['comparisons'] > 0:
                diff = abs(diag['comparisons'] - ideal_comparisons)
                if diff < min_diff:
                    fallback_rule_info = (diag['name'], diag['rule'], diag['comparisons'])
                    min_diff = diff
        if fallback_rule_info:
            selected_rules_with_names.append((fallback_rule_info[0], fallback_rule_info[1]))
            for diag in diagnostics:
                if diag['name'] == fallback_rule_info[0]:
                    diag['kept'] = True
                    diag['reason'] = "fallback_closest_to_ideal"
                    break
        else:
            # Ultimate fallback: Use first candidate rule
            selected_rules_with_names.append(candidate_rules[0])
            for diag in diagnostics:
                if diag['name'] == candidate_rules[0][0]:
                    diag['kept'] = True
                    diag['reason'] = "fallback_first_candidate"
                    break
            else:
                diagnostics.append({"name": candidate_rules[0][0], "comparisons": "unknown", "kept": True, "reason": "fallback_first_candidate", "rule": candidate_rules[0][1]})

    return selected_rules_with_names, diagnostics

def auto_generate_settings(df: SparkDataFrame, db_api: SparkAPI, max_rules: Optional[int] = 5, universal_mode: bool = True, spark=None) -> Tuple[Dict, Dict, List, SparkDataFrame]:
    """
    Main orchestration function to automatically generate Splink settings from a DataFrame.
    Compatible with Splink using SparkAPI, optimized for laptop execution.
    Requires an existing SparkSession to be passed as 'spark' if not in global scope.
    """
    if df.isEmpty() or df.count() < 2:
        raise ValueError("Input DataFrame must have at least two rows to perform deduplication.")

    if spark is None:
        raise ValueError("SparkSession 'spark' must be provided as an argument or defined in the global scope.")

    # Repartition to create smaller data chunks, reducing memory pressure on each worker.
    # Increased from 8 to 16 for better load distribution.
    df = df.repartition(16)
    spark.conf.set("spark.sql.shuffle.partitions", 16)  # Align with parallelism

    # --- KEY CHANGES FOR STABILITY ---
    # 1. Give the Python worker processes more memory. This is the most critical fix.
    spark.sparkContext._conf.set("spark.python.worker.memory", "2g")
    # 2. Increase the max result size for the driver, preventing errors on actions like .collect() or .count().
    spark.sparkContext._conf.set("spark.driver.maxResultSize", "2g")
    # 3. Existing executor settings are good for the JVM part.
    spark.sparkContext._conf.set("spark.executor.memory", "4g")
    spark.sparkContext._conf.set("spark.executor.cores", "2")
    spark.sparkContext._conf.set("spark.executor.instances", "4")


    total_rows = df.count()
    # Smaller sample for faster inference (reduced from 100k)
    inference_df_count = min(total_rows, 10000)
    inference_df = df.sample(False, float(inference_df_count) / total_rows, seed=42).repartition(8)
    inference_df.cache()  # Cache sample for reuse
    roles = infer_roles_enhanced(inference_df)
    inference_df.unpersist()  # Free memory
    print(f"✅ Detected roles: {roles}")
    
    df_enhanced = ensure_derived_columns_enhanced(df, roles)
    
    candidate_rules = generate_robust_blocking_rules(df_enhanced, roles)
    print(f"\n⚙️  Generated {len(candidate_rules)} candidate blocking rules.")
    
    selected_rules, diagnostics = select_optimal_blocking_rules(
        df_enhanced, candidate_rules, db_api, max_rules=max_rules
    )
    
    print("\n📊 Blocking rule analysis:")
    for diag in diagnostics:
        status = "✓" if diag["kept"] else "✗"
        reason_detail = f" ({diag['reason']})" if not diag["kept"] else ""
        comps = f"{diag['comparisons']:,}" if isinstance(diag['comparisons'], int) and diag['comparisons'] >= 0 else diag['comparisons']
        print(f"{status} {diag['name']}: {comps} comparisons{reason_detail}")

    # The original df_enhanced.unpersist() was here. It's better to unpersist after it's truly no longer needed.
    # Since it's returned by the function, the caller should be responsible for unpersisting.
    # df_enhanced.unpersist()
    
    final_blocking_rules = [rule for _, rule in selected_rules]
    settings_creator = build_settings_enhanced(df_enhanced, roles, final_blocking_rules)
    print("\n✅ Successfully generated complete Splink settings.")
    
    settings_dict = None
    if isinstance(settings_creator, dict):
        settings_dict = settings_creator
    elif hasattr(settings_creator, "as_dict") and callable(settings_creator.as_dict):
        settings_dict = settings_creator.as_dict()
    elif hasattr(settings_creator, "settings_dict"):
        settings_dict = settings_creator.settings_dict
    elif hasattr(settings_creator, "to_dict") and callable(settings_creator.to_dict):
        settings_dict = settings_creator.to_dict()
    elif hasattr(settings_creator, "create_settings_dict") and callable(settings_creator.create_settings_dict):
        settings_dict = settings_creator.create_settings_dict(sql_dialect_str="spark")  # Required for SparkAPI
    else:
        raise TypeError(
            f"Could not convert Splink SettingsCreator object ({type(settings_creator)}) "
            f"to a dictionary. This is likely due to an incompatible Splink library version. "
            f"Available attributes: {dir(settings_creator)}"
        )

    return settings_dict, roles, diagnostics, df_enhanced

class BlockingBotSpark:
    def __init__(self, db_api: SparkAPI):
        """
        Initializes the BlockingBotSpark with a SparkAPI object.

        Args:
            db_api: An initialized Splink SparkAPI object.
        """
        self.db_api = db_api

    def auto_generate_settings(self, df: SparkDataFrame, max_rules: Optional[int] = 5) -> Tuple[Dict, Dict, List, SparkDataFrame]:
        """
        Orchestrates the automatic generation of Splink settings for Spark DataFrames.

        Args:
            df: The input Spark DataFrame.
            max_rules: The maximum number of blocking rules to select.

        Returns:
            A tuple containing:
            - A dictionary representing the Splink settings.
            - A dictionary of inferred column roles.
            - A list of diagnostics for blocking rule selection.
            - The enhanced Spark DataFrame with derived columns.
        """
        return auto_generate_settings(df, self.db_api, max_rules=max_rules)