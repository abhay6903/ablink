# auto_blocking.py
from __future__ import annotations
import re
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd

# Splink imports
from splink import block_on, SettingsCreator
import splink.comparison_library as cl
from splink.blocking_analysis import count_comparisons_from_blocking_rule

# Gracefully import metaphone and unidecode for normalization
from metaphone import doublemetaphone
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

# For Spark support
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, when, split, element_at, upper, lit, size,
    coalesce, to_date, date_format, regexp_extract, sha2, substring, conv,
    monotonically_increasing_id
)
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

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

# ----------------------- Small helpers -----------------------
def _nonnull_share(series: pd.Series) -> float:
    total = len(series)
    return 0.0 if total == 0 else series.notna().sum() / total

def _cardinality_ratio(series: pd.Series) -> float:
    total = len(series)
    return 0.0 if total == 0 else series.nunique(dropna=True) / total


# ---------------------------------------------------------
# Role inference helpers
# ---------------------------------------------------------
ROLE_ALIASES = {
    "first_name": ["fname", "first", "forename"],
    "last_name": ["lname", "surname", "family_name"],
    "full_name": ["fullname", "name"],
    "email": ["email_address", "mail"],
    "phone": ["phone_number", "mobile", "telephone"],
    "address": ["addr", "street"],
    "city": ["town", "locality"],
    "state": ["province", "region"],
    "zip": ["zipcode", "postal", "postcode"],
    "date": ["dob", "birthdate", "date_of_birth"],
    "numeric_id": ["id", "key", "account_number", "record_id", "customer_id", "user_id", "person_id", "orderkey"],
}


def infer_roles(columns: List[str]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    lowercase_to_original = {c.lower(): c for c in columns}
    
    for role, aliases in ROLE_ALIASES.items():
        if role in roles:
            continue
        # Check exact matches first
        if role in lowercase_to_original:
            if role == "numeric_id" and lowercase_to_original[role].lower() == "unique_id":
                continue 
            roles[role] = lowercase_to_original[role]
            continue
        # Check aliases
        for alias in aliases:
            if alias in lowercase_to_original:
                if role == "numeric_id" and lowercase_to_original[alias].lower() == "unique_id":
                    continue
                roles[role] = lowercase_to_original[alias]
                break
    
    # Additional inference for generic 'id' or 'key'
    if "numeric_id" not in roles:
        for col_lower, col_original in lowercase_to_original.items():
            if ("id" in col_lower or "key" in col_lower) and "unique_id" not in col_lower:
                roles["numeric_id"] = col_original
                break
    
    return roles


# ---------------------------------------------------------
# Derived column generation
# ---------------------------------------------------------
def ensure_derived_columns_enhanced_spark(df: SparkDataFrame, roles: Dict[str, str]) -> SparkDataFrame:
    """
    Ensure normalized/derived versions of important columns are present (Spark version).
    """
    for role, colname in roles.items():
        if colname not in df.columns:
            continue
        if role in ("first_name", "last_name", "full_name", "address", "city", "state"):
            df = df.withColumn(f"{colname}_norm", lower(trim(coalesce(col(colname), lit("")))))
        elif role == "phone":
            df = df.withColumn(f"{colname}_digits", regexp_replace(coalesce(col(colname), lit("")), "[^0-9]", ""))
        elif role == "email":
            df = df.withColumn(f"{colname}_norm", lower(trim(coalesce(col(colname), lit("")))))
        elif role == "zip":
            df = df.withColumn(f"{colname}_norm", upper(regexp_replace(coalesce(col(colname), lit("")), r"\s", "")))
        elif role == "date":
            df = df.withColumn(f"{colname}_norm", date_format(to_date(coalesce(col(colname), lit(""))), "yyyy-MM-dd"))
        elif role == "numeric_id":
            df = df.withColumn(f"{colname}_hash", sha2(coalesce(col(colname), lit("")), 256))
    
    if "first_name" in roles and roles["first_name"] in df.columns:
        first_name_col = roles["first_name"]
        df = df.withColumn("first_name_first_char", upper(substring(coalesce(col(first_name_col), lit("")), 1, 1)))

    return df


# ---------------------------------------------------------
# Blocking rule generation
# ---------------------------------------------------------
def generate_candidate_rules(roles: Dict[str, str]) -> List[Tuple[str, object]]:
    rules = []
    
    if "email" in roles:
        rules.append((f"exact_{roles['email']}_norm", block_on(f"{roles['email']}_norm")))
    if "phone" in roles:
        rules.append((f"exact_{roles['phone']}_digits", block_on(f"{roles['phone']}_digits")))
    if "zip" in roles:
        rules.append((f"exact_{roles['zip']}_norm", block_on(f"{roles['zip']}_norm")))
    if "date" in roles:
        rules.append((f"exact_{roles['date']}_norm", block_on(f"{roles['date']}_norm")))

    if "first_name" in roles and "last_name" in roles:
        rules.append(("exact_first_last_name", block_on(f"{roles['first_name']}_norm", f"{roles['last_name']}_norm")))
    
    if "zip" in roles and "last_name" in roles:
        rules.append(("zip_lastname_norm", block_on(f"{roles['zip']}_norm", f"{roles['last_name']}_norm")))
    if "city" in roles and "first_name" in roles:
        rules.append(("city_firstname_norm", block_on(f"{roles['city']}_norm", f"{roles['first_name']}_norm")))
        
    if "first_name" in roles:
        rules.append(("first_name_first_char", block_on("first_name_first_char")))

    if "numeric_id" in roles:
        rules.append((f"exact_{roles['numeric_id']}", block_on(roles['numeric_id'])))

    if not rules:
        print("Warning: No suitable blocking rules generated. Attempting to create a very broad fallback rule.")
        if "numeric_id" in roles:
            rules.append(("fallback_numeric_id", block_on(roles['numeric_id'])))
        else:
            rules.append(("very_loose_fallback_all_pairs", block_on("1=1"))) 

    return rules


# ---------------------------------------------------------
# Build Splink Settings
# ---------------------------------------------------------
def build_settings_enhanced(df: pd.DataFrame, roles: Dict[str, str], blocking_rules: List[object]) -> SettingsCreator:
    comparisons = []

    def add_comparison_if_valid(comparison_func, col_key: str, derived_suffix: str = "", arg=None, **kwargs):
        base_col = roles.get(col_key)
        if not base_col:
            return

        final_col_name = f"{base_col}{derived_suffix}" if derived_suffix else base_col
        if final_col_name not in df.columns or _nonnull_share(df[final_col_name]) <= 0:
            return

        # Handle functions that take thresholds (list) vs exact match
        if arg is not None:
            comp = comparison_func(final_col_name, arg, **kwargs)
        else:
            comp = comparison_func(final_col_name, **kwargs)

        # Validate comparison levels
        levels = getattr(comp, "comparison_levels", []) or []
        for lvl in levels:
            if not isinstance(lvl, dict):
                print(f"Warning: Invalid comparison level in {final_col_name}: {lvl} (type: {type(lvl)}). Skipping.")
                return

        comparisons.append(comp)

    # Name comparisons
    add_comparison_if_valid(cl.JaroWinklerAtThresholds, "first_name", "_norm", [0.7, 0.9])
    add_comparison_if_valid(cl.JaroWinklerAtThresholds, "last_name", "_norm", [0.7, 0.9])

    # Email comparisons
    add_comparison_if_valid(cl.ExactMatch, "email", "_norm")
    add_comparison_if_valid(cl.EmailComparison, "email", "_norm")

    # Other attribute comparisons
    add_comparison_if_valid(cl.LevenshteinAtThresholds, "phone", "_digits", [2, 4])
    add_comparison_if_valid(cl.JaroWinklerAtThresholds, "address", "_norm", [0.7, 0.9])
    add_comparison_if_valid(cl.JaroWinklerAtThresholds, "city", "_norm", [0.8, 0.95])
    add_comparison_if_valid(cl.ExactMatch, "state", "_norm")
    add_comparison_if_valid(cl.PostcodeComparison, "zip", "_norm")

    # Custom date comparison for Spark
    if "date" in roles:
        date_col = f"{roles['date']}_norm"
        if date_col in df.columns and _nonnull_share(df[date_col]) > 0:
            comparison_levels = [
                {
                    "sql_condition": f"{date_col}_l IS NULL OR {date_col}_r IS NULL",
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": f"{date_col}_l = {date_col}_r",
                    "label_for_charts": "Exact match",
                },
                {
                    "sql_condition": f"ABS(unix_timestamp(to_date({date_col}_l, 'yyyy-MM-dd')) - unix_timestamp(to_date({date_col}_r, 'yyyy-MM-dd'))) <= 2629800.0",
                    "label_for_charts": "Within 1 month",
                },
                {
                    "sql_condition": f"ABS(unix_timestamp(to_date({date_col}_l, 'yyyy-MM-dd')) - unix_timestamp(to_date({date_col}_r, 'yyyy-MM-dd'))) <= 31557600.0",
                    "label_for_charts": "Within 1 year",
                },
                {
                    "sql_condition": f"ABS(unix_timestamp(to_date({date_col}_l, 'yyyy-MM-dd')) - unix_timestamp(to_date({date_col}_r, 'yyyy-MM-dd'))) <= 315576000.0",
                    "label_for_charts": "Within 10 years",
                },
                {
                    "sql_condition": "ELSE",
                    "label_for_charts": "No match",
                },
            ]
            comparisons.append(cl.CustomComparison(date_col, comparison_levels))

    # Full name comparison
    add_comparison_if_valid(cl.JaroWinklerAtThresholds, "full_name", "_norm", [0.8, 0.95])

    # Numeric ID handling
    if "numeric_id" in roles:
        raw_col = roles["numeric_id"]
        hash_col = f"{raw_col}_hash"
        added = any(getattr(comp, "col_name", None) == raw_col for comp in comparisons)
        if raw_col in df.columns and _nonnull_share(df[raw_col]) > 0 and not added:
            comparisons.append(cl.ExactMatch(raw_col))
        else:
            added_hash = any(getattr(comp, "col_name", None) == hash_col for comp in comparisons)
            if hash_col in df.columns and _nonnull_share(df[hash_col]) > 0 and not added_hash:
                comparisons.append(cl.ExactMatch(hash_col))

    if not comparisons:
        raise ValueError(
            "Splink model cannot be trained because no valid comparison columns were generated. "
            f"The table may lack suitable fields for deduplication. Detected roles: {list(roles.keys())}. "
            f"Inference DataFrame columns: {df.columns.tolist()}"
        )

    return SettingsCreator(
        link_type="dedupe_only",
        em_convergence=0.001,
        max_iterations=25,
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=False,
    )


# ---------------------------------------------------------
# Select blocking rules
# ---------------------------------------------------------
def select_optimal_blocking_rules(
    df: SparkDataFrame,
    candidate_rules: List[Tuple[str, object]],
    db_api,
    max_rules: Optional[int] = 5,
    max_comparisons: int = 20_000_000,
) -> Tuple[List[Tuple[str, object]], List[Dict]]:
    diagnostics = []
    scored = []
    
    if "unique_id" not in df.columns:
        df = df.withColumn("unique_id", monotonically_increasing_id())

    for name, rule in candidate_rules:
        cnt = float('inf')
        try:
            result = count_comparisons_from_blocking_rule(
                table_or_tables=df, blocking_rule=rule, link_type="dedupe_only", db_api=db_api
            )
            cnt = int(result.get("number_of_comparisons_to_be_scored_post_filter_conditions", float('inf')))
        except Exception as e:
            print(f"Warning: Could not count comparisons for rule '{name}': {e}")

        print(f"Rule '{name}': {cnt if cnt != float('inf') else 'inf'} comparisons")
        if 0 < cnt <= max_comparisons:
            scored.append((name, rule, cnt))
            diagnostics.append({"name": name, "comparisons": cnt, "kept": True, "reason": "selected"})
        else:
            reason = "too_many_comparisons" if cnt > max_comparisons else "zero_comparisons"
            diagnostics.append({"name": name, "comparisons": cnt, "kept": False, "reason": reason})

    scored.sort(key=lambda x: x[2])
    selected = [(name, rule) for name, rule, cnt in scored[:max_rules]]

    if not selected and candidate_rules:
        selected.append(candidate_rules[0])
        for diag in diagnostics:
            if diag['name'] == candidate_rules[0][0]:
                diag['kept'] = True
                diag['reason'] = "fallback_to_first_candidate"
                break
        else:
            diagnostics.append({"name": candidate_rules[0][0], "comparisons": "unknown", "kept": True, "reason": "fallback_to_first_candidate"})
    elif not selected:
        print("Warning: No candidate blocking rules generated or selected. Creating a generic fallback rule.")
        fallback_rule = block_on("1=1")
        selected.append(("very_loose_default_fallback", fallback_rule))
        diagnostics.append({"name": "very_loose_default_fallback", "comparisons": "very high (estimated)", "kept": True, "reason": "no_other_rules_available"})

    return selected, diagnostics


# ---------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------
def auto_generate_settings(
    df: SparkDataFrame,
    db_api,
    max_rules: Optional[int] = 5,
    universal_mode: bool = True
) -> Tuple[Dict, Dict, List, List, SparkDataFrame]:
    row_count = df.count()
    if row_count < 2:
        raise ValueError("Input DataFrame must have at least two rows to perform deduplication.")

    # Sampling for inference
    sample_fraction = min(1.0, 10000 / row_count)
    inference_df = df.sample(fraction=sample_fraction, seed=42).toPandas()

    # Ensure unique_id for inference
    if "unique_id" not in inference_df.columns:
        inference_df = inference_df.reset_index().rename(columns={"index": "unique_id"})

    # Infer roles
    roles = infer_roles(inference_df.columns)
    print(f"✅ Detected roles: {roles}")

    # Enhance the full DF
    df_enhanced = ensure_derived_columns_enhanced_spark(df, roles)
    if "unique_id" not in df_enhanced.columns:
        df_enhanced = df_enhanced.withColumn("unique_id", monotonically_increasing_id())

    # Ensure inference_df has derived columns
    inference_df = ensure_derived_columns_enhanced_spark(df.sample(fraction=sample_fraction, seed=42), roles).toPandas()

    # Deterministic rules
    deterministic_rules_sql: List[str] = []
    if "numeric_id" in roles and roles["numeric_id"] in df_enhanced.columns:
        id_col = roles["numeric_id"]
        if _nonnull_share(inference_df[id_col]) > 0:
            rule_sql = f'l."{id_col}" = r."{id_col}"'
            deterministic_rules_sql.append(rule_sql)
            print(f"✅ Generated deterministic rule: {rule_sql}")

    # Candidate rules and selection
    candidate_rules = generate_candidate_rules(roles)
    print(f"\n⚙️  Generated {len(candidate_rules)} candidate blocking rules.")

    selected_rules, diagnostics = select_optimal_blocking_rules(
        df_enhanced, candidate_rules, db_api, max_rules=max_rules
    )

    print("\n📊 Blocking rule analysis:")
    for diag in diagnostics:
        status = "✓" if diag["kept"] else "✗"
        reason = f" ({diag['reason']})" if not diag["kept"] else ""
        comps = f"{diag['comparisons']:,}" if isinstance(diag['comparisons'], int) else diag['comparisons']
        print(f"{status} {diag['name']}: {comps} comparisons{reason}")

    final_blocking_rules_for_settings = [rule for _, rule in selected_rules]
    
    settings_creator = build_settings_enhanced(inference_df, roles, final_blocking_rules_for_settings)
    print("\n✅ Successfully generated complete Splink settings.")

    # Convert SettingsCreator to dict
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
        settings_dict = settings_creator.create_settings_dict(sql_dialect_str="spark")
    else:
        raise TypeError(
            f"Could not convert Splink SettingsCreator object ({type(settings_creator)}) "
            f"to a dictionary. Available attributes: {dir(settings_creator)}"
        )

    # Add deterministic rules
    if deterministic_rules_sql:
        settings_dict["deterministic_rules"] = [{"blocking_rule": rule} for rule in deterministic_rules_sql]

    return settings_dict, roles, diagnostics, deterministic_rules_sql, df_enhanced


# Convenience wrapper class
class BlockingBot:
    def __init__(self, db_api):
        self.db_api = db_api

    def auto_generate_settings(self, df: SparkDataFrame, max_rules: Optional[int] = 5):
        return auto_generate_settings(df, self.db_api, max_rules=max_rules)