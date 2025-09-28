from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import (
    col, lower, upper, trim, regexp_replace, substring, split,
    element_at, expr, when, length
)
import splink.comparison_library as cl
from splink import SettingsCreator, block_on, SparkAPI
from splink.blocking_analysis import count_comparisons_from_blocking_rule


# -----------------------------
# Helpers
# -----------------------------
def _nonnull_share(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    if total_rows == 0:
        return 0.0
    return df.filter(col(col_name).isNotNull()).count() / total_rows

def _cardinality_ratio(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    if total_rows == 0:
        return 0.0
    unique_count = df.select(col_name).distinct().count()
    return unique_count / total_rows


# -----------------------------
# Role inference
# -----------------------------
def create_enhanced_column_mapper() -> Dict[str, List[str]]:
    return {
        "first_name": ["first_name", "fname", "given_name", "forename"],
        "last_name": ["last_name", "surname", "lname", "family_name"],
        "full_name": ["full_name", "name", "customer_name"],
        "email": ["email", "email_address"],
        "phone": ["phone", "mobile", "phone_number", "contact_no"],
        "zip": ["zip", "zipcode", "postal_code", "pincode"],
        "city": ["city", "town"],
        "state": ["state", "province", "region"],
        "address": ["address", "street", "street_address"],
        "date": ["dob", "date", "birth_date"],
        "numeric_id": ["id", "customer_id", "user_id", "person_id"],
    }

def create_semantic_patterns() -> Dict[str, str]:
    return {
        "email": r".*@.*",
        "phone": r"^\+?\d{6,15}$",
        "zip": r"^\d{4,10}$",
        "date": r"^\d{4}-\d{2}-\d{2}$",
    }

def infer_roles_enhanced(df: SparkDataFrame) -> Dict[str, str]:
    roles = {}
    lower_map = {c.lower(): c for c in df.columns}
    for role, aliases in create_enhanced_column_mapper().items():
        for alias in aliases:
            if alias in lower_map:
                roles[role] = lower_map[alias]
                break

    # Regex / heuristic inference
    total_rows = df.count()
    patterns = create_semantic_patterns()
    for col_name in df.columns:
        if col_name in roles.values():
            continue
        sample_val = df.select(col_name).filter(col(col_name).isNotNull()).limit(1).collect()
        if not sample_val:
            continue
        val = str(sample_val[0][0])
        for role, pattern in patterns.items():
            if re.match(pattern, val):
                roles[role] = col_name
    return roles


# -----------------------------
# Derived Columns
# -----------------------------
from pyspark.sql.functions import col, lower, trim, upper, regexp_replace, split, element_at, expr

def ensure_derived_columns_enhanced(df: SparkDataFrame, roles: Dict[str, str]) -> SparkDataFrame:
    df_transformed = df

    # Split full name → first/last
    if "full_name" in roles and ("first_name" not in roles or "last_name" not in roles):
        parts = split(col(roles["full_name"]), " ", 2)
        if "first_name" not in roles:
            df_transformed = df_transformed.withColumn("first_name", element_at(parts, 1))
            roles["first_name"] = "first_name"
        if "last_name" not in roles:
            df_transformed = df_transformed.withColumn("last_name", element_at(parts, 2))
            roles["last_name"] = "last_name"

    # Normalised + phonetic names
    for name_type in ["first_name", "last_name"]:
        if name_type in roles:
            base_col = roles[name_type]
            df_transformed = df_transformed.withColumn(
                f"{name_type}_norm",
                lower(trim(expr(f"accent_remove(`{base_col}`)")))
            )
            df_transformed = df_transformed.withColumn(
                f"{name_type}_metaphone",
                expr(f"double_metaphone(`{name_type}_norm`)")
            )

    # Email
    if "email" in roles:
        df_transformed = df_transformed.withColumn(
            "email_norm", lower(trim(col(roles["email"])))
        )

    # Phone digits
    if "phone" in roles:
        df_transformed = df_transformed.withColumn(
            "phone_digits", regexp_replace(col(roles["phone"]).cast("string"), r"\D", "")
        )

    # ZIP
    if "zip" in roles:
        df_transformed = df_transformed.withColumn(
            "zip_norm", upper(regexp_replace(col(roles["zip"]).cast("string"), r"\s", ""))
        )

    # Address
    if "address" in roles:
        df_transformed = df_transformed.withColumn(
            "address_norm", lower(trim(expr(f"accent_remove(`{roles['address']}`)")))
        )

    # City
    if "city" in roles:
        df_transformed = df_transformed.withColumn(
            "city_norm", lower(trim(expr(f"accent_remove(`{roles['city']}`)")))
        )

    # State
    if "state" in roles:
        df_transformed = df_transformed.withColumn(
            "state_norm", lower(trim(expr(f"accent_remove(`{roles['state']}`)")))
        )

    return df_transformed.checkpoint(eager=True)



# -----------------------------
# Blocking Rules
# -----------------------------
from splink import block_on

def generate_robust_blocking_rules(df: SparkDataFrame, roles: Dict[str, str]):
    rules = []

    def add_if(*cols):
        if all(c in df.columns for c in cols):
            rules.append(block_on(*cols))

    # Simple blocking
    if "email_norm" in df.columns:
        add_if("email_norm")
    if "phone_digits" in df.columns:
        add_if("phone_digits")

    # Name-based
    if "first_name_norm" in df.columns and "last_name_norm" in df.columns:
        add_if("first_name_norm", "last_name_norm")
    if "first_name_metaphone" in df.columns and "last_name_metaphone" in df.columns:
        add_if("first_name_metaphone", "last_name_metaphone")

    # Mixed
    if "zip_norm" in df.columns and "last_name_metaphone" in df.columns:
        add_if("zip_norm", "last_name_metaphone")
    if "city_norm" in df.columns and "first_name_norm" in df.columns:
        add_if("city_norm", "first_name_norm")

    return rules



# -----------------------------
# Select optimal rules
# -----------------------------
def select_optimal_blocking_rules(
    df: SparkDataFrame,
    candidate_rules,
    db_api: SparkAPI,
    max_rules=5,
    max_comparisons=2e7
):
    diagnostics, scored = [], []
    for rule in candidate_rules:
        try:
            result = count_comparisons_from_blocking_rule(df, rule, "dedupe_only", db_api)
            count = int(result.get("number_of_comparisons_to_be_scored_post_filter_conditions", float('inf')))
            if 0 < count <= max_comparisons:
                scored.append((rule, count))
                diagnostics.append({"rule": str(rule), "comparisons": count, "kept": True})
            else:
                diagnostics.append({"rule": str(rule), "comparisons": count, "kept": False})
        except Exception as e:
            diagnostics.append({"rule": str(rule), "comparisons": "error", "kept": False, "reason": str(e)})

    scored.sort(key=lambda x: x[1])
    selected = [rule for rule, _ in scored[:max_rules]]
    if not selected and candidate_rules:
        selected = [candidate_rules[0]]
    return selected, diagnostics


# -----------------------------
# Build Splink Settings
# -----------------------------
import splink.comparison_library as cl
from splink import SettingsCreator

def build_settings_enhanced(df: SparkDataFrame, roles: Dict[str, str], blocking_rules):
    comparisons = []

    if "first_name_norm" in df.columns:
        comparisons.append(cl.NameComparison("first_name_norm"))
    if "last_name_norm" in df.columns:
        comparisons.append(cl.NameComparison("last_name_norm"))
    if "dob" in roles:
        comparisons.append(cl.LevenshteinAtThresholds("dob"))
    if "city_norm" in df.columns:
        comparisons.append(cl.ExactMatch("city_norm").configure(term_frequency_adjustments=True))
    if "email_norm" in df.columns:
        comparisons.append(cl.EmailComparison("email_norm"))

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=True,
        em_convergence=0.01,
    )
    return settings



# -----------------------------
# Main orchestration
# -----------------------------
def auto_generate_settings(df: SparkDataFrame, db_api: SparkAPI, max_rules=5):
    roles = infer_roles_enhanced(df)
    df_enhanced = ensure_derived_columns_enhanced(df, roles)
    candidate_rules = generate_robust_blocking_rules(df_enhanced, roles)
    selected, diagnostics = select_optimal_blocking_rules(df_enhanced, candidate_rules, db_api, max_rules)

    # Build Splink settings
    settings = build_settings_enhanced(df_enhanced, roles, selected)

    # Deterministic rules for training (use selected blocking rules)
    deterministic_rules = selected

    # ✅ return 5 values
    return settings, roles, diagnostics, df_enhanced, deterministic_rules
