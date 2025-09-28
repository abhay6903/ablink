# auto_blocking.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from metaphone import doublemetaphone
import hashlib

# Gracefully import unidecode for accent normalization
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

# Splink imports
from splink import block_on, SettingsCreator
import splink.comparison_library as cl
from splink.blocking_analysis import count_comparisons_from_blocking_rule

# --- helper functions ---

def _nonnull_share(series: pd.Series) -> float:
    """Calculate the share of non-null values in a Series."""
    total_rows = len(series)
    if total_rows == 0:
        return 0.0
    return series.notna().sum() / total_rows

def _cardinality_ratio(series: pd.Series) -> float:
    """Calculate the ratio of unique values to total values in a Series."""
    total_rows = len(series)
    if total_rows == 0:
        return 0.0
    return series.nunique(dropna=True) / total_rows

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
        "date": [r".*date.*", r".*_at$", r".*timestamp.*", r"^\d{4}-\d{2}-\d{2}"],
        "url": [r".*(url|website|http|www).*"],
        "geo_lat": [r".*lat(itude)?.*"],
        "geo_lon": [r".*lon(gitude)?.*"],
        "numeric_id": [r".*(id|key|number|no)$"],
        "name": [r".*name.*"],
        "address": [r".*(addr|address|street).*"],
        "geo": [r".*(city|state|province).*"]
    }

def infer_roles_enhanced(input_df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer column roles using a multi-pass approach: direct aliasing, regex patterns,
    and data type/distribution analysis.
    """
    if len(input_df) == 0:
        return {}
        
    lowercase_to_original = {c.lower(): c for c in input_df.columns}
    roles: Dict[str, str] = {}
    
    sample_df = input_df.head(1000)

    # Pass 1: Direct alias matching
    for role, aliases in create_enhanced_column_mapper().items():
        if role in roles: continue
        for alias in aliases:
            if alias in lowercase_to_original:
                col = lowercase_to_original[alias]
                if "unique_id" not in col.lower():
                    roles[role] = col
                    break

    # Pass 2: Pattern-based matching
    for role, patterns in create_semantic_patterns().items():
        if role in roles: continue
        for pattern in patterns:
            for col_lower, col_original in lowercase_to_original.items():
                if re.search(pattern, col_lower, re.IGNORECASE) and "unique_id" not in col_lower:
                    roles[role] = col_original
                    break
            if role in roles: break

    # Pass 3: Data type and distribution-based inference
    for col in input_df.columns:
        if col in roles.values() or "unique_id" in col.lower():
            continue
        
        series_sample = sample_df[col]
        dtype = pd.api.types.infer_dtype(series_sample, skipna=True)
        
        if dtype in ("integer", "floating") and "numeric_id" not in roles:
            if _cardinality_ratio(input_df[col]) >= 0.85 and _nonnull_share(input_df[col]) >= 0.9:
                roles["numeric_id"] = col
        elif dtype == "datetime64" and "date" not in roles:
            roles["date"] = col
        elif dtype == "string":
            card_ratio = _cardinality_ratio(input_df[col])
            if card_ratio < 0.1 and card_ratio > 0 and "category_enum" not in roles:
                roles["category_enum"] = col
            elif card_ratio > 0.5 and "text_freeform" not in roles:
                roles["text_freeform"] = col
                
    return roles

def ensure_derived_columns_enhanced(input_df: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    """
    Safely create derived columns for normalization and phonetic matching.
    """
    df = input_df.copy()
    
    if "full_name" in roles and ("first_name" not in roles or "last_name" not in roles):
        name_parts = df[roles["full_name"]].astype(str).str.split(n=1, expand=True)
        if "first_name" not in roles:
            df["first_name_derived"] = name_parts[0].fillna("")
            roles["first_name"] = "first_name_derived"
        if "last_name" not in roles and len(name_parts.columns) > 1:
            df["last_name_derived"] = name_parts[1].fillna("")
            roles["last_name"] = "last_name_derived"

    for name_type in ["first_name", "last_name"]:
        if name_type in roles and roles[name_type] in df.columns and _nonnull_share(df[roles[name_type]]) > 0.2:
            base_col = roles[name_type]
            if pd.api.types.infer_dtype(df[base_col], skipna=True) in ("string", "mixed"):
                norm_col = f"{name_type}_norm"
                meta_col = f"{name_type}_metaphone"
                
                series = df[base_col].astype(str)
                if unidecode:
                    series = series.apply(unidecode)
                df[norm_col] = series.str.lower().str.strip()
                
                df[meta_col] = df[norm_col].apply(lambda x: doublemetaphone(x)[0] if pd.notna(x) and x else "")

    for role, col_name in roles.items():
        if col_name not in df.columns or _nonnull_share(df[col_name]) <= 0.2:
            continue
        
        try:
            dtype = pd.api.types.infer_dtype(df[col_name], skipna=True)
            if role == "email" and dtype in ("string", "mixed"):
                df["email_norm"] = df[col_name].astype(str).str.lower().str.strip()
            elif role == "phone" and dtype in ("string", "mixed", "integer"):
                df["phone_digits"] = df[col_name].astype(str).str.replace(r"\D", "", regex=True)
            elif role == "zip" and dtype in ("string", "mixed", "integer"):
                df["zip_norm"] = df[col_name].astype(str).str.replace(r"\s", "", regex=True).str.upper()
            elif role in ("city", "state", "address") and dtype in ("string", "mixed"):
                df[f"{role}_norm"] = df[col_name].astype(str).str.lower().str.strip()
            elif role == "date":
                # Ensure the derived column is created before it's used in comparisons
                df["date_norm"] = pd.to_datetime(df[col_name], errors='coerce').dt.strftime('%Y-%m-%d')
            elif role == "url" and dtype in ("string", "mixed"):
                 df['url_domain'] = df[col_name].astype(str).str.extract(r'https?://(?:www\.)?([^/]+)')[0].str.lower()
            elif role == "numeric_id":
                df[f"{col_name}_hash"] = df[col_name].astype(str).apply(lambda x: int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % 10**9)
        except Exception as e:
            print(f"Warning: Could not create derived column for role '{role}': {e}")
            
    if "first_name" in roles and roles["first_name"] in df.columns:
        df["first_name_first_char"] = df[roles["first_name"]].astype(str).str[:1].str.upper()

    return df

def generate_robust_blocking_rules(df: pd.DataFrame, roles: Dict[str, str]) -> List[Tuple[str, object]]:
    """
    Generate a diverse list of candidate blocking rules based on available columns.
    """
    rules = []
    
    def add_rule_if_valid(name, *cols):
        min_non_null_share = 0.3
        if all(c in df.columns and _nonnull_share(df[c]) > min_non_null_share for c in cols):
            rules.append((name, block_on(*cols)))

    if "numeric_id" in roles:
         add_rule_if_valid(f"exact_{roles['numeric_id']}", roles['numeric_id'])
         if pd.api.types.is_numeric_dtype(df.get(roles['numeric_id'])):
             rules.append((f"{roles['numeric_id']}_mod_1000", block_on(f"`{roles['numeric_id']}` % 1000")))
    add_rule_if_valid("exact_email", "email_norm")
    add_rule_if_valid("exact_phone", "phone_digits")
    add_rule_if_valid("exact_zip", "zip_norm")
    add_rule_if_valid("exact_date", "date_norm")
    add_rule_if_valid("exact_first_last_name", "first_name_norm", "last_name_norm")
    add_rule_if_valid("metaphone_full_name", "first_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("exact_last_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("zip_lastname_meta", "zip_norm", "last_name_metaphone")
    add_rule_if_valid("city_firstname", "city_norm", "first_name_norm")
    add_rule_if_valid("first_name_first_char", "first_name_first_char")
    if "category_enum" in roles:
        add_rule_if_valid(f"exact_{roles['category_enum']}", roles['category_enum'])
    if "text_freeform" in roles:
        hash_col = f"{roles['text_freeform']}_hash"
        if hash_col in df.columns:
            add_rule_if_valid(f"hash_{roles['text_freeform']}", hash_col)

    return rules

def build_settings_enhanced(df: pd.DataFrame, roles: Dict[str, str], blocking_rules: List[object]) -> SettingsCreator:
    """
    Build Splink settings by adding comparisons only for columns that exist in the dataframe.
    This version uses CamelCase function names and provides all required arguments.
    """
    comparisons = []

    def add_comparison_if_valid(comparison_func, col_name: str, *args, **kwargs):
        """
        Adds a comparison to the list if the primary column name exists in the DataFrame.
        """
        if col_name in df.columns:
            comparisons.append(comparison_func(col_name, *args, **kwargs))

    if "first_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["first_name"], [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "first_name_metaphone")
    if "last_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["last_name"], [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "last_name_metaphone")
    if "email" in roles:
        add_comparison_if_valid(cl.EmailComparison, "email_norm")
    if "phone" in roles:
        add_comparison_if_valid(cl.LevenshteinAtThresholds, "phone_digits", [2, 4])
    if "address" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "address_norm", [0.7, 0.9])
    if "city" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "city_norm", [0.8, 0.95])
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
    
    if not comparisons:
        if "numeric_id" in roles and f"{roles['numeric_id']}_hash" in df.columns:
            add_comparison_if_valid(cl.ExactMatch, f"{roles['numeric_id']}_hash")
        else:
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
    df: pd.DataFrame,
    candidate_rules: List[Tuple[str, object]],
    db_api,
    max_rules: Optional[int] = 5,
    max_comparisons: int = 20_000_000,
) -> Tuple[List[Tuple[str, object]], List[Dict]]:
    """
    Selects an optimal set of blocking rules by evaluating the number of comparisons.
    """
    diagnostics = []
    scored = []
    for name, rule in candidate_rules:
        cnt = float('inf')
        try:
            # CORRECTED: Pass the 'rule' object directly to the function
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
                diag['reason'] = "fallback"
                break
        else:
             diagnostics.append({"name": candidate_rules[0][0], "comparisons": "unknown", "kept": True, "reason": "fallback"})
             
    return selected, diagnostics

def auto_generate_settings(df: pd.DataFrame, db_api, max_rules: Optional[int] = 5, universal_mode: bool = True) -> Tuple[Dict, Dict, List, pd.DataFrame]:
    """
    Main orchestration function to automatically generate Splink settings from a DataFrame.
    """
    if len(df) < 2:
        raise ValueError("Input DataFrame must have at least two rows to perform deduplication.")

    inference_df = df.sample(n=10000, random_state=42) if len(df) > 100000 else df
    
    roles = infer_roles_enhanced(inference_df)
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
        reason = f" ({diag['reason']})" if not diag["kept"] else ""
        comps = f"{diag['comparisons']:,}" if isinstance(diag['comparisons'], int) else diag['comparisons']
        print(f"{status} {diag['name']}: {comps} comparisons{reason}")

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
        settings_dict = settings_creator.create_settings_dict(sql_dialect_str="duckdb")
    else:
        raise TypeError(
            f"Could not convert Splink SettingsCreator object ({type(settings_creator)}) "
            f"to a dictionary. This is likely due to an incompatible Splink library version. "
            f"Available attributes: {dir(settings_creator)}"
        )

    # CORRECTED: Return the enhanced dataframe along with other artifacts
    return settings_dict, roles, diagnostics, df_enhanced

class BlockingBot:
    def __init__(self, db_api):
        self.db_api = db_api

    def auto_generate_settings(self, df: pd.DataFrame, max_rules: Optional[int] = 5):
        return auto_generate_settings(df, self.db_api, max_rules=max_rules)