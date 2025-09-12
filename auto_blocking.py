from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
from metaphone import doublemetaphone

# Splink imports kept minimal so this module can be imported before a DB is ready
from splink import block_on
import splink.comparison_library as cl
from splink import SettingsCreator
from splink.blocking_analysis import (
    count_comparisons_from_blocking_rule,
    n_largest_blocks,
)


def _nonnull_share(series: pd.Series) -> float:
    total_rows = len(series)
    if total_rows == 0:
        return 0.0
    non_null_count = series.notna().sum()
    return non_null_count / total_rows


def _cardinality_ratio(series: pd.Series) -> float:
    total_rows = len(series)
    if total_rows == 0:
        return 0.0
    unique_count = series.nunique(dropna=True)
    return unique_count / total_rows


def create_enhanced_column_mapper() -> Dict[str, List[str]]:
    """Enhanced column name mapping with more variations and semantic understanding"""
    return {
        "first_name": [
            "first_name", "firstname", "given_name", "fname", "forename", 
            "name_first", "first", "given", "christian_name", "personal_name"
        ],
        "last_name": [
            "last_name", "lastname", "surname", "lname", "family_name", 
            "name_last", "last", "family", "sur_name"
        ],
        "full_name": [
            "full_name", "customer_name", "name", "person_name", "complete_name",
            "fullname", "personname", "client_name", "user_name", "individual_name",
            "name_full", "contact_name", "party_name", "entity_name"
        ],
        "email": [
            "email", "email_id", "mail", "email_address", "e_mail", "emailid",
            "mail_id", "electronic_mail", "contact_email", "primary_email"
        ],
        "phone": [
            "phone", "mobile", "mobile_no", "contact", "contact_no", "phone_number", 
            "tel", "telephone", "cell", "cellular", "phone_no", "contact_number",
            "mobile_number", "primary_phone", "home_phone", "work_phone"
        ],
        "zip": [
            "zip", "zipcode", "postal_code", "pincode", "postcode", "zip_code",
            "postalcode", "post_code", "area_code", "pin", "postcode_fake"
        ],
        "city": [
            "city", "town", "locality", "municipality", "place", "township",
            "urban_area", "settlement", "location_city"
        ],
        "state": [
            "state", "province", "region", "territory", "district", "county",
            "state_province", "administrative_area", "governo"
        ],
        "address": [
            "address", "street", "street_address", "addr", "location", "residence",
            "home_address", "mailing_address", "physical_address", "street_addr"
        ],
        "id": [
            "account", "acct", "account_no", "account_number", "record_id", 
            "customer_id", "user_id", "pan", "vat", "aadhaar", "gov_id",
            "person_id", "client_id", "entity_id", "reference_id", "ref_id",
            "identification", "identifier", "primary_key", "pk", "customer_no"
        ],
    }


def create_semantic_patterns() -> Dict[str, List[str]]:
    """Regex patterns to identify columns by content/naming patterns"""
    return {
        "email": [r".*mail.*", r".*@.*"],
        "phone": [r".*(phone|mobile|tel|contact).*", r".*\d{10,}.*"],
        "zip": [r".*(zip|postal|pin).*code.*", r".*post.*code.*"],
        "id": [r".*(id|key|number|no)$", r".*(account|customer|user|person).*id.*"],
        "name": [r".*name.*", r".*(first|last|full|given|family).*"],
        "address": [r".*(addr|address|street|location).*"],
        "geo": [r".*(city|state|province|region|country).*"]
    }


def infer_roles_enhanced(input_df: pd.DataFrame) -> Dict[str, str]:
    """Enhanced role inference with better semantic understanding"""
    lowercase_to_original = {column.lower(): column for column in input_df.columns}
    column_mappers = create_enhanced_column_mapper()
    semantic_patterns = create_semantic_patterns()
    
    roles: Dict[str, str] = {}

    # First pass: Direct alias matching
    for role_name, aliases in column_mappers.items():
        if role_name in roles:  # Skip if already found
            continue
        for alias in aliases:
            if alias in lowercase_to_original:
                candidate_col = lowercase_to_original[alias]
                # Skip unique_id columns for non-unique roles
                if role_name == "id" and ("unique_id" in candidate_col.lower()):
                    continue
                roles[role_name] = candidate_col
                break

    # Second pass: Pattern-based matching for missing roles
    for role_name in column_mappers.keys():
        if role_name in roles:
            continue
        
        # Get relevant patterns
        patterns_to_check = []
        if role_name in ["first_name", "last_name", "full_name"]:
            patterns_to_check = semantic_patterns.get("name", [])
        elif role_name in semantic_patterns:
            patterns_to_check = semantic_patterns[role_name]
        
        for pattern in patterns_to_check:
            for col_lower, col_original in lowercase_to_original.items():
                if re.search(pattern, col_lower, re.IGNORECASE):
                    # Additional validation based on role
                    if role_name == "id":
                        if "unique_id" in col_lower:
                            continue
                        if (_cardinality_ratio(input_df[col_original]) >= 0.7 and 
                            _nonnull_share(input_df[col_original]) >= 0.8):
                            roles[role_name] = col_original
                            break
                    elif role_name in ["email", "phone"]:
                        if _nonnull_share(input_df[col_original]) >= 0.3:
                            roles[role_name] = col_original
                            break
                    else:
                        roles[role_name] = col_original
                        break
            if role_name in roles:
                break

    # Third pass: Distribution-based inference for high-cardinality ID fields
    if "id" not in roles:
        for col_lower, col_original in lowercase_to_original.items():
            if "unique_id" in col_lower:
                continue
            try:
                cardinality = _cardinality_ratio(input_df[col_original])
                completeness = _nonnull_share(input_df[col_original])
                if cardinality >= 0.85 and completeness >= 0.9:
                    roles["id"] = col_original
                    break
            except Exception:
                continue

    return roles


def ensure_derived_columns_enhanced(input_df: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    """Enhanced derived column creation with better error handling"""
    df = input_df.copy()
    
    try:
        # Handle full name normalization and splitting
        if "full_name" in roles:
            col_name = roles["full_name"]
            if col_name in df.columns:
                df["full_name_norm"] = df[col_name].astype(str).str.strip().str.lower()
                
                # Smart name splitting - handle various formats
                if "first_name" not in roles or "last_name" not in roles:
                    # Try to extract first and last names
                    name_parts = df["full_name_norm"].str.split(n=1, expand=True)
                    if len(name_parts.columns) >= 2:
                        if "first_name" not in roles:
                            df["first_name"] = name_parts[0].fillna("")
                            roles["first_name"] = "first_name"
                        if "last_name" not in roles:
                            df["last_name"] = name_parts[1].fillna("")
                            roles["last_name"] = "last_name"
                    else:
                        # Single name case
                        if "first_name" not in roles:
                            df["first_name"] = df["full_name_norm"]
                            roles["first_name"] = "first_name"

        # Normalize individual name fields
        for name_type in ["first_name", "last_name"]:
            if name_type in roles and f"{name_type}_norm" not in df.columns:
                col_name = roles[name_type]
                if col_name in df.columns:
                    df[f"{name_type}_norm"] = df[col_name].astype(str).str.lower().str.strip()
        
        # Create metaphone encodings with error handling
        for name_type in ["first_name", "last_name"]:
            metaphone_col = f"{name_type}_metaphone"
            norm_col = f"{name_type}_norm"
            if name_type in roles and metaphone_col not in df.columns and norm_col in df.columns:
                def safe_metaphone(value):
                    try:
                        if pd.isna(value) or str(value).strip() == "":
                            return ""
                        return doublemetaphone(str(value))[0] or ""
                    except Exception:
                        return ""
                
                df[metaphone_col] = df[norm_col].apply(safe_metaphone)

        # Handle email normalization
        if "email" in roles:
            col_name = roles["email"]
            if col_name in df.columns:
                df["email_norm"] = df[col_name].astype(str).str.lower().str.strip()
                if "email_domain" not in df.columns:
                    df["email_domain"] = df["email_norm"].str.extract(r"@([^@\s]+)$")[0].fillna("")

        # Handle phone normalization
        if "phone" in roles:
            col_name = roles["phone"]
            if col_name in df.columns:
                df["phone_digits"] = df[col_name].astype(str).str.replace(r"\D", "", regex=True)

        # Handle address/location fields
        location_fields = ["zip", "city", "state"]
        for field in location_fields:
            if field in roles:
                col_name = roles[field]
                norm_col = f"{field}_norm"
                if col_name in df.columns and norm_col not in df.columns:
                    if field == "zip":
                        df[norm_col] = df[col_name].astype(str).str.replace(r"\s", "", regex=True).str.upper()
                    else:
                        df[norm_col] = df[col_name].astype(str).str.lower().str.strip()

    except Exception as e:
        print(f"Warning: Error in derived column creation: {e}")
    
    return df


def generate_robust_blocking_rules(df: pd.DataFrame, roles: Dict[str, str]) -> List[Tuple[str, object]]:
    """Generate robust blocking rules based on available columns and their characteristics"""
    rules: List[Tuple[str, object]] = []
    
    # High selectivity rules first
    if "id" in roles:
        col = roles["id"]
        if col in df.columns and col.lower() != "unique_id":
            completeness = _nonnull_share(df[col])
            cardinality = _cardinality_ratio(df[col])
            if completeness > 0.85 and cardinality > 0.7:
                rules.append((f"exact_{col}", block_on(col)))

    # Email-based blocking
    if "email" in roles and "email_norm" in df.columns:
        if _nonnull_share(df["email_norm"]) > 0.4 and _cardinality_ratio(df["email_norm"]) > 0.3:
            rules.append(("exact_email", block_on("email_norm")))
    
    # Phone-based blocking
    if "phone" in roles and "phone_digits" in df.columns:
        if _nonnull_share(df["phone_digits"]) > 0.4 and _cardinality_ratio(df["phone_digits"]) > 0.3:
            rules.append(("exact_phone", block_on("phone_digits")))

    # Name-based compound rules
    if all(f"{name}_metaphone" in df.columns for name in ["first_name", "last_name"]):
        rules.append(("metaphone_full_name", block_on("first_name_metaphone", "last_name_metaphone")))
    
    # Geographic compound rules
    if "zip_norm" in df.columns and "last_name_metaphone" in df.columns:
        rules.append(("zip_lastname", block_on("zip_norm", "last_name_metaphone")))
    
    if "city_norm" in df.columns and "first_name_norm" in df.columns:
        rules.append(("city_firstname_initial", block_on("city_norm", "substr(first_name_norm, 1, 1)")))

    # Medium selectivity geographic rules
    geographic_fields = ["zip_norm", "city_norm", "state_norm"]
    for field in geographic_fields:
        if field in df.columns:
            field_name = field.replace("_norm", "")
            rules.append((f"exact_{field_name}", block_on(field)))

    # Fallback rules for name initials
    if "first_name_norm" in df.columns:
        rules.append(("first_name_initial", block_on("substr(first_name_norm, 1, 1)")))
    
    if "last_name_metaphone" in df.columns:
        rules.append(("last_name_metaphone", block_on("last_name_metaphone")))

    return rules


def build_settings_enhanced(roles: Dict[str, str]) -> SettingsCreator:
    """Build enhanced Splink settings based on detected roles"""
    comparisons: List[object] = []

    # Name comparisons with multiple thresholds
    if "first_name" in roles and "first_name_norm" in roles:  # Check if derived column exists
        comparisons.append(cl.JaroWinklerAtThresholds("first_name_norm", [0.7, 0.85, 0.95]))
        if "first_name_metaphone" in roles:
            comparisons.append(cl.ExactMatch("first_name_metaphone"))
    
    if "last_name" in roles and "last_name_norm" in roles:
        comparisons.append(cl.JaroWinklerAtThresholds("last_name_norm", [0.7, 0.85, 0.95]))
        if "last_name_metaphone" in roles:
            comparisons.append(cl.ExactMatch("last_name_metaphone"))
    
    # Contact information
    if "email" in roles and "email_norm" in roles:
        comparisons.append(cl.EmailComparison("email_norm"))
    
    if "phone" in roles and "phone_digits" in roles:
        comparisons.append(cl.ExactMatch("phone_digits"))
    
    # Geographic comparisons
    if "zip" in roles and "zip_norm" in roles:
        comparisons.append(cl.ExactMatch("zip_norm"))
    
    if "city" in roles and "city_norm" in roles:
        comparisons.append(cl.JaroWinklerAtThresholds("city_norm", [0.85, 0.95]))
    
    if "state" in roles and "state_norm" in roles:
        comparisons.append(cl.ExactMatch("state_norm"))
    
    # ID comparison (if available)
    if "id" in roles:
        comparisons.append(cl.ExactMatch(roles["id"]))

    return SettingsCreator(
        link_type="dedupe_only",
        em_convergence=0.001,
        max_iterations=25,  # Prevent infinite loops
        comparisons=comparisons,
    )


def select_optimal_blocking_rules(
    df: pd.DataFrame,
    candidate_rules: List[Tuple[str, object]],
    db_api,
    max_rules: int = 3,
    max_comparisons: int = 50_000_000,  # Prevent memory issues
    link_type: str = "dedupe_only",
) -> Tuple[List[Tuple[str, object]], List[Dict]]:
    """Select optimal blocking rules with memory and performance constraints"""
    
    diagnostics = []
    scored: List[Tuple[str, object, int]] = []
    
    for name, rule in candidate_rules:
        try:
            cnt = count_comparisons_from_blocking_rule(
                table_or_tables=df,
                blocking_rule=rule,
                link_type=link_type,
                db_api=db_api,
            )
            # Handle dictionary output from Splink
            if isinstance(cnt, dict):
                if "count" in cnt:
                    cnt = int(cnt["count"])  # Extract the count
                else:
                    print(f"Warning: Invalid dictionary from count_comparisons for rule '{name}': {cnt}")
                    cnt = float('inf')  # Fallback for invalid dict
            elif not isinstance(cnt, (int, float)) or cnt is None:
                print(f"Warning: Invalid count type for rule '{name}': {cnt} (type: {type(cnt)})")
                cnt = float('inf')  # Fallback for non-numeric
            else:
                cnt = int(cnt)  # Ensure integer
        except Exception as e:
            print(f"Warning: Could not count comparisons for rule '{name}': {e}")
            cnt = float('inf')  # Fallback for errors
            
        # Debug: Log count value and type
        print(f"Rule '{name}': comparisons={cnt}, type={type(cnt)}")

        # Skip rules that would create too many comparisons
        if cnt != float('inf') and cnt > max_comparisons:
            diagnostics.append({"name": name, "comparisons": cnt, "kept": False, "reason": "too_many_comparisons"})
            continue
            
        scored.append((name, rule, cnt))

    # Sort by fewest comparisons first (most selective)
    try:
        scored.sort(key=lambda x: x[2])
    except Exception as e:
        print(f"Error sorting scored rules: {e}")
        print("Scored rules (unsorted):", [(name, cnt) for name, _, cnt in scored])
        # Fallback: Use unsorted list
        pass
    
    selected: List[Tuple[str, object]] = []
    for name, rule, cnt in scored:
        keep = cnt > 0 and cnt != float('inf') and len(selected) < max_rules
        reason = "selected" if keep else ("zero_comparisons" if cnt == 0 else "max_rules_reached" if len(selected) >= max_rules else "invalid_count")
        diagnostics.append({"name": name, "comparisons": cnt, "kept": keep, "reason": reason})
        if keep:
            selected.append((name, rule))

    # Fallback if no good rules
    if not selected and candidate_rules:
        selected = [(candidate_rules[0][0], candidate_rules[0][1])]  # First safe rule
        diagnostics.append({"name": candidate_rules[0][0], "comparisons": float('inf'), "kept": True, "reason": "fallback"})

    return selected, diagnostics


def auto_generate_blocking_rules(df: pd.DataFrame, db_api, max_rules: int = 3) -> Tuple[List[Tuple[str, object]], Dict[str, str]]:
    """Main function to automatically generate and select blocking rules"""
    
    # Step 1: Infer column roles
    roles = infer_roles_enhanced(df)
    print(f"Detected roles: {roles}")
    
    # Step 2: Ensure derived columns exist
    df_enhanced = ensure_derived_columns_enhanced(df, roles)
    
    # Step 3: Generate candidate blocking rules
    candidate_rules = generate_robust_blocking_rules(df_enhanced, roles)
    print(f"Generated {len(candidate_rules)} candidate blocking rules")
    
    # Step 4: Select optimal rules
    selected_rules, diagnostics = select_optimal_blocking_rules(
        df_enhanced, candidate_rules, db_api, max_rules=max_rules
    )
    
    # Print diagnostics
    print("\nBlocking rule analysis:")
    for diag in diagnostics:
        status = "✓" if diag["kept"] else "✗"
        reason = f" ({diag['reason']})" if not diag["kept"] else ""
        print(f"{status} {diag['name']}: {diag['comparisons']:,} comparisons{reason}")
    
    return selected_rules, roles


# Legacy compatibility functions
def infer_roles_generic(input_df: pd.DataFrame) -> Dict[str, str]:
    """Legacy compatibility - redirects to enhanced version"""
    return infer_roles_enhanced(input_df)

def ensure_derived_columns(input_df: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    """Legacy compatibility - redirects to enhanced version"""
    return ensure_derived_columns_enhanced(input_df, roles)

def build_settings(roles: Dict[str, str]) -> SettingsCreator:
    """Legacy compatibility - redirects to enhanced version"""
    return build_settings_enhanced(roles)

def auto_select_notebook_style_rules(df: pd.DataFrame, db_api, max_rules: int = 3) -> List[Tuple[str, object]]:
    """Legacy compatibility - uses enhanced auto-generation"""
    selected_rules, _ = auto_generate_blocking_rules(df, db_api, max_rules)
    return selected_rules