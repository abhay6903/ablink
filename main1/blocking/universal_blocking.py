import re
import pandas as pd
import polars as pl
import jellyfish
from concurrent.futures import ProcessPoolExecutor, as_completed
import altair as alt
import numpy as np
from typing import Optional, Dict, List, Union
from sqlalchemy import create_engine
import logging
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('mode.chained_assignment', None)

# Helper Functions (Unchanged)
def normalize_col(col: str) -> str:
    return re.sub(r'[^a-z0-9]', '', col.lower())

def normalize_date(date_value: Union[str, pd.Timestamp], dayfirst: bool = True) -> Optional[str]:
    if pd.isna(date_value) or not date_value: return None
    if isinstance(date_value, str):
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d']
        for fmt in date_formats:
            try: return pd.to_datetime(date_value, format=fmt).strftime('%Y-%m-%d')
            except: continue
        try: return pd.to_datetime(date_value, dayfirst=dayfirst, errors='coerce').strftime('%Y-%m-%d')
        except: return None
    return pd.to_datetime(date_value).strftime('%Y-%m-%d')

def normalize_time(time_value: Union[str, pd.Timestamp]) -> Optional[str]:
    if pd.isna(time_value) or not time_value: return None
    if isinstance(time_value, str):
        time_formats = ['%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']
        for fmt in time_formats:
            try: return pd.to_datetime(time_value, format=fmt).strftime('%Y-%m-%d %H:%M:%S')
            except: continue
        try: return pd.to_datetime(time_value, errors='coerce').strftime('%Y-%m-%d %H:%M:%S')
        except: return None
    return pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M:%S')

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    return jellyfish.jaro_winkler_similarity(s1, s2) if s1 and s2 else 0.0

def levenshtein_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2: return 0.0
    max_len = max(len(s1), len(s2))
    return 1.0 - (jellyfish.levenshtein_distance(s1, s2) / max_len) if max_len else 1.0

def fuzzy_match(s1: str, s2: str, threshold: float, method: str = "jaro_winkler") -> bool:
    if pd.isna(s1) or pd.isna(s2): return False
    s1, s2 = str(s1).strip(), str(s2).strip()
    sim = jaro_winkler_similarity(s1, s2) if method == "jaro_winkler" else levenshtein_similarity(s1, s2)
    return sim >= threshold

ATTRIBUTE_PATTERNS = {
    "first_name": ["first", "fname", "given"], "last_name": ["last", "lname", "surname", "family"],
    "dob": ["dob", "birth", "dateofbirth"], "year_of_birth": ["yob", "birthyear"],
    "month_of_birth": ["mob", "birthmonth"], "email": ["email", "mail"],
    "phone": ["phone", "mobile", "tel", "contact"], "passport": ["passport"],
    "national_id": ["nid", "nationalid", "ssn", "aadhar"], "street": ["street", "st"],
    "house_number": ["houseno", "house", "building"], "city": ["city", "town"],
    "postcode": ["postcode", "zip", "postal"], "time": ["time", "timestamp", "created", "updated"]
}

def match_attribute(col_name: str) -> str:
    norm = normalize_col(col_name)
    for attr, keywords in ATTRIBUTE_PATTERNS.items():
        for kw in keywords:
            if kw in norm: return attr
    return None

# Top-level function for process pool
def _process_fuzzy_block(group_data: pd.DataFrame, id_col: str, col1: str, col2: Optional[str], threshold1: float, threshold2: Optional[float], rule_name: str) -> List[tuple]:
    pairs = []
    ids = group_data[id_col].values
    col1_vals = group_data[col1].values
    col2_vals = group_data[col2].values if col2 else None

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            match1 = fuzzy_match(col1_vals[i], col1_vals[j], threshold1)
            if col2 and match1:
                if fuzzy_match(col2_vals[i], col2_vals[j], threshold2):
                    pairs.append((ids[i], ids[j], rule_name))
            elif match1:
                pairs.append((ids[i], ids[j], rule_name))
    return pairs

class TrinoBlocking:
    def __init__(self, conn_str: str, view_name: str, record_id_col: str = "RecordID"):
        self.conn_str = conn_str
        self.view_name = view_name
        self.id_col = record_id_col
        self.engine = create_engine(conn_str, connect_args={'http_scheme': 'http'})
        self.available_columns = pd.read_sql(f"SELECT * FROM {self.view_name} LIMIT 0", self.engine).columns
        self.attr_map = {}
        for col in self.available_columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map: self.attr_map[attr] = col
        self.rules = self._generate_exact_rules()
        logging.info(f"Optimized TrinoBlocking rules: {list(self.rules.keys())}")

    def _generate_exact_rules(self) -> dict:
        rules = {}

        def create_sql_rule(rule_name: str, blocking_keys: Dict[str, str]):
            def rule():
                with_clauses = ",\n".join([f"{alias} AS ({expr})" for alias, expr in blocking_keys.items()])
                group_cols = list(blocking_keys.keys())
                
                query = f"""
                WITH NormalizedData AS (
                    SELECT
                        {self.id_col},
                        {with_clauses}
                    FROM {self.view_name}
                )
                SELECT
                    t1.{self.id_col} AS RecordID1,
                    t2.{self.id_col} AS RecordID2,
                    '{rule_name}' AS Rule
                FROM NormalizedData t1
                JOIN NormalizedData t2 ON {' AND '.join([f't1.{col} = t2.{col}' for col in group_cols])}
                WHERE t1.{self.id_col} < t2.{self.id_col}
                AND {' AND '.join([f't1.{col} IS NOT NULL' for col in group_cols])}
                """
                try:
                    return pl.from_pandas(pd.read_sql(query, self.engine))
                except Exception as e:
                    logging.error(f"SQL query failed for rule '{rule_name}': {e}")
                    return pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])
            
            rule.columns = [expr for expr in blocking_keys.values()]
            return rule

        if "email" in self.attr_map:
            c = self.attr_map["email"]
            rules["Email"] = create_sql_rule("Email", {"key1": f"LOWER(TRIM({c}))"})
        if "passport" in self.attr_map:
            c = self.attr_map["passport"]
            rules["Passport"] = create_sql_rule("Passport", {"key1": f"TRIM(REPLACE({c}, ' ', ''))"})
        if "national_id" in self.attr_map:
            c = self.attr_map["national_id"]
            rules["NationalID"] = create_sql_rule("NationalID", {"key1": f"TRIM(REPLACE({c}, '-', ''))"})
        if "phone" in self.attr_map:
            c = self.attr_map["phone"]
            rules["Phone"] = create_sql_rule("Phone", {"key1": f"REGEXP_REPLACE({c}, '[^0-9]', '')"})

        if "first_name" in self.attr_map and "last_name" in self.attr_map and "dob" in self.attr_map:
            f, l, d = self.attr_map["first_name"], self.attr_map["last_name"], self.attr_map["dob"]
            rules["Name_DOB_Soundex"] = create_sql_rule("Name_DOB_Soundex", {
                "key1": f"SOUNDEX(LOWER(TRIM({f})))",
                "key2": f"LOWER(TRIM({l}))",
                "key3": f"CAST({d} AS VARCHAR)"
            })
        return rules

    def run_rule(self, rule_name: str) -> pl.DataFrame:
        if rule_name not in self.rules: return pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])
        return self.rules[rule_name]()

    def run_all(self, parallel=True, max_workers=8) -> dict:
        res = {}
        # SQL rules are executed on the server, so parallelism here is less critical
        for rule_name in self.rules.keys():
            df = self.run_rule(rule_name)
            if not df.is_empty():
                res[rule_name] = df
        return res

class PolarsBlocking:
    def __init__(self, df: pl.DataFrame, record_id_col: str = "RecordID"):
        self.df = df.select([col for col in df.columns if df[col].dtype in [pl.String, pl.Int64, pl.Float64]])
        self.id_col = record_id_col
        self.attr_map = {}
        for col in df.columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map: self.attr_map[attr] = col
        self.rules = self._generate_fuzzy_rules()
        logging.info(f"Optimized PolarsBlocking rules (using ProcessPoolExecutor): {list(self.rules.keys())}")

    def _generate_fuzzy_rules(self) -> dict:
        rules = {}

        def create_fuzzy_rule(rule_name: str, blocking_col: str, fuzzy_col1: str, fuzzy_col2: Optional[str] = None, threshold1: float = 0.9, threshold2: Optional[float] = 0.9):
            def rule(max_workers: int):
                # Ensure blocking column is string for grouping
                if self.df[blocking_col].dtype != pl.String:
                    temp_df = self.df.with_columns(pl.col(blocking_col).cast(pl.String))
                else:
                    temp_df = self.df
                
                # Filter out nulls in blocking column and convert to Pandas for processing
                grouped = temp_df.filter(pl.col(blocking_col).is_not_null()).group_by(blocking_col)
                
                all_pairs = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for _, group_df in grouped:
                        if len(group_df) > 1:
                            # Pass data as Pandas DataFrame, which is pickleable
                            future = executor.submit(_process_fuzzy_block, group_df.to_pandas(), self.id_col, fuzzy_col1, fuzzy_col2, threshold1, threshold2, rule_name)
                            futures.append(future)
                    
                    for future in as_completed(futures):
                        all_pairs.extend(future.result())

                if not all_pairs: return pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])
                return pl.DataFrame(all_pairs, schema=["RecordID1", "RecordID2", "Rule"])

            rule.columns = [fuzzy_col1, fuzzy_col2] if fuzzy_col2 else [fuzzy_col1]
            return rule

        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f, l = self.attr_map["first_name"], self.attr_map["last_name"]
            rules["FuzzyName_SameLastName"] = create_fuzzy_rule("FuzzyName_SameLastName", blocking_col=l, fuzzy_col1=f, threshold1=0.90)

        if "street" in self.attr_map and "postcode" in self.attr_map:
            s, p = self.attr_map["street"], self.attr_map["postcode"]
            rules["FuzzyStreet_SamePostcode"] = create_fuzzy_rule("FuzzyStreet_SamePostcode", blocking_col=p, fuzzy_col1=s, threshold1=0.92)
        
        return rules

    def run_rule(self, rule_name: str, max_workers: int) -> pl.DataFrame:
        if rule_name not in self.rules: return pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])
        return self.rules[rule_name](max_workers)

    def run_all(self, parallel=True, max_workers=8) -> dict:
        res = {}
        for rule_name in self.rules.keys():
            df = self.run_rule(rule_name, max_workers=max_workers if parallel else 1)
            if not df.is_empty():
                res[rule_name] = df
        return res

class HybridBlocking:
    def __init__(self, df: pl.DataFrame, conn_str: str, view_name: str, record_id_col: str = "RecordID"):
        self.trino_blocker = TrinoBlocking(conn_str, view_name, record_id_col)
        self.polars_blocker = PolarsBlocking(df, record_id_col)
        self.existing_pairs = set()

    def run_all(self, parallel=True, max_workers=8) -> dict:
        logging.info("Starting Trino exact blocking...")
        trino_results = self.trino_blocker.run_all(parallel, max_workers)
        
        # Deduplicate pairs found by Trino
        if trino_results:
            combined_trino = pl.concat(list(trino_results.values()))
            for row in combined_trino.iter_rows():
                self.existing_pairs.add(tuple(sorted(row[:2])))
        
        logging.info(f"Found {len(self.existing_pairs)} exact pairs. Starting Polars fuzzy blocking...")
        polars_results = self.polars_blocker.run_all(parallel, max_workers)

        # Filter out pairs already found by Trino
        final_polars_results = {}
        for rule, df in polars_results.items():
            new_pairs = []
            for row in df.iter_rows(named=True):
                pair = tuple(sorted([row["RecordID1"], row["RecordID2"]]))
                if pair not in self.existing_pairs:
                    self.existing_pairs.add(pair)
                    new_pairs.append(row)
            if new_pairs:
                final_polars_results[rule] = pl.from_dicts(new_pairs)

        logging.info(f"Found {len(self.existing_pairs) - len(trino_results)} new fuzzy pairs.")
        return {**trino_results, **final_polars_results}
    
    def merge_all(self, parallel=True, max_workers=8) -> pl.DataFrame:
        per_rule = self.run_all(parallel=parallel, max_workers=max_workers)
        if not per_rule:
            return pl.DataFrame(schema=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pl.concat(list(per_rule.values()))
        return combined.group_by(["RecordID1", "RecordID2"]).agg(RulesUsed=pl.col("Rule").unique().sort().str.join(","))

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs: return None, None
        stats = []
        for rule, df in per_rule_dfs.items():
            if df.is_empty(): continue
            pairs = len(df)
            unique_records = len(set(df["RecordID1"].to_list() + df["RecordID2"].to_list()))
            stats.append((rule, pairs, unique_records))
        
        if not stats: return None, None
        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords"])
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()
        stats_df = stats_df.sort_values("Pairs", ascending=False).head(show_top_n)
        
        chart = alt.Chart(stats_df).mark_bar().encode(
            x=alt.X("Rule:N", sort="-y", title="Blocking Rule"),
            y=alt.Y("Pairs:Q", title="Unique Pairs"),
            tooltip=["Rule:N", "Pairs:Q", "PairsPct:Q"],
            color=alt.Color("Pairs:Q", scale=alt.Scale(scheme="viridis"))
        ).properties(title="Unique Pairs by Rule", width=600, height=400)
        
        if save_html_path: chart.save(save_html_path)
        return stats_df, chart

class BlockingFactory:
    @staticmethod
    def auto_create(df: Optional[pl.DataFrame] = None, conn_str: Optional[str] = None, view_name: Optional[str] = None, record_id_col: str = "RecordID"):
        if df is not None and conn_str and view_name:
            return HybridBlocking(df, conn_str, view_name, record_id_col)
        raise ValueError("Provide Polars DataFrame, conn_str, and view_name for HybridBlocking")