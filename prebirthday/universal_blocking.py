# universal_blocking.py
import re
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import jellyfish
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import altair as alt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from difflib import SequenceMatcher
import networkx as nx
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import gc
import warnings
warnings.filterwarnings('ignore')

# Performance optimizations
pd.set_option('mode.chained_assignment', None)

# -------------------------
# Splink-style Comparison Functions
# -------------------------
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)

def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_similarity(s1, s2)

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity (1 - normalized distance)."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = jellyfish.levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)

def exact_match(s1: str, s2: str) -> bool:
    """Check if two strings are exactly equal (case-insensitive)."""
    if pd.isna(s1) or pd.isna(s2):
        return pd.isna(s1) and pd.isna(s2)
    return str(s1).strip().lower() == str(s2).strip().lower()

def fuzzy_match(s1: str, s2: str, threshold: float = 0.8, method: str = "jaro_winkler") -> bool:
    """Check if two strings match above a threshold using specified method."""
    if pd.isna(s1) or pd.isna(s2):
        return False
    
    s1_clean = str(s1).strip()
    s2_clean = str(s2).strip()
    
    if method == "jaro_winkler":
        similarity = jaro_winkler_similarity(s1_clean, s2_clean)
    elif method == "jaro":
        similarity = jaro_similarity(s1_clean, s2_clean)
    elif method == "levenshtein":
        similarity = levenshtein_similarity(s1_clean, s2_clean)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return similarity >= threshold

def normalize_date(date_value, dayfirst=True):
    """Normalize date values with support for multiple formats."""
    if pd.isna(date_value):
        return None
    
    if isinstance(date_value, str):
        date_str = str(date_value).strip()
        if not date_str:
            return None
        
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',      # 2023-12-25
            '%d-%m-%Y',      # 25-12-2023
            '%d/%m/%Y',      # 25/12/2023
            '%m/%d/%Y',      # 12/25/2023
            '%Y/%m/%d',      # 2023/12/25
            '%d.%m.%Y',      # 25.12.2023
            '%Y.%m.%d',      # 2023.12.25
            '%d %m %Y',      # 25 12 2023
            '%Y %m %d',      # 2023 12 25
        ]
        
        # Try format-based parsing first
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Fallback to pandas auto-parsing
        try:
            return pd.to_datetime(date_str, dayfirst=dayfirst, errors='coerce')
        except:
            return None
    
    # If already a datetime object, return as is
    elif isinstance(date_value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(date_value)
    
    return None

# __all__ will be defined at the end of the file after all classes are defined

# -------------------------
# Helpers
# -------------------------
def normalize_col(col: str) -> str:
    return re.sub(r'[^a-z0-9]', '', col.lower())

ATTRIBUTE_PATTERNS = {
    "first_name": ["first", "fname", "given"],
    "last_name": ["last", "lname", "surname", "family"],
    "customer_name": ["name", "fullname", "custname"],
    "dob": ["dob", "birth", "dateofbirth"],
    "year_of_birth": ["yob", "birthyear"],
    "month_of_birth": ["mob", "birthmonth"],
    "email": ["email", "mail"],
    "phone": ["phone", "mobile", "tel", "contact"],
    "passport": ["passport"],
    "national_id": ["nid", "nationalid", "ssn", "aadhar"],
    "street": ["street", "st"],
    "house_number": ["houseno", "house", "building"],
    "city": ["city", "town"],
    "postcode": ["postcode", "zip", "postal"],
    "time": ["time", "timestamp", "created", "updated", "eventtime"]
}

def match_attribute(col_name: str) -> str:
    norm = normalize_col(col_name)
    for attr, keywords in ATTRIBUTE_PATTERNS.items():
        for kw in keywords:
            if kw in norm:
                return attr
    return None

# -------------------------
# Base class
# -------------------------
class BaseBlocking:
    def run_rule(self, rule_name: str) -> pd.DataFrame:
        raise NotImplementedError
    def run_all(self, **kwargs) -> dict:
        raise NotImplementedError
    def merge_all(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

# -------------------------
# SQL Blocking
# -------------------------
class SQLBlocking(BaseBlocking):
    def __init__(self, conn_str, view_name, record_id_col="RecordID", dialect="mysql"):
        self.conn_str = conn_str
        self.engine = create_engine(conn_str)
        self.view = view_name
        self.id_col = record_id_col
        self.dialect = dialect.lower()
        self.fn_map, self.soundex_supported, self.quote_style = self._dialect_config()
        self.available_columns = self._get_view_columns()
        self.attr_map = {}
        for col in self.available_columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        # Initialize rules dictionary
        self.rules = self._generate_all_rules()
        # Speed knobs
        self.fast_stats_only: bool = True  # compute counts via aggregates, do not materialize all pairs
        self.max_pairs_per_rule: int | None = None  # optional cap if materializing pairs

    def _dialect_config(self):
        if self.dialect == "mysql":
            return {"lower":"LOWER","trim":"TRIM","substr":"SUBSTRING","instr":"INSTR","replace":"REPLACE"}, True, "`"
        elif self.dialect in ["postgresql","postgres"]:
            return {"lower":"LOWER","trim":"TRIM","substr":"SUBSTRING","instr":"POSITION","replace":"REPLACE"}, True, '"'
        else:  # hive, trino
            return {"lower":"LOWER","trim":"TRIM","substr":"SUBSTR","instr":"POSITION","replace":"REPLACE"}, False, "`"

    def _get_view_columns(self):
        with self.engine.connect() as conn:
            try:
                insp = inspect(self.engine)
                cols = [c["name"] for c in insp.get_columns(self.view)]
                if cols:
                    return cols
            except Exception:
                try:
                    res = conn.execute(text(f"SELECT * FROM {self.view} LIMIT 0"))
                    return list(res.keys())
                except Exception:
                    return []
        return []

    def q(self, col): 
        return f"{self.quote_style}{col}{self.quote_style}"

    def run_rule(self, rule_name: str, sql: str) -> pd.DataFrame:
        try:
            engine = create_engine(self.conn_str)
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
            if df.empty:
                return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
            df = df.iloc[:, :2].copy()
            df.columns = ["RecordID1", "RecordID2"]
            df["Rule"] = rule_name
            return df
        except Exception as e:
            print(f"⚠️ Skipping {rule_name}: {e}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    # ---------- FAST COUNT PATH (no pair materialization) ----------
    def _rule_key_expr(self, attr: str) -> str | None:
        """Return a SQL expression that extracts the blocking key for a given attribute.
        The expression should be usable in SELECT/GROUP BY. Returns None if attr not available."""
        fn = self.fn_map
        qcol = lambda a: self.q(self.attr_map[a]) if a in self.attr_map else None
        if attr == "email" and qcol("email"):
            c = qcol("email")
            return f"{fn['lower']}({fn['trim']}({c}))"
        if attr == "email_domain" and qcol("email"):
            c = qcol("email")
            if self.dialect in ["postgresql", "postgres"]:
                # substring from position('@' in email)+1
                return f"SUBSTRING({c} FROM POSITION('@' IN {c})+1)"
            else:
                # mysql / hive / trino style
                return f"{fn['substr']}({c},{fn['instr']}({c},'@')+1)"
        if attr == "passport" and qcol("passport"):
            c = qcol("passport")
            return f"{fn['replace']}({fn['trim']}({c}),' ','')"
        if attr == "national_id" and qcol("national_id"):
            c = qcol("national_id")
            return f"{fn['replace']}({fn['trim']}({c}),'-','')"
        if attr == "phone" and qcol("phone"):
            c = qcol("phone")
            return f"{fn['replace']}({fn['replace']}({fn['replace']}({fn['trim']}({c}),' ',''),'-',''),'(', '')"
        if attr == "postcode" and qcol("postcode"):
            c = qcol("postcode")
            return f"{fn['lower']}({fn['trim']}({c}))"
        if attr == "street_house2" and qcol("street") and qcol("house_number"):
            s, h = self.q(self.attr_map["street"]), self.q(self.attr_map["house_number"])
            return f"CONCAT({fn['lower']}({fn['trim']}({s})), '::', {fn['substr']}({fn['replace']}({fn['trim']}({h}),' ',''),1,2))"
        if attr == "street_city" and qcol("street") and qcol("city") and self.soundex_supported:
            s, c = self.q(self.attr_map["street"]), self.q(self.attr_map["city"])
            return f"CONCAT(SOUNDEX({s}), '::', {fn['lower']}({fn['trim']}({c})))"
        if attr == "last_firstchar" and qcol("last_name") and qcol("first_name"):
            l, f = self.q(self.attr_map["last_name"]), self.q(self.attr_map["first_name"])
            return f"CONCAT({fn['lower']}({fn['trim']}({l})), '::', {fn['substr']}({fn['lower']}({fn['trim']}({f})),1,1))"
        if attr == "prefix2_name" and qcol("last_name") and qcol("first_name"):
            l, f = self.q(self.attr_map["last_name"]), self.q(self.attr_map["first_name"])
            return f"CONCAT({fn['substr']}({fn['lower']}({fn['trim']}({f})),1,2), '::', {fn['substr']}({fn['lower']}({fn['trim']}({l})),1,2))"
        if attr == "soundex_name" and qcol("last_name") and qcol("first_name") and self.soundex_supported:
            l, f = self.q(self.attr_map["last_name"]), self.q(self.attr_map["first_name"])
            return f"CONCAT(SOUNDEX({f}), '::', SOUNDEX({l}))"
        if attr == "dob" and qcol("dob"):
            return self.q(self.attr_map["dob"])
        if attr == "yob" and qcol("year_of_birth"):
            return self.q(self.attr_map["year_of_birth"])
        if attr == "monthyear" and qcol("month_of_birth") and qcol("year_of_birth"):
            m, y = self.q(self.attr_map["month_of_birth"]), self.q(self.attr_map["year_of_birth"])
            return f"CONCAT({m},'::',{y})"
        if attr == "time" and qcol("time"):
            return self.q(self.attr_map["time"])
        return None

    def _pair_count_sql(self, key_expr: str) -> str:
        """Return SQL that computes total candidate pairs from grouped counts using n*(n-1)/2."""
        if self.dialect in ["postgresql", "postgres"]:
            return (
                f"SELECT COALESCE(SUM(cnt*(cnt-1)/2),0)::bigint AS pairs "
                f"FROM (SELECT {key_expr} AS k, COUNT(*) AS cnt FROM {self.view} WHERE {key_expr} IS NOT NULL GROUP BY 1) t"
            )
        else:
            return (
                f"SELECT CAST(COALESCE(SUM(cnt*(cnt-1)/2),0) AS UNSIGNED) AS pairs "
                f"FROM (SELECT {key_expr} AS k, COUNT(*) AS cnt FROM {self.view} WHERE {key_expr} IS NOT NULL GROUP BY 1) t"
            )

    def run_all_counts(self) -> pd.DataFrame:
        """Compute candidate pair counts per rule very fast using aggregates only."""
        rule_to_attr = {
            "Email": "email",
            "Passport": "passport",
            "NationalID": "national_id",
            "Phone": "phone",
            "Postcode": "postcode",
            "Street_House2": "street_house2",
            "Street_City": "street_city",
            "SoundexName": "soundex_name",
            "Prefix2Name": "prefix2_name",
            "Last_FirstChar": "last_firstchar",
            "DOB": "dob",
            "YOB": "yob",
            "MonthYear": "monthyear",
            "TimeExact": "time",
        }

        rows = []
        with self.engine.connect() as conn:
            # precompute total rows for normalization
            total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {self.view}")).scalar() or 0
            for rule, attr in rule_to_attr.items():
                key_expr = self._rule_key_expr(attr)
                if not key_expr:
                    continue
                try:
                    sql = self._pair_count_sql(key_expr)
                    pairs = conn.execute(text(sql)).scalar() or 0
                    rows.append((rule, int(pairs), total_rows))
                except Exception as e:
                    print(f"⚠️ Skipping {rule}: {e}")
                    continue
        if not rows:
            return pd.DataFrame(columns=["Rule", "Pairs", "TotalRows", "PairsPct"])
        df = pd.DataFrame(rows, columns=["Rule", "Pairs", "TotalRows"]).sort_values("Pairs", ascending=False)
        total_pairs = df["Pairs"].sum() or 1
        df["PairsPct"] = df["Pairs"] * 100.0 / total_pairs
        return df

    def _generate_all_rules(self):
        fn = self.fn_map; idc = self.q(self.id_col); rules = {}
        qcol = lambda a: self.q(self.attr_map[a]) if a in self.attr_map else None

        # ---- Identifiers ----
        if qcol("email"):
            c = qcol("email")
            # Only exact normalized email match; domain-based rule removed
            rules["Email"] = f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['lower']}({fn['trim']}(r1.{c}))={fn['lower']}({fn['trim']}(r2.{c})) WHERE r1.{idc}<r2.{idc}"
        if qcol("passport"):
            c = qcol("passport")
            rules["Passport"] = f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['replace']}({fn['trim']}(r1.{c}),' ','')={fn['replace']}({fn['trim']}(r2.{c}),' ','') WHERE r1.{idc}<r2.{idc}"
        if qcol("national_id"):
            c = qcol("national_id")
            rules["NationalID"] = f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['replace']}({fn['trim']}(r1.{c}),'-','')={fn['replace']}({fn['trim']}(r2.{c}),'-','') WHERE r1.{idc}<r2.{idc}"
        if qcol("phone"):
            c = qcol("phone")
            norm1=f"{fn['replace']}({fn['replace']}({fn['replace']}({fn['trim']}(r1.{c}),' ',''),'-',''),'(', '')"
            norm2=f"{fn['replace']}({fn['replace']}({fn['replace']}({fn['trim']}(r2.{c}),' ',''),'-',''),'(', '')"
            rules["Phone"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {norm1}={norm2} WHERE r1.{idc}<r2.{idc}"

        # ---- Names ----
        if qcol("first_name") and qcol("last_name"):
            f,l=qcol("first_name"),qcol("last_name")
            if self.soundex_supported:
                rules["SoundexName"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON SOUNDEX(r1.{f})=SOUNDEX(r2.{f}) AND SOUNDEX(r1.{l})=SOUNDEX(r2.{l}) WHERE r1.{idc}<r2.{idc}"
            rules["Prefix2Name"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['substr']}({fn['lower']}({fn['trim']}(r1.{f})),1,2)={fn['substr']}({fn['lower']}({fn['trim']}(r2.{f})),1,2) AND {fn['substr']}({fn['lower']}({fn['trim']}(r1.{l})),1,2)={fn['substr']}({fn['lower']}({fn['trim']}(r2.{l})),1,2) WHERE r1.{idc}<r2.{idc}"
            rules["Last_FirstChar"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['lower']}({fn['trim']}(r1.{l}))={fn['lower']}({fn['trim']}(r2.{l})) AND {fn['substr']}({fn['lower']}({fn['trim']}(r1.{f})),1,1)={fn['substr']}({fn['lower']}({fn['trim']}(r2.{f})),1,1) WHERE r1.{idc}<r2.{idc}"

        # ---- Address ----
        if qcol("postcode"):
            c=qcol("postcode")
            rules["Postcode"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['lower']}({fn['trim']}(r1.{c}))={fn['lower']}({fn['trim']}(r2.{c})) WHERE r1.{idc}<r2.{idc}"
        if qcol("street") and qcol("house_number"):
            s,h=qcol("street"),qcol("house_number")
            rules["Street_House2"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['lower']}({fn['trim']}(r1.{s}))={fn['lower']}({fn['trim']}(r2.{s})) AND {fn['substr']}({fn['replace']}({fn['trim']}(r1.{h}),' ',''),1,2)={fn['substr']}({fn['replace']}({fn['trim']}(r2.{h}),' ',''),1,2) WHERE r1.{idc}<r2.{idc}"
        if qcol("street") and qcol("city") and self.soundex_supported:
            s,c=qcol("street"),qcol("city")
            rules["Street_City"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON SOUNDEX(r1.{s})=SOUNDEX(r2.{s}) AND {fn['lower']}({fn['trim']}(r1.{c}))={fn['lower']}({fn['trim']}(r2.{c})) WHERE r1.{idc}<r2.{idc}"

        # ---- Dates/Times ----
        if qcol("dob"):
            c=qcol("dob")
            rules["DOB"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON r1.{c}=r2.{c} WHERE r1.{idc}<r2.{idc}"
        if qcol("year_of_birth"):
            c=qcol("year_of_birth")
            rules["YOB"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON r1.{c}=r2.{c} WHERE r1.{idc}<r2.{idc}"
        if qcol("month_of_birth") and qcol("year_of_birth"):
            m,y=qcol("month_of_birth"),qcol("year_of_birth")
            rules["MonthYear"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON r1.{m}=r2.{m} AND r1.{y}=r2.{y} WHERE r1.{idc}<r2.{idc}"
        if qcol("time"):
            c=qcol("time")
            rules["TimeExact"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON r1.{c}=r2.{c} WHERE r1.{idc}<r2.{idc}"

        # ---- Combination ----
        if qcol("last_name") and qcol("first_name") and qcol("dob") and self.soundex_supported:
            l,f,d=qcol("last_name"),qcol("first_name"),qcol("dob")
            rules["Last_First_DOB"]=f"SELECT r1.{idc}, r2.{idc} FROM {self.view} r1 JOIN {self.view} r2 ON {fn['lower']}({fn['trim']}(r1.{l}))={fn['lower']}({fn['trim']}(r2.{l})) AND SOUNDEX(r1.{f})=SOUNDEX(r2.{f}) AND r1.{d}=r2.{d} WHERE r1.{idc}<r2.{idc}"

        return rules

    def run_all(self, parallel: bool = True, max_workers: int = None, rules_to_run: list = None) -> dict:
        all_rules = self._generate_all_rules()
        if rules_to_run:
            all_rules = {k: v for k, v in all_rules.items() if k in rules_to_run}
        results = {}

        if not all_rules:
            return results

        if parallel:
            suggested = (os.cpu_count() or 2) * 5
            max_workers = max_workers or min(32, suggested)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.run_rule, name, sql): name for name, sql in all_rules.items()}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        df = fut.result()
                        if not df.empty:
                            results[name] = df
                    except Exception as e:
                        print(f"⚠️ Rule {name} failed in thread: {e}")
        else:
            for name, sql in all_rules.items():
                df = self.run_rule(name, sql)
                if not df.empty:
                    results[name] = df

        return results

    def merge_all(self, parallel: bool = True, max_workers: int = None, rules_to_run: list = None) -> pd.DataFrame:
        per_rule = self.run_all(parallel=parallel, max_workers=max_workers, rules_to_run=rules_to_run)
        if not per_rule:
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        agg = agg.rename(columns={"Rule": "RulesUsed"})
        return agg

    def generate_rule_report(self, per_rule_dfs: dict | None = None, show_top_n: int = 10, save_html_path: str = None):
        """Create analytical chart. If per_rule_dfs is None, uses fast aggregate counts.
        The chart is a diverging horizontal bar similar to model weight viz: green (PairsPct),
        red (residual to 100%)."""
        if per_rule_dfs is None:
            stats_df = self.run_all_counts()
            if stats_df.empty:
                print("No rule results to report.")
                return None
            # derive fields used previously for compatibility
            stats_df["UniqueRecords"] = None
            stats_df["AvgBlockSize"] = None
        else:
            if not per_rule_dfs:
                print("No rule results to report.")
                return None
            stats = []
            for rule, df in per_rule_dfs.items():
                if df.empty:
                    continue
                pairs = len(df)
                unique_records = pd.unique(df[["RecordID1", "RecordID2"]].values.ravel()).size
                avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
                stats.append((rule, pairs, unique_records, avg_block_size))
            if not stats:
                print("No candidate pairs found by any rule.")
                return None
            stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"]).sort_values("Pairs", ascending=False).reset_index(drop=True)
            stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        print("\n=== Blocking Rules Summary ===")
        print(stats_df.to_string(index=False, float_format="%.2f"))

        top = stats_df.head(show_top_n).copy()
        top["Residual"] = 100.0 - top["PairsPct"].fillna(0)
        top["NegResidual"] = -top["Residual"]

        # Diverging horizontal bar: residual (red, negative) + pairs (green, positive)
        base = alt.Chart(top).transform_calculate(
            pairs_str = "format(datum.PairsPct, '.2f') + '%'",
            residual_str = "format(datum.Residual, '.2f') + '%'"
        )
        left = base.mark_bar().encode(
            x=alt.X("NegResidual:Q", title="Match coverage (%)", scale=alt.Scale(domain=[-100, 100])),
            y=alt.Y("Rule:N", sort="-x"),
            color=alt.Color("NegResidual:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Rule:N", alt.Tooltip("PairsPct:Q", title="Coverage %", format=".2f"), alt.Tooltip("Pairs:Q", format=",.0f")]
        )
        right = base.mark_bar().encode(
            x="PairsPct:Q",
            y="Rule:N",
            color=alt.Color("PairsPct:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["Rule:N", alt.Tooltip("PairsPct:Q", title="Coverage %", format=".2f"), alt.Tooltip("Pairs:Q", format=",.0f")]
        )
        # dashed zero-line
        zero = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(strokeDash=[4,4], color="#555").encode(x="x:Q")
        chart = (left + right + zero).properties(title="Blocking Rule Coverage (diverging)", width=800, height=300 + 20*len(top))

        if save_html_path:
            chart.save(save_html_path)
            print(f"Saved interactive plot to {save_html_path}")
        return stats_df, chart
    def merge_all_sql(self, rules_to_run: list[str] = None) -> pd.DataFrame:
        """Generate all candidate pairs from selected rules via fast SQL pushdown.
        Returns pairs with RecordID_l < RecordID_r to avoid duplicates."""
        rules_to_run = rules_to_run or list(self.rules.keys())
        union_parts = []
        for rule in rules_to_run:
            if rule in self.rules:
                union_parts.append(self.rules[rule])
        
        if not union_parts:
            print("⚠️ No valid rules selected for merge_all_sql.")
            return pd.DataFrame(columns=["RecordID_l", "RecordID_r"])
        
        sql = "\nUNION\n".join(union_parts)
        
        print("🚀 Running merge_all_sql query...")
        print(sql)  # Debug print
        
        # Use context manager with explicit rollback for robustness
        engine = create_engine(self.conn_str)  # Recreate engine if needed
        try:
            with engine.connect() as conn:
                # Explicit rollback to clear any invalid transaction
                try:
                    conn.rollback()
                except Exception as rb_err:
                    print(f"⚠️ Rollback warning: {rb_err}")
                return pd.read_sql(sql, conn)
        except Exception as e:
            print(f"❌ Error executing merge_all_sql: {e}")
            # Attempt final rollback
            with engine.connect() as conn:
                conn.rollback()
            raise


# -------------------------
# Pandas + Mongo Blocking (stubs)
# -------------------------
class PandasBlocking(BaseBlocking):
    def __init__(self, df, record_id_col="RecordID"):
        self.df = df.copy()
        self.id_col = record_id_col
        self.attr_map = {}
        for col in df.columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
    def run_all(self): return {}
    def merge_all(self): return pd.DataFrame()

class MongoBlocking(BaseBlocking):
    def __init__(self, conn_str, collection, record_id_col="_id"):
        from pymongo import MongoClient
        self.client = MongoClient(conn_str)
        self.db = self.client.get_default_database()
        self.collection = self.db[collection]
        self.id_col = record_id_col

    def _group_pairs_by_field(self, field: str) -> pd.DataFrame:
        pipeline = [
            {"$match": {field: {"$ne": None}}},
            {"$group": {"_id": f"${field}", "ids": {"$push": f"${self.id_col}"}}},
        ]
        groups = list(self.collection.aggregate(pipeline))
        pairs = []
        for g in groups:
            ids = g.get("ids", [])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((str(ids[i]), str(ids[j])))
        if not pairs:
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        df = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2"])
        df["Rule"] = f"Mongo_{field}"
        return df

    def run_rule(self, rule_name: str) -> pd.DataFrame:
        # Example: only Email implemented here
        if rule_name == "Email":
            return self._group_pairs_by_field("Email")
        return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def run_all(self) -> dict:
        res = {}
        for r in ["Email"]:  # extendable
            df = self.run_rule(r)
            if not df.empty:
                res[r] = df
        return res

    def merge_all(self) -> pd.DataFrame:
        all_res = self.run_all()
        if not all_res:
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pd.concat(all_res.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"] \
            .apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        agg = agg.rename(columns={"Rule": "RulesUsed"})
        return agg

# -------------------------
# Factory
# -------------------------
class BlockingFactory:
    @staticmethod
    def auto_create(conn_str=None, view_name=None, record_id_col="RecordID", df=None, collection=None, 
                   enhanced=False, high_performance=False, comparison_thresholds=None, 
                   chunk_size=10000, max_workers=None):
        # Case 1: Pandas DataFrame
        if df is not None:
            if high_performance:
                return HighPerformanceBlocking(df, record_id_col, comparison_thresholds, 
                                             chunk_size, max_workers)
            elif enhanced:
                return EnhancedPandasBlocking(df, record_id_col, comparison_thresholds)
            else:
                return PandasBlocking(df, record_id_col)

        # Case 2: MongoDB
        if conn_str and conn_str.lower().startswith("mongo"):
            if not collection:
                raise ValueError("collection required for Mongo backend")
            return MongoBlocking(conn_str, collection, record_id_col)

        # Case 3: SQL
        if conn_str:
            # Detect SQL dialect
            dialect = conn_str.split(":")[0].lower()
            if not view_name:
                raise ValueError("view_name required for SQL backend")
            return SQLBlocking(conn_str, view_name, record_id_col, dialect=dialect)

        raise ValueError("Could not auto-detect backend. Provide conn_str, df, or Mongo conn_str+collection.")

# -------------------------
# High-Performance Blocking for Massive Datasets
# -------------------------
class HighPerformanceBlocking(BaseBlocking):
    def __init__(self, df, record_id_col="RecordID", comparison_thresholds=None, 
                 chunk_size=10000, max_workers=None, use_multiprocessing=True):
        self.df = df.copy()
        self.id_col = record_id_col
        self.chunk_size = chunk_size
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        # Attribute mapping
        self.attr_map = {}
        for col in df.columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        
        # Default comparison thresholds (Splink-style)
        self.thresholds = comparison_thresholds or {
            "jaro_winkler": {"first_name": 0.85, "last_name": 0.85, "email": 0.9},
            "jaro": {"first_name": 0.75, "last_name": 0.75},
            "levenshtein": {"phone": 0.85, "postcode": 0.9}
        }

    def _chunk_dataframe(self, df, chunk_size=None):
        """Split dataframe into chunks for parallel processing."""
        chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size].copy()

    def _process_chunk(self, chunk_df, rule_name, comparison_func, **kwargs):
        """Process a single chunk of data."""
        pairs = []
        ids = chunk_df[self.id_col].tolist()
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                row1 = chunk_df[chunk_df[self.id_col] == id1].iloc[0]
                row2 = chunk_df[chunk_df[self.id_col] == id2].iloc[0]
                
                if comparison_func(row1, row2, **kwargs):
                    pairs.append((id1, id2, rule_name))
        
        return pairs

    def _parallel_blocking(self, df, rule_name, comparison_func, **kwargs):
        """Run blocking in parallel across chunks."""
        all_pairs = []
        
        if self.use_multiprocessing and len(df) > self.chunk_size:
            # Use multiprocessing for large datasets
            chunks = list(self._chunk_dataframe(df, self.chunk_size))
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                process_func = partial(self._process_chunk, 
                                     rule_name=rule_name, 
                                     comparison_func=comparison_func, 
                                     **kwargs)
                futures = [executor.submit(process_func, chunk) for chunk in chunks]
                
                for future in as_completed(futures):
                    try:
                        chunk_pairs = future.result()
                        all_pairs.extend(chunk_pairs)
                    except Exception as e:
                        print(f"⚠️ Chunk processing error: {e}")
        else:
            # Use threading for smaller datasets
            chunks = list(self._chunk_dataframe(df, self.chunk_size))
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                process_func = partial(self._process_chunk, 
                                     rule_name=rule_name, 
                                     comparison_func=comparison_func, 
                                     **kwargs)
                futures = [executor.submit(process_func, chunk) for chunk in chunks]
                
                for future in as_completed(futures):
                    try:
                        chunk_pairs = future.result()
                        all_pairs.extend(chunk_pairs)
                    except Exception as e:
                        print(f"⚠️ Chunk processing error: {e}")
        
        return all_pairs

    def _compare_names_fast(self, row1, row2, method="jaro_winkler", threshold=0.9):
        """Fast name comparison with caching."""
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f1 = str(row1[self.attr_map["first_name"]]).strip()
            f2 = str(row2[self.attr_map["first_name"]]).strip()
            l1 = str(row1[self.attr_map["last_name"]]).strip()
            l2 = str(row2[self.attr_map["last_name"]]).strip()
            
            if pd.isna(f1) or pd.isna(f2) or pd.isna(l1) or pd.isna(l2):
                return False
            
            # Fast exact match check first
            if f1 == f2 and l1 == l2:
                return True
            
            # Use fuzzy matching only if needed
            f_match = fuzzy_match(f1, f2, threshold, method)
            l_match = fuzzy_match(l1, l2, threshold, method)
            return f_match and l_match
        return False

    def _compare_email_fast(self, row1, row2, threshold=0.95):
        """Fast email comparison."""
        if "email" not in self.attr_map:
            return False
        
        e1 = str(row1[self.attr_map["email"]]).strip().lower()
        e2 = str(row2[self.attr_map["email"]]).strip().lower()
        
        if pd.isna(e1) or pd.isna(e2):
            return False
        
        # Fast exact match first
        if e1 == e2:
            return True
        
        return fuzzy_match(e1, e2, threshold, "jaro_winkler")

    def _compare_phone_fast(self, row1, row2, threshold=0.9):
        """Fast phone comparison with normalization."""
        if "phone" not in self.attr_map:
            return False
        
        p1 = re.sub(r'[^\d]', '', str(row1[self.attr_map["phone"]]))
        p2 = re.sub(r'[^\d]', '', str(row2[self.attr_map["phone"]]))
        
        if not p1 or not p2:
            return False
        
        # Fast exact match first
        if p1 == p2:
            return True
        
        return fuzzy_match(p1, p2, threshold, "levenshtein")

    def run_all(self) -> dict:
        """Run all enhanced blocking rules with high performance."""
        res = {}
        df = self.df.copy()
        df[self.id_col] = df[self.id_col].astype(str)
        
        print(f"🚀 Processing {len(df)} records with {self.max_workers} workers...")

        # Enhanced Name Matching
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            print("🔍 Running Jaro-Winkler name matching...")
            threshold = self.thresholds["jaro_winkler"].get("first_name", 0.9)
            pairs = self._parallel_blocking(
                df, "JaroWinkler_Names", 
                self._compare_names_fast, 
                method="jaro_winkler", 
                threshold=threshold
            )
            if pairs:
                res["JaroWinkler_Names"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

            print("🔍 Running Jaro name matching...")
            threshold = self.thresholds["jaro"].get("first_name", 0.8)
            pairs = self._parallel_blocking(
                df, "Jaro_Names", 
                self._compare_names_fast, 
                method="jaro", 
                threshold=threshold
            )
            if pairs:
                res["Jaro_Names"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Enhanced Email Matching
        if "email" in self.attr_map:
            print("🔍 Running email matching...")
            threshold = self.thresholds["jaro_winkler"].get("email", 0.95)
            pairs = self._parallel_blocking(
                df, "Fuzzy_Email", 
                self._compare_email_fast, 
                threshold=threshold
            )
            if pairs:
                res["Fuzzy_Email"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Enhanced Phone Matching
        if "phone" in self.attr_map:
            print("🔍 Running phone matching...")
            threshold = self.thresholds["levenshtein"].get("phone", 0.9)
            pairs = self._parallel_blocking(
                df, "Fuzzy_Phone", 
                self._compare_phone_fast, 
                threshold=threshold
            )
            if pairs:
                res["Fuzzy_Phone"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Clean up memory
        gc.collect()
        return res

    def merge_all(self) -> pd.DataFrame:
        """Merge all blocking results efficiently."""
        per_rule = self.run_all()
        if not per_rule:
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        
        # Efficient concatenation
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(
            lambda s: ",".join(sorted(set(s)))
        ).reset_index()
        return agg.rename(columns={"Rule": "RulesUsed"})

    def create_clusters_fast(self, pairs_df: pd.DataFrame, min_cluster_size: int = 2) -> pd.DataFrame:
        """Create clusters efficiently using optimized network analysis."""
        if pairs_df.empty:
            return pd.DataFrame(columns=["RecordID", "ClusterID"])
        
        print(f"🔗 Creating clusters from {len(pairs_df)} pairs...")
        
        # Use efficient graph creation
        G = nx.Graph()
        
        # Add edges in batches for memory efficiency
        batch_size = 10000
        for i in range(0, len(pairs_df), batch_size):
            batch = pairs_df.iloc[i:i + batch_size]
            edges = [(row["RecordID1"], row["RecordID2"]) for _, row in batch.iterrows()]
            G.add_edges_from(edges)
            
            # Clear batch from memory
            del batch
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        # Create cluster mapping efficiently
        cluster_data = []
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) >= min_cluster_size:
                for record_id in cluster:
                    cluster_data.append({
                        "RecordID": record_id,
                        "ClusterID": f"cluster_{cluster_id}",
                        "ClusterSize": len(cluster)
                    })
        
        # Clear graph from memory
        del G
        gc.collect()
        
        return pd.DataFrame(cluster_data)

    def generate_performance_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        """Generate performance-optimized visualization report."""
        if not per_rule_dfs:
            print("No rule results to report.")
            return None, None

        stats = []
        for rule, df in per_rule_dfs.items():
            if df.empty:
                continue
            pairs = len(df)
            unique_records = pd.unique(df[["RecordID1", "RecordID2"]].values.ravel()).size
            avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
            stats.append((rule, pairs, unique_records, avg_block_size))

        if not stats:
            print("No candidate pairs found by any rule.")
            return None, None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        print("\n=== High-Performance Blocking Rules Summary ===")
        print(stats_df.to_string(index=False, float_format="%.2f"))

        # Create optimized visualization
        top = stats_df.head(show_top_n)
        
        # Use simpler chart for better performance
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("Rule:N", sort="-y", title="Blocking Rule"),
                y=alt.Y("PairsPct:Q", title="Candidate Pairs (%)"),
                tooltip=[
                    alt.Tooltip("Rule:N"),
                    alt.Tooltip("Pairs:Q", format=",.0f"),
                    alt.Tooltip("PairsPct:Q", format=".2f"),
                    alt.Tooltip("UniqueRecords:Q", format=",.0f"),
                    alt.Tooltip("AvgBlockSize:Q", format=".2f"),
                ],
                color=alt.Color("PairsPct:Q", scale=alt.Scale(scheme="viridis")),
            )
            .properties(
                title="High-Performance Blocking Rules by % of Candidate Pairs", 
                width=600, 
                height=400
            )
        )

        # if save_html_path:
        #     chart.save(save_html_path)
        #     print(f"Saved performance-optimized plot to {save_html_path}")

        return stats_df, chart

# -------------------------
# Enhanced Pandas Blocking with Splink-style Comparisons
# -------------------------
class EnhancedPandasBlocking(BaseBlocking):
    def __init__(self, df, record_id_col="RecordID", comparison_thresholds=None):
        self.df = df.copy()
        self.id_col = record_id_col
        self.attr_map = {}
        for col in df.columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        
        # Default comparison thresholds (Splink-style)
        self.thresholds = comparison_thresholds or {
            "jaro_winkler": {"first_name": 0.9, "last_name": 0.9, "email": 0.95},
            "jaro": {"first_name": 0.8, "last_name": 0.8},
            "levenshtein": {"phone": 0.9, "postcode": 0.95}
        }

    def _create_candidate_pairs(self, df_subset, rule_name, comparison_func, **kwargs):
        """Create candidate pairs based on comparison function."""
        pairs = []
        ids = df_subset[self.id_col].tolist()
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                row1 = df_subset[df_subset[self.id_col] == id1].iloc[0]
                row2 = df_subset[df_subset[self.id_col] == id2].iloc[0]
                
                if comparison_func(row1, row2, **kwargs):
                    pairs.append((id1, id2, rule_name))
        
        return pairs

    def _compare_names(self, row1, row2, method="jaro_winkler", threshold=0.9):
        """Compare names using fuzzy matching."""
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f1 = str(row1[self.attr_map["first_name"]]).strip()
            f2 = str(row2[self.attr_map["first_name"]]).strip()
            l1 = str(row1[self.attr_map["last_name"]]).strip()
            l2 = str(row2[self.attr_map["last_name"]]).strip()
            
            if pd.isna(f1) or pd.isna(f2) or pd.isna(l1) or pd.isna(l2):
                return False
            
            f_match = fuzzy_match(f1, f2, threshold, method)
            l_match = fuzzy_match(l1, l2, threshold, method)
            return f_match and l_match
        return False

    def _compare_email(self, row1, row2, threshold=0.95):
        """Compare emails with normalization."""
        if "email" not in self.attr_map:
            return False
        
        e1 = str(row1[self.attr_map["email"]]).strip().lower()
        e2 = str(row2[self.attr_map["email"]]).strip().lower()
        
        if pd.isna(e1) or pd.isna(e2):
            return False
        
        return fuzzy_match(e1, e2, threshold, "jaro_winkler")

    def _compare_phone(self, row1, row2, threshold=0.9):
        """Compare phone numbers with normalization."""
        if "phone" not in self.attr_map:
            return False
        
        p1 = re.sub(r'[^\d]', '', str(row1[self.attr_map["phone"]]))
        p2 = re.sub(r'[^\d]', '', str(row2[self.attr_map["phone"]]))
        
        if not p1 or not p2:
            return False
        
        return fuzzy_match(p1, p2, threshold, "levenshtein")

    def _compare_dob(self, row1, row2, tolerance_days=0):
        """Compare dates of birth with proper date normalization."""
        if "dob" not in self.attr_map:
            return False
        
        d1 = row1[self.attr_map["dob"]]
        d2 = row2[self.attr_map["dob"]]
        
        if pd.isna(d1) or pd.isna(d2):
            return False
        
        try:
            # Use the normalize_date function for consistent parsing
            d1_normalized = normalize_date(d1, dayfirst=True)
            d2_normalized = normalize_date(d2, dayfirst=True)
            
            # Check if parsing was successful
            if pd.isna(d1_normalized) or pd.isna(d2_normalized):
                return False
            
            diff = abs((d1_normalized - d2_normalized).days)
            return diff <= tolerance_days
        except Exception as e:
            # Log the error for debugging if needed
            # print(f"Date comparison error: {e}")
            return False

    def run_all(self) -> dict:
        """Run all enhanced blocking rules with fuzzy matching."""
        res = {}
        df = self.df.copy()
        df[self.id_col] = df[self.id_col].astype(str)

        # Enhanced Name Matching
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            # Jaro-Winkler name matching
            threshold = self.thresholds["jaro_winkler"].get("first_name", 0.9)
            pairs = self._create_candidate_pairs(
                df, "JaroWinkler_Names", 
                self._compare_names, 
                method="jaro_winkler", 
                threshold=threshold
            )
            if pairs:
                res["JaroWinkler_Names"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

            # Jaro name matching (more lenient)
            threshold = self.thresholds["jaro"].get("first_name", 0.8)
            pairs = self._create_candidate_pairs(
                df, "Jaro_Names", 
                self._compare_names, 
                method="jaro", 
                threshold=threshold
            )
            if pairs:
                res["Jaro_Names"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Enhanced Email Matching
        if "email" in self.attr_map:
            threshold = self.thresholds["jaro_winkler"].get("email", 0.95)
            pairs = self._create_candidate_pairs(
                df, "Fuzzy_Email", 
                self._compare_email, 
                threshold=threshold
            )
            if pairs:
                res["Fuzzy_Email"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Enhanced Phone Matching
        if "phone" in self.attr_map:
            threshold = self.thresholds["levenshtein"].get("phone", 0.9)
            pairs = self._create_candidate_pairs(
                df, "Fuzzy_Phone", 
                self._compare_phone, 
                threshold=threshold
            )
            if pairs:
                res["Fuzzy_Phone"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        # Enhanced DOB Matching
        if "dob" in self.attr_map:
            pairs = self._create_candidate_pairs(
                df, "DOB_Exact", 
                self._compare_dob, 
                tolerance_days=0
            )
            if pairs:
                res["DOB_Exact"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

            # DOB with tolerance
            pairs = self._create_candidate_pairs(
                df, "DOB_Tolerance", 
                self._compare_dob, 
                tolerance_days=1
            )
            if pairs:
                res["DOB_Tolerance"] = pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"])

        return res

    def merge_all(self) -> pd.DataFrame:
        """Merge all blocking results."""
        per_rule = self.run_all()
        if not per_rule:
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(
            lambda s: ",".join(sorted(set(s)))
        ).reset_index()
        return agg.rename(columns={"Rule": "RulesUsed"})

    def create_clusters(self, pairs_df: pd.DataFrame, min_cluster_size: int = 2) -> pd.DataFrame:
        """Create clusters from blocking pairs using network analysis."""
        if pairs_df.empty:
            return pd.DataFrame(columns=["RecordID", "ClusterID"])
        
        # Create graph from pairs
        G = nx.Graph()
        for _, row in pairs_df.iterrows():
            G.add_edge(row["RecordID1"], row["RecordID2"])
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        # Create cluster mapping
        cluster_data = []
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) >= min_cluster_size:
                for record_id in cluster:
                    cluster_data.append({
                        "RecordID": record_id,
                        "ClusterID": f"cluster_{cluster_id}",
                        "ClusterSize": len(cluster)
                    })
        
        return pd.DataFrame(cluster_data)

    def generate_enhanced_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        """Generate enhanced visualization report."""
        if not per_rule_dfs:
            print("No rule results to report.")
            return None, None

        stats = []
        for rule, df in per_rule_dfs.items():
            if df.empty:
                continue
            pairs = len(df)
            unique_records = pd.unique(df[["RecordID1", "RecordID2"]].values.ravel()).size
            avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
            stats.append((rule, pairs, unique_records, avg_block_size))

        if not stats:
            print("No candidate pairs found by any rule.")
            return None, None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        print("\n=== Enhanced Blocking Rules Summary ===")
        print(stats_df.to_string(index=False, float_format="%.2f"))

        # Create enhanced visualization
        top = stats_df.head(show_top_n)
        
        # Main bar chart
        main_chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("Rule:N", sort="-y", title="Blocking Rule"),
                y=alt.Y("PairsPct:Q", title="Candidate Pairs (%)"),
                tooltip=[
                    alt.Tooltip("Rule:N"),
                    alt.Tooltip("Pairs:Q", format=",.0f"),
                    alt.Tooltip("PairsPct:Q", format=".2f"),
                    alt.Tooltip("UniqueRecords:Q", format=",.0f"),
                    alt.Tooltip("AvgBlockSize:Q", format=".2f"),
                ],
                color=alt.Color("PairsPct:Q", scale=alt.Scale(scheme="viridis")),
            )
            .properties(title="Enhanced Blocking Rules by % of Candidate Pairs", width=600, height=400)
        )

        # Cluster size distribution (if we have clusters)
        if "ClusterSize" in stats_df.columns:
            cluster_chart = (
                alt.Chart(stats_df)
                .mark_bar()
                .encode(
                    x=alt.X("AvgBlockSize:Q", bin=alt.Bin(maxbins=20), title="Average Block Size"),
                    y=alt.Y("count():Q", title="Number of Rules"),
                    color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"))
                )
                .properties(title="Distribution of Average Block Sizes", width=400, height=300)
            )
            
            # Combine charts
            chart = alt.hconcat(main_chart, cluster_chart)
        else:
            chart = main_chart

        # if save_html_path:
        #     chart.save(save_html_path)
        #     print(f"Saved enhanced interactive plot to {save_html_path}")

        return stats_df, chart

# -------------------------
# Pandas Blocking (full rules)
# -------------------------
class PandasBlocking(BaseBlocking):
    def __init__(self, df, record_id_col="RecordID"):
        self.df = df.copy()
        self.id_col = record_id_col
        self.attr_map = {}
        for col in df.columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col

    def _pairs_from_block(self, block_df, rule_name):
        pairs = []
        ids = block_df[self.id_col].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j], rule_name))
        return pairs

    def run_all(self) -> dict:
        res = {}
        # work on copy to normalize
        df = self.df.copy()
        df[self.id_col] = df[self.id_col].astype(str)

        # ---- Email ----
        if "email" in self.attr_map:
            col = self.attr_map["email"]
            df["_email_norm"] = df[col].astype(str).str.strip().str.lower()
            blocks = df.groupby("_email_norm")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Email")
            if pairs:
                res["Email"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

            # Email domain
            df["_email_domain"] = df[col].astype(str).str.extract(r'@(.+)$')[0].str.lower()
            blocks = df.groupby("_email_domain")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "EmailDomain")
            if pairs:
                res["EmailDomain"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- Passport ----
        if "passport" in self.attr_map:
            col = self.attr_map["passport"]
            df["_passport_norm"] = df[col].astype(str).str.replace(r"\s+","",regex=True)
            blocks = df.groupby("_passport_norm")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Passport")
            if pairs:
                res["Passport"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- National ID ----
        if "national_id" in self.attr_map:
            col = self.attr_map["national_id"]
            df["_nid_norm"] = df[col].astype(str).str.replace("-","",regex=False)
            blocks = df.groupby("_nid_norm")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "NationalID")
            if pairs:
                res["NationalID"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- Phone ----
        if "phone" in self.attr_map:
            col = self.attr_map["phone"]
            df["_phone_norm"] = df[col].astype(str).str.replace(r"[ \-\(\)]","",regex=True)
            blocks = df.groupby("_phone_norm")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Phone")
            if pairs:
                res["Phone"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- Names ----
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f, l = self.attr_map["first_name"], self.attr_map["last_name"]
            # Soundex
            df["_f_soundex"] = df[f].dropna().astype(str).apply(lambda x: jellyfish.soundex(x))
            df["_l_soundex"] = df[l].dropna().astype(str).apply(lambda x: jellyfish.soundex(x))
            blocks = df.groupby(["_f_soundex","_l_soundex"])
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "SoundexName")
            if pairs:
                res["SoundexName"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

            # Prefix 2
            df["_f_pref2"] = df[f].astype(str).str[:2].str.lower()
            df["_l_pref2"] = df[l].astype(str).str[:2].str.lower()
            blocks = df.groupby(["_f_pref2","_l_pref2"])
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Prefix2Name")
            if pairs:
                res["Prefix2Name"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

            # Last + First initial
            df["_f_initial"] = df[f].astype(str).str[0].str.lower()
            df["_l_lower"] = df[l].astype(str).str.lower()
            blocks = df.groupby(["_l_lower","_f_initial"])
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Last_FirstChar")
            if pairs:
                res["Last_FirstChar"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- Address ----
        if "postcode" in self.attr_map:
            c = self.attr_map["postcode"]
            df["_postcode"] = df[c].astype(str).str.lower().str.strip()
            blocks = df.groupby("_postcode")
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Postcode")
            if pairs:
                res["Postcode"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        if "street" in self.attr_map and "house_number" in self.attr_map:
            s,h=self.attr_map["street"], self.attr_map["house_number"]
            df["_street"] = df[s].astype(str).str.lower().str.strip()
            df["_house2"] = df[h].astype(str).str.replace(" ","").str[:2]
            blocks = df.groupby(["_street","_house2"])
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Street_House2")
            if pairs:
                res["Street_House2"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        if "street" in self.attr_map and "city" in self.attr_map:
            s,c=self.attr_map["street"], self.attr_map["city"]
            df["_street_sdx"] = df[s].astype(str).apply(lambda x: jellyfish.soundex(x))
            df["_city_lower"] = df[c].astype(str).str.lower().str.strip()
            blocks = df.groupby(["_street_sdx","_city_lower"])
            pairs = []
            for _, g in blocks:
                pairs += self._pairs_from_block(g, "Street_City")
            if pairs:
                res["Street_City"] = pd.DataFrame(pairs, columns=["RecordID1","RecordID2","Rule"])

        # ---- Dates ----
        if "dob" in self.attr_map:
            c=self.attr_map["dob"]
            blocks = df.groupby(c)
            pairs=[]
            for _,g in blocks:
                pairs+=self._pairs_from_block(g,"DOB")
            if pairs:
                res["DOB"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])

        if "year_of_birth" in self.attr_map:
            c=self.attr_map["year_of_birth"]
            blocks = df.groupby(c)
            pairs=[]
            for _,g in blocks:
                pairs+=self._pairs_from_block(g,"YOB")
            if pairs:
                res["YOB"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])

        if "month_of_birth" in self.attr_map and "year_of_birth" in self.attr_map:
            m,y=self.attr_map["month_of_birth"], self.attr_map["year_of_birth"]
            blocks = df.groupby([m,y])
            pairs=[]
            for _,g in blocks:
                pairs+=self._pairs_from_block(g,"MonthYear")
            if pairs:
                res["MonthYear"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])

        if "time" in self.attr_map:
            c=self.attr_map["time"]
            blocks = df.groupby(c)
            pairs=[]
            for _,g in blocks:
                pairs+=self._pairs_from_block(g,"TimeExact")
            if pairs:
                res["TimeExact"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])

        # ---- Combination ----
        if "last_name" in self.attr_map and "first_name" in self.attr_map and "dob" in self.attr_map:
            l,f,d=self.attr_map["last_name"],self.attr_map["first_name"],self.attr_map["dob"]
            df["_lname_lower"]=df[l].astype(str).str.lower().str.strip()
            df["_fname_sdx"]=df[f].astype(str).apply(lambda x: jellyfish.soundex(x))
            blocks = df.groupby(["_lname_lower","_fname_sdx",d])
            pairs=[]
            for _,g in blocks:
                pairs+=self._pairs_from_block(g,"Last_First_DOB")
            if pairs:
                res["Last_First_DOB"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])

        return res

    def merge_all(self) -> pd.DataFrame:
        per_rule=self.run_all()
        if not per_rule: 
            return pd.DataFrame(columns=["RecordID1","RecordID2","RulesUsed"])
        combined=pd.concat(per_rule.values(),ignore_index=True)
        agg=combined.groupby(["RecordID1","RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        return agg.rename(columns={"Rule":"RulesUsed"})

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            print("No rule results to report.")
            return None

        stats = []
        for rule, df in per_rule_dfs.items():
            if df.empty:
                continue
            pairs = len(df)
            unique_records = pd.unique(df[["RecordID1", "RecordID2"]].values.ravel()).size
            avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
            stats.append((rule, pairs, unique_records, avg_block_size))

        if not stats:
            print("No candidate pairs found by any rule.")
            return None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        print("\n=== Blocking Rules Summary ===")
        print(stats_df.to_string(index=False, float_format="%.2f"))

        top = stats_df.head(show_top_n)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("Rule:N", sort="-y", title="Blocking Rule"),
                y=alt.Y("PairsPct:Q", title="Candidate Pairs (%)"),
                tooltip=[
                    alt.Tooltip("Rule:N"),
                    alt.Tooltip("Pairs:Q", format=",.0f"),
                    alt.Tooltip("PairsPct:Q", format=".2f"),
                    alt.Tooltip("UniqueRecords:Q", format=",.0f"),
                    alt.Tooltip("AvgBlockSize:Q", format=".2f"),
                ],
                color=alt.Color("PairsPct:Q", scale=alt.Scale(scheme="blues")),
            )
            .properties(title="Top Blocking Rules by % of Candidate Pairs", width=600, height=400)
        )

        # if save_html_path:
        #     chart.save(save_html_path)
        #     print(f"Saved interactive plot to {save_html_path}")

        return stats_df, chart

# -------------------------
# Mongo Blocking (full rules, hybrid)
# -------------------------
class MongoBlocking(BaseBlocking):
    def __init__(self, conn_str, collection, record_id_col="_id"):
        from pymongo import MongoClient
        self.client = MongoClient(conn_str)
        self.db = self.client.get_default_database()
        self.collection = self.db[collection]
        self.id_col = record_id_col

    def _pairs_from_groups(self, groups, rule_name):
        pairs=[]
        for g in groups:
            ids=g.get("ids",[])
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):
                    pairs.append((str(ids[i]),str(ids[j]),rule_name))
        return pairs

    def _group_by_field(self, field, transform=None, rule_name=None):
        pipeline=[{"$match":{field:{"$ne":None}}},
                  {"$group":{"_id":f"${field}","ids":{"$push":f"${self.id_col}"}}}]
        groups=list(self.collection.aggregate(pipeline))
        return self._pairs_from_groups(groups,rule_name or field)

    def run_all(self) -> dict:
        res={}
        # Example: implement Email and Phone inside Mongo, others fallback
        if self.collection.count_documents({"Email":{"$exists":True}})>0:
            pairs=self._group_by_field("Email",rule_name="Email")
            if pairs: res["Email"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])
        if self.collection.count_documents({"Phone":{"$exists":True}})>0:
            pairs=self._group_by_field("Phone",rule_name="Phone")
            if pairs: res["Phone"]=pd.DataFrame(pairs,columns=["RecordID1","RecordID2","Rule"])
        # TODO: For soundex or complex rules, fetch documents to pandas and reuse PandasBlocking
        return res

    def merge_all(self) -> pd.DataFrame:
        per_rule=self.run_all()
        if not per_rule: return pd.DataFrame(columns=["RecordID1","RecordID2","RulesUsed"])
        combined=pd.concat(per_rule.values(),ignore_index=True)
        agg=combined.groupby(["RecordID1","RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        return agg.rename(columns={"Rule":"RulesUsed"})

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            print("No rule results to report.")
            return None

        stats = []
        for rule, df in per_rule_dfs.items():
            if df.empty:
                continue
            pairs = len(df)
            unique_records = pd.unique(df[["RecordID1", "RecordID2"]].values.ravel()).size
            avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
            stats.append((rule, pairs, unique_records, avg_block_size))

        if not stats:
            print("No candidate pairs found by any rule.")
            return None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        print("\n=== Blocking Rules Summary ===")
        print(stats_df.to_string(index=False, float_format="%.2f"))

        top = stats_df.head(show_top_n)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("Rule:N", sort="-y", title="Blocking Rule"),
                y=alt.Y("PairsPct:Q", title="Candidate Pairs (%)"),
                tooltip=[
                    alt.Tooltip("Rule:N"),
                    alt.Tooltip("Pairs:Q", format=",.0f"),
                    alt.Tooltip("PairsPct:Q", format=".2f"),
                    alt.Tooltip("UniqueRecords:Q", format=",.0f"),
                    alt.Tooltip("AvgBlockSize:Q", format=".2f"),
                ],
                color=alt.Color("PairsPct:Q", scale=alt.Scale(scheme="blues")),
            )
            .properties(title="Top Blocking Rules by % of Candidate Pairs", width=600, height=400)
        )

        if save_html_path:
            chart.save(save_html_path)
            print(f"Saved interactive plot to {save_html_path}")

        return stats_df, chart


def run_blocking_pipeline(
    conn_str=None,
    view_name=None,
    df=None,
    collection=None,
    record_id_col="RecordID",
    parallel=True,
    max_workers=8,
    show_top_n=10,
    save_html_path=None,
    enhanced=False,
    high_performance=False,
    comparison_thresholds=None,
    create_clusters=True,
    min_cluster_size=2,
    chunk_size=10000,
):
    """
    Unified pipeline for blocking with Splink-style enhancements:
      - Detects backend (SQL, Pandas, Mongo) automatically
      - Runs all blocking rules with fuzzy matching
      - Creates clusters of similar records
      - Generates enhanced summary report + Altair charts
      - Supports high-performance mode for massive datasets
    Returns: (blocker, per_rule, stats_df, chart, clusters_df)
    """
    # Step 1: auto-detect backend
    blocker = BlockingFactory.auto_create(
        conn_str=conn_str,
        view_name=view_name,
        record_id_col=record_id_col,
        df=df,
        collection=collection,
        enhanced=enhanced,
        high_performance=high_performance,
        comparison_thresholds=comparison_thresholds,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )

    # Step 2: run blocking
    if hasattr(blocker, "run_all"):
        try:
            per_rule = blocker.run_all(parallel=parallel, max_workers=max_workers)
        except TypeError:
            # for Pandas/Mongo which don't support parallel
            per_rule = blocker.run_all()
    else:
        raise RuntimeError("Blocker does not implement run_all()")

    # Step 3: generate report
    if high_performance and hasattr(blocker, "generate_performance_report"):
        result = blocker.generate_performance_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
        if result is None:
            stats_df, chart = None, None
        else:
            stats_df, chart = result
    elif enhanced and hasattr(blocker, "generate_enhanced_report"):
        result = blocker.generate_enhanced_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
        if result is None:
            stats_df, chart = None, None
        else:
            stats_df, chart = result
    elif hasattr(blocker, "generate_rule_report"):
        result = blocker.generate_rule_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
        if result is None:
            stats_df, chart = None, None
        else:
            stats_df, chart = result
    else:
        stats_df, chart = None, None

    # Step 4: create clusters if requested
    clusters_df = None
    if create_clusters and per_rule:
        merged_pairs = blocker.merge_all()
        if not merged_pairs.empty:
            if high_performance and hasattr(blocker, "create_clusters_fast"):
                clusters_df = blocker.create_clusters_fast(merged_pairs, min_cluster_size)
            elif hasattr(blocker, "create_clusters"):
                clusters_df = blocker.create_clusters(merged_pairs, min_cluster_size)
            
            if clusters_df is not None and not clusters_df.empty:
                print(f"\n🎯 Created {len(clusters_df['ClusterID'].unique())} clusters with {len(clusters_df)} records")

    return blocker, per_rule, stats_df, chart, clusters_df

def create_sample_data(n_records=100):
    """Create sample data for testing the blocking system."""
    np.random.seed(42)
    
    # Generate sample data with intentional duplicates and variations
    data = []
    base_names = [
        ("John", "Smith"), ("Jane", "Doe"), ("Michael", "Johnson"), 
        ("Sarah", "Williams"), ("David", "Brown"), ("Lisa", "Davis"),
        ("Robert", "Miller"), ("Jennifer", "Wilson"), ("William", "Moore"),
        ("Elizabeth", "Taylor")
    ]
    
    for i in range(n_records):
        # Create some duplicates and variations
        if i < len(base_names):
            first, last = base_names[i]
        else:
            first, last = base_names[i % len(base_names)]
        
        # Add variations
        if i % 3 == 0:  # Exact duplicates
            first_var, last_var = first, last
        elif i % 3 == 1:  # Minor variations
            first_var = first + " Jr" if len(first) < 6 else first
            last_var = last + "son" if len(last) < 6 else last
        else:  # Typos and variations
            first_var = first.replace('a', 'e') if 'a' in first else first
            last_var = last.replace('i', 'y') if 'i' in last else last
        
        # Generate other fields
        email = f"{first_var.lower()}.{last_var.lower()}@example.com"
        phone = f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        dob = f"{np.random.randint(1950, 2000)}-{np.random.randint(1, 12):02d}-{np.random.randint(1, 28):02d}"
        
        data.append({
            "RecordID": f"R{i:03d}",
            "first_name": first_var,
            "last_name": last_var,
            "email": email,
            "phone": phone,
            "dob": dob,
            "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
        })
    
    return pd.DataFrame(data)

# Make all classes and functions available at module level
__all__ = [
    'normalize_date', 
    'BlockingFactory', 
    'run_blocking_pipeline', 
    'EnhancedPandasBlocking',
    'HighPerformanceBlocking',
    'PandasBlocking',
    'SQLBlocking',
    'MongoBlocking',
    'create_sample_data'
]

   
