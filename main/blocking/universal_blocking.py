
import re
import os
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text, inspect
import jellyfish
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
import networkx as nx
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d.%m.%Y', '%Y.%m.%d', '%d %m %Y', '%Y %m %d'
        ]
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(date_str, dayfirst=dayfirst, errors='coerce')
        except:
            return None
    elif isinstance(date_value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(date_value)
    return None

# ------------------------- Helpers -------------------------
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

# ------------------------- Base Class -------------------------
class BaseBlocking:
    def run_rule(self, rule_name: str) -> Union[pd.DataFrame, pl.DataFrame]:
        raise NotImplementedError
    def run_all(self, **kwargs) -> dict:
        raise NotImplementedError
    def merge_all(self, **kwargs) -> Union[pd.DataFrame, pl.DataFrame]:
        raise NotImplementedError
    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        raise NotImplementedError

# ------------------------- Polars Blocking -------------------------
class PolarsBlocking(BaseBlocking):
    def __init__(self, df: pl.DataFrame, record_id_col: str = "RecordID"):
        logging.info(f"Initializing PolarsBlocking with DataFrame columns: {df.columns}")
        self.df = df
        self.id_col = record_id_col
        self.available_columns = df.columns
        self.attr_map = {}
        for col in self.available_columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        logging.info(f"Attribute map: {self.attr_map}")
        self.rules = self._generate_all_rules()

    def _pairs_from_block(self, group: pl.DataFrame, rule_name: str) -> list:
        ids = group[self.id_col].to_list()
        return [(ids[i], ids[j], rule_name) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    def run_rule(self, rule_name: str) -> pl.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])
        pairs = self.rules[rule_name]()
        return pl.DataFrame(pairs, schema=["RecordID1", "RecordID2", "Rule"]) if pairs else pl.DataFrame(schema=["RecordID1", "RecordID2", "Rule"])

    def _generate_all_rules(self) -> dict:
        rules = {}
        if "email" in self.attr_map:
            c = self.attr_map["email"]
            def email_rule():
                df = self.df.with_columns(pl.col(c).str.strip_chars().str.to_lowercase().alias("_email"))
                blocks = df.group_by("_email")
                pairs = []
                for _, g in blocks:
                    if g["_email"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Email")
                return pairs
            rules["Email"] = email_rule

        if "passport" in self.attr_map:
            c = self.attr_map["passport"]
            def passport_rule():
                df = self.df.with_columns(pl.col(c).str.strip_chars().str.replace_all(" ", "").alias("_passport"))
                blocks = df.group_by("_passport")
                pairs = []
                for _, g in blocks:
                    if g["_passport"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Passport")
                return pairs
            rules["Passport"] = passport_rule

        if "national_id" in self.attr_map:
            c = self.attr_map["national_id"]
            def national_id_rule():
                df = self.df.with_columns(pl.col(c).str.strip_chars().str.replace_all("-", "").alias("_national_id"))
                blocks = df.group_by("_national_id")
                pairs = []
                for _, g in blocks:
                    if g["_national_id"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "NationalID")
                return pairs
            rules["NationalID"] = national_id_rule

        if "phone" in self.attr_map:
            c = self.attr_map["phone"]
            def phone_rule():
                df = self.df.with_columns(
                    pl.col(c).str.strip_chars().str.replace_all(" ", "").str.replace_all("-", "").str.replace_all("(", "").alias("_phone")
                )
                blocks = df.group_by("_phone")
                pairs = []
                for _, g in blocks:
                    if g["_phone"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Phone")
                return pairs
            rules["Phone"] = phone_rule

        if "postcode" in self.attr_map:
            c = self.attr_map["postcode"]
            def postcode_rule():
                df = self.df.with_columns(pl.col(c).str.strip_chars().str.to_lowercase().alias("_postcode"))
                blocks = df.group_by("_postcode")
                pairs = []
                for _, g in blocks:
                    if g["_postcode"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Postcode")
                return pairs
            rules["Postcode"] = postcode_rule

        if "street" in self.attr_map and "house_number" in self.attr_map:
            s, h = self.attr_map["street"], self.attr_map["house_number"]
            def street_house2_rule():
                df = self.df.with_columns([
                    pl.col(s).str.strip_chars().str.to_lowercase().alias("_street"),
                    pl.col(h).str.strip_chars().str.replace_all(" ", "").str.slice(0, 2).alias("_house2")
                ])
                blocks = df.group_by(["_street", "_house2"])
                pairs = []
                for _, g in blocks:
                    if g["_street"].is_not_null().any() and g["_house2"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Street_House2")
                return pairs
            rules["Street_House2"] = street_house2_rule

        if "street" in self.attr_map and "city" in self.attr_map:
            s, c = self.attr_map["street"], self.attr_map["city"]
            def street_city_rule():
                df = self.df.with_columns([
                    pl.col(s).map_elements(lambda x: jellyfish.soundex(x) if x else "", return_dtype=pl.String).alias("_street_sdx"),
                    pl.col(c).str.strip_chars().str.to_lowercase().alias("_city_lower")
                ])
                blocks = df.group_by(["_street_sdx", "_city_lower"])
                pairs = []
                for _, g in blocks:
                    if g["_street_sdx"].is_not_null().any() and g["_city_lower"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Street_City")
                return pairs
            rules["Street_City"] = street_city_rule

        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f, l = self.attr_map["first_name"], self.attr_map["last_name"]
            def soundex_name_rule():
                df = self.df.with_columns([
                    pl.col(f).map_elements(lambda x: jellyfish.soundex(x) if x else "", return_dtype=pl.String).alias("_fname_sdx"),
                    pl.col(l).map_elements(lambda x: jellyfish.soundex(x) if x else "", return_dtype=pl.String).alias("_lname_sdx")
                ])
                blocks = df.group_by(["_fname_sdx", "_lname_sdx"])
                pairs = []
                for _, g in blocks:
                    if g["_fname_sdx"].is_not_null().any() and g["_lname_sdx"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "SoundexName")
                return pairs
            rules["SoundexName"] = soundex_name_rule

            def prefix2_name_rule():
                df = self.df.with_columns([
                    pl.col(f).str.strip_chars().str.to_lowercase().str.slice(0, 2).alias("_fname_prefix"),
                    pl.col(l).str.strip_chars().str.to_lowercase().str.slice(0, 2).alias("_lname_prefix")
                ])
                blocks = df.group_by(["_fname_prefix", "_lname_prefix"])
                pairs = []
                for _, g in blocks:
                    if g["_fname_prefix"].is_not_null().any() and g["_lname_prefix"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Prefix2Name")
                return pairs
            rules["Prefix2Name"] = prefix2_name_rule

            def last_firstchar_rule():
                df = self.df.with_columns([
                    pl.col(l).str.strip_chars().str.to_lowercase().alias("_l_lower"),
                    pl.col(f).str.strip_chars().str.to_lowercase().str.slice(0, 1).alias("_f_initial")
                ])
                blocks = df.group_by(["_l_lower", "_f_initial"])
                pairs = []
                for _, g in blocks:
                    if g["_l_lower"].is_not_null().any() and g["_f_initial"].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Last_FirstChar")
                return pairs
            rules["Last_FirstChar"] = last_firstchar_rule

        if "dob" in self.attr_map:
            c = self.attr_map["dob"]
            def dob_rule():
                blocks = self.df.group_by(c)
                pairs = []
                for _, g in blocks:
                    if g[c].is_not_null().any():
                        pairs += self._pairs_from_block(g, "DOB")
                return pairs
            rules["DOB"] = dob_rule

        if "year_of_birth" in self.attr_map:
            c = self.attr_map["year_of_birth"]
            def yob_rule():
                blocks = self.df.group_by(c)
                pairs = []
                for _, g in blocks:
                    if g[c].is_not_null().any():
                        pairs += self._pairs_from_block(g, "YOB")
                return pairs
            rules["YOB"] = yob_rule

        if "month_of_birth" in self.attr_map and "year_of_birth" in self.attr_map:
            m, y = self.attr_map["month_of_birth"], self.attr_map["year_of_birth"]
            def monthyear_rule():
                blocks = self.df.group_by([m, y])
                pairs = []
                for _, g in blocks:
                    if g[m].is_not_null().any() and g[y].is_not_null().any():
                        pairs += self._pairs_from_block(g, "MonthYear")
                return pairs
            rules["MonthYear"] = monthyear_rule

        if "time" in self.attr_map:
            c = self.attr_map["time"]
            def time_rule():
                blocks = self.df.group_by(c)
                pairs = []
                for _, g in blocks:
                    if g[c].is_not_null().any():
                        pairs += self._pairs_from_block(g, "TimeExact")
                return pairs
            rules["TimeExact"] = time_rule

        if "last_name" in self.attr_map and "first_name" in self.attr_map and "dob" in self.attr_map:
            l, f, d = self.attr_map["last_name"], self.attr_map["first_name"], self.attr_map["dob"]
            def last_first_dob_rule():
                df = self.df.with_columns([
                    pl.col(l).str.strip_chars().str.to_lowercase().alias("_lname_lower"),
                    pl.col(f).map_elements(lambda x: jellyfish.soundex(x) if x else "", return_dtype=pl.String).alias("_fname_sdx")
                ])
                blocks = df.group_by(["_lname_lower", "_fname_sdx", d])
                pairs = []
                for _, g in blocks:
                    if g["_lname_lower"].is_not_null().any() and g["_fname_sdx"].is_not_null().any() and g[d].is_not_null().any():
                        pairs += self._pairs_from_block(g, "Last_First_DOB")
                return pairs
            rules["Last_First_DOB"] = last_first_dob_rule

        logging.info(f"Generated rules: {list(rules.keys())}")
        return rules

    def run_all(self, parallel=True, max_workers=100) -> dict:
        logging.info(f"Running all rules (parallel={parallel}, max_workers={max_workers})")
        res = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rule = {executor.submit(self.run_rule, rule): rule for rule in self.rules}
                for future in as_completed(future_to_rule):
                    rule = future_to_rule[future]
                    try:
                        df = future.result()
                        if not df.is_empty():
                            res[rule] = df
                            logging.info(f"Rule {rule} generated {len(df)} pairs")
                    except Exception as e:
                        logging.error(f"Skipping {rule}: {str(e)}")
        else:
            for rule in self.rules:
                try:
                    df = self.run_rule(rule)
                    if not df.is_empty():
                        res[rule] = df
                        logging.info(f"Rule {rule} generated {len(df)} pairs")
                except Exception as e:
                    logging.error(f"Skipping {rule}: {str(e)}")
        return res

    def merge_all(self) -> pl.DataFrame:
        per_rule = self.run_all()
        if not per_rule:
            logging.info("No pairs generated by any rule")
            return pl.DataFrame(schema=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pl.concat(list(per_rule.values()))
        agg = combined.group_by(["RecordID1", "RecordID2"]).agg(RulesUsed=pl.col("Rule").unique().sort().str.join(","))
        logging.info(f"Merged {len(agg)} unique pairs")
        return agg

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            logging.info("No rule results to report.")
            return None, None

        stats = []
        for rule, df in per_rule_dfs.items():
            if df.is_empty():
                continue
            pairs = len(df)
            unique_records = len(pl.concat([df["RecordID1"], df["RecordID2"]]).unique())
            avg_block_size = (2 * pairs / unique_records) if unique_records > 0 else 0
            stats.append((rule, pairs, unique_records, avg_block_size))

        if not stats:
            logging.info("No candidate pairs found by any rule.")
            return None, None

        stats_df = pl.DataFrame(stats, schema=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort("Pairs", descending=True)
        stats_df = stats_df.with_columns(PairsPct=100 * pl.col("Pairs") / pl.col("Pairs").sum())
        stats_df_pandas = stats_df.to_pandas()

        logging.info("\n=== Blocking Rules Summary ===")
        print(stats_df_pandas.to_string(index=False, float_format="%.2f"))

        top = stats_df_pandas.head(show_top_n)
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
            logging.info(f"Saved interactive plot to {save_html_path}")

        return stats_df_pandas, chart

# ------------------------- SQL Blocking -------------------------
class SQLBlocking(BaseBlocking):
    def __init__(self, conn_str, view_name, record_id_col="RecordID", dialect="mysql"):
        logging.info(f"Initializing SQLBlocking with conn_str={conn_str}, view_name={view_name}")
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
        self.rules = self._generate_all_rules()
        self.fast_stats_only = True
        self.max_pairs_per_rule = None

    def _dialect_config(self):
        if self.dialect == "mysql":
            return {"lower":"LOWER","trim":"TRIM","substr":"SUBSTRING","instr":"INSTR","replace":"REPLACE"}, True, "`"
        elif self.dialect in ["postgresql","postgres"]:
            return {"lower":"LOWER","trim":"TRIM","substr":"SUBSTRING","instr":"POSITION","replace":"REPLACE"}, True, '"'
        else:
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

    def run_rule(self, rule_name: str, sql: str = None) -> pd.DataFrame:
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
            logging.error(f"Skipping {rule_name}: {str(e)}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def _rule_key_expr(self, attr: str) -> str | None:
        fn = self.fn_map
        qcol = lambda a: self.q(self.attr_map[a]) if a in self.attr_map else None
        if attr == "email" and qcol("email"):
            c = qcol("email")
            return f"{fn['lower']}({fn['trim']}({c}))"
        if attr == "email_domain" and qcol("email"):
            c = qcol("email")
            if self.dialect in ["postgresql", "postgres"]:
                return f"SUBSTRING({c} FROM POSITION('@' IN {c})+1)"
            else:
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

    def run_all(self, parallel=True, max_workers=8) -> dict:
        logging.info(f"Running all SQL rules (parallel={parallel}, max_workers={max_workers})")
        res = {}
        for rule, sql in self.rules.items():
            try:
                df = self.run_rule(rule, sql)
                if not df.empty:
                    res[rule] = df
                    logging.info(f"Rule {rule} generated {len(df)} pairs")
            except Exception as e:
                logging.error(f"Skipping {rule}: {str(e)}")
        return res

    def _generate_all_rules(self) -> dict:
        rules = {}
        for attr in [
            "email", "email_domain", "passport", "national_id", "phone", "postcode",
            "street_house2", "street_city", "last_firstchar", "prefix2_name", "soundex_name",
            "dob", "yob", "monthyear", "time"
        ]:
            key_expr = self._rule_key_expr(attr)
            if key_expr is None:
                continue
            rule_name = attr.capitalize()
            if attr == "street_house2":
                rule_name = "Street_House2"
            elif attr == "street_city":
                rule_name = "Street_City"
            elif attr == "last_firstchar":
                rule_name = "Last_FirstChar"
            elif attr == "prefix2_name":
                rule_name = "Prefix2Name"
            elif attr == "soundex_name":
                rule_name = "SoundexName"
            elif attr == "yob":
                rule_name = "YOB"
            elif attr == "monthyear":
                rule_name = "MonthYear"
            sql = (
                f"SELECT t1.{self.q(self.id_col)} AS RecordID1, t2.{self.q(self.id_col)} AS RecordID2 "
                f"FROM {self.view} t1 "
                f"JOIN {self.view} t2 ON {key_expr.replace('t1.', '')} = {key_expr.replace('t1.', 't2.')}"
                f" AND t1.{self.q(self.id_col)} < t2.{self.q(self.id_col)} "
                f"WHERE {key_expr} IS NOT NULL"
            )
            rules[rule_name] = sql
        logging.info(f"Generated SQL rules: {list(rules.keys())}")
        return rules

    def merge_all(self) -> pd.DataFrame:
        per_rule = self.run_all()
        if not per_rule:
            logging.info("No pairs generated by any rule")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        logging.info(f"Merged {len(agg)} unique pairs")
        return agg.rename(columns={"Rule":"RulesUsed"})

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            logging.info("No rule results to report.")
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
            logging.info("No candidate pairs found by any rule.")
            return None, None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        logging.info("\n=== Blocking Rules Summary ===")
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
            logging.info(f"Saved interactive plot to {save_html_path}")

        return stats_df, chart

# ------------------------- Pandas Blocking -------------------------
class PandasBlocking(BaseBlocking):
    def __init__(self, df: pd.DataFrame, record_id_col: str = "RecordID"):
        logging.info(f"Initializing PandasBlocking with DataFrame columns: {df.columns}")
        self.df = df
        self.id_col = record_id_col
        self.available_columns = df.columns
        self.attr_map = {}
        for col in self.available_columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        self.rules = self._generate_all_rules()

    def _pairs_from_block(self, group: pd.DataFrame, rule_name: str) -> list:
        ids = group[self.id_col].tolist()
        return [(ids[i], ids[j], rule_name) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    def run_rule(self, rule_name: str) -> pd.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        pairs = self.rules[rule_name]()
        return pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"]) if pairs else pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def _generate_all_rules(self) -> dict:
        rules = {}
        if "email" in self.attr_map:
            c = self.attr_map["email"]
            def email_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                df["_email"] = df[c].str.strip().str.lower()
                blocks = df.groupby("_email")
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Email")
                return pairs
            rules["Email"] = email_rule

        if "passport" in self.attr_map:
            c = self.attr_map["passport"]
            def passport_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                df["_passport"] = df[c].str.strip().str.replace(" ", "")
                blocks = df.groupby("_passport")
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Passport")
                return pairs
            rules["Passport"] = passport_rule

        if "national_id" in self.attr_map:
            c = self.attr_map["national_id"]
            def national_id_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                df["_national_id"] = df[c].str.strip().str.replace("-", "")
                blocks = df.groupby("_national_id")
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "NationalID")
                return pairs
            rules["NationalID"] = national_id_rule

        if "phone" in self.attr_map:
            c = self.attr_map["phone"]
            def phone_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                df["_phone"] = df[c].str.strip().str.replace(" ", "").str.replace("-", "").str.replace("(", "")
                blocks = df.groupby("_phone")
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Phone")
                return pairs
            rules["Phone"] = phone_rule

        if "postcode" in self.attr_map:
            c = self.attr_map["postcode"]
            def postcode_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                df["_postcode"] = df[c].str.strip().str.lower()
                blocks = df.groupby("_postcode")
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Postcode")
                return pairs
            rules["Postcode"] = postcode_rule

        if "street" in self.attr_map and "house_number" in self.attr_map:
            s, h = self.attr_map["street"], self.attr_map["house_number"]
            def street_house2_rule():
                df = self.df[[self.id_col, s, h]].dropna(subset=[s, h])
                df["_street"] = df[s].str.strip().str.lower()
                df["_house2"] = df[h].str.strip().str.replace(" ", "").str[:2]
                blocks = df.groupby(["_street", "_house2"])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Street_House2")
                return pairs
            rules["Street_House2"] = street_house2_rule

        if "street" in self.attr_map and "city" in self.attr_map:
            s, c = self.attr_map["street"], self.attr_map["city"]
            def street_city_rule():
                df = self.df[[self.id_col, s, c]].dropna(subset=[s, c])
                df["_street_sdx"] = df[s].apply(lambda x: jellyfish.soundex(x) if x else "")
                df["_city_lower"] = df[c].str.strip().str.lower()
                blocks = df.groupby(["_street_sdx", "_city_lower"])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Street_City")
                return pairs
            rules["Street_City"] = street_city_rule

        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            f, l = self.attr_map["first_name"], self.attr_map["last_name"]
            def soundex_name_rule():
                df = self.df[[self.id_col, f, l]].dropna(subset=[f, l])
                df["_fname_sdx"] = df[f].apply(lambda x: jellyfish.soundex(x) if x else "")
                df["_lname_sdx"] = df[l].apply(lambda x: jellyfish.soundex(x) if x else "")
                blocks = df.groupby(["_fname_sdx", "_lname_sdx"])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "SoundexName")
                return pairs
            rules["SoundexName"] = soundex_name_rule

            def prefix2_name_rule():
                df = self.df[[self.id_col, f, l]].dropna(subset=[f, l])
                df["_fname_prefix"] = df[f].str.strip().str.lower().str[:2]
                df["_lname_prefix"] = df[l].str.strip().str.lower().str[:2]
                blocks = df.groupby(["_fname_prefix", "_lname_prefix"])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Prefix2Name")
                return pairs
            rules["Prefix2Name"] = prefix2_name_rule

            def last_firstchar_rule():
                df = self.df[[self.id_col, f, l]].dropna(subset=[f, l])
                df["_l_lower"] = df[l].str.strip().str.lower()
                df["_f_initial"] = df[f].str.strip().str.lower().str[:1]
                blocks = df.groupby(["_l_lower", "_f_initial"])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Last_FirstChar")
                return pairs
            rules["Last_FirstChar"] = last_firstchar_rule

        if "dob" in self.attr_map:
            c = self.attr_map["dob"]
            def dob_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                blocks = df.groupby(c)
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "DOB")
                return pairs
            rules["DOB"] = dob_rule

        if "year_of_birth" in self.attr_map:
            c = self.attr_map["year_of_birth"]
            def yob_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                blocks = df.groupby(c)
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "YOB")
                return pairs
            rules["YOB"] = yob_rule

        if "month_of_birth" in self.attr_map and "year_of_birth" in self.attr_map:
            m, y = self.attr_map["month_of_birth"], self.attr_map["year_of_birth"]
            def monthyear_rule():
                df = self.df[[self.id_col, m, y]].dropna(subset=[m, y])
                blocks = df.groupby([m, y])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "MonthYear")
                return pairs
            rules["MonthYear"] = monthyear_rule

        if "time" in self.attr_map:
            c = self.attr_map["time"]
            def time_rule():
                df = self.df[[self.id_col, c]].dropna(subset=[c])
                blocks = df.groupby(c)
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "TimeExact")
                return pairs
            rules["TimeExact"] = time_rule

        if "last_name" in self.attr_map and "first_name" in self.attr_map and "dob" in self.attr_map:
            l, f, d = self.attr_map["last_name"], self.attr_map["first_name"], self.attr_map["dob"]
            def last_first_dob_rule():
                df = self.df[[self.id_col, l, f, d]].dropna(subset=[l, f, d])
                df["_lname_lower"] = df[l].str.strip().str.lower()
                df["_fname_sdx"] = df[f].apply(lambda x: jellyfish.soundex(x) if x else "")
                blocks = df.groupby(["_lname_lower", "_fname_sdx", d])
                pairs = []
                for _, g in blocks:
                    pairs += self._pairs_from_block(g, "Last_First_DOB")
                return pairs
            rules["Last_First_DOB"] = last_first_dob_rule

        logging.info(f"Generated Pandas rules: {list(rules.keys())}")
        return rules

    def run_all(self, parallel=True, max_workers=8) -> dict:
        logging.info(f"Running all Pandas rules (parallel={parallel}, max_workers={max_workers})")
        res = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rule = {executor.submit(self.run_rule, rule): rule for rule in self.rules}
                for future in as_completed(future_to_rule):
                    rule = future_to_rule[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            res[rule] = df
                            logging.info(f"Rule {rule} generated {len(df)} pairs")
                    except Exception as e:
                        logging.error(f"Skipping {rule}: {str(e)}")
        else:
            for rule in self.rules:
                try:
                    df = self.run_rule(rule)
                    if not df.empty:
                        res[rule] = df
                        logging.info(f"Rule {rule} generated {len(df)} pairs")
                except Exception as e:
                    logging.error(f"Skipping {rule}: {str(e)}")
        return res

    def merge_all(self) -> pd.DataFrame:
        per_rule = self.run_all()
        if not per_rule:
            logging.info("No pairs generated by any rule")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        logging.info(f"Merged {len(agg)} unique pairs")
        return agg.rename(columns={"Rule":"RulesUsed"})

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            logging.info("No rule results to report.")
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
            logging.info("No candidate pairs found by any rule.")
            return None, None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        logging.info("\n=== Blocking Rules Summary ===")
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
            logging.info(f"Saved interactive plot to {save_html_path}")

        return stats_df, chart

# ------------------------- Enhanced Pandas Blocking -------------------------
class EnhancedPandasBlocking(PandasBlocking):
    def __init__(self, df: pd.DataFrame, record_id_col: str = "RecordID", comparison_thresholds: Dict = None):
        super().__init__(df, record_id_col)
        self.thresholds = comparison_thresholds or {
            "email": 0.9,
            "first_name": 0.85,
            "last_name": 0.85,
            "phone": 0.95,
            "postcode": 0.9,
            "street": 0.85,
            "dob": 1.0
        }

    def run_rule(self, rule_name: str) -> pd.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        pairs = []
        if rule_name in ["Email", "Phone", "Postcode", "DOB"]:
            pairs = super().run_rule(rule_name)
        elif rule_name == "FuzzyName":
            if "first_name" in self.attr_map and "last_name" in self.attr_map:
                f, l = self.attr_map["first_name"], self.attr_map["last_name"]
                df = self.df[[self.id_col, f, l]].dropna(subset=[f, l])
                df["_fname_lower"] = df[f].str.strip().str.lower()
                df["_lname_lower"] = df[l].str.strip().str.lower()
                for i in range(len(df)):
                    for j in range(i + 1, len(df)):
                        fname_sim = jaro_winkler_similarity(df.iloc[i]["_fname_lower"], df.iloc[j]["_fname_lower"])
                        lname_sim = jaro_winkler_similarity(df.iloc[i]["_lname_lower"], df.iloc[j]["_lname_lower"])
                        if fname_sim >= self.thresholds.get("first_name", 0.85) and lname_sim >= self.thresholds.get("last_name", 0.85):
                            pairs.append((df.iloc[i][self.id_col], df.iloc[j][self.id_col], "FuzzyName"))
        return pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"]) if pairs else pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def _generate_all_rules(self) -> dict:
        rules = super()._generate_all_rules()
        if "first_name" in self.attr_map and "last_name" in self.attr_map:
            rules["FuzzyName"] = self.run_rule
        logging.info(f"Generated EnhancedPandas rules: {list(rules.keys())}")
        return rules

# ------------------------- High Performance Blocking -------------------------
class HighPerformanceBlocking(PandasBlocking):
    def __init__(self, df: pd.DataFrame, record_id_col: str = "RecordID", chunk_size: int = 10000, max_workers: int = 8):
        super().__init__(df, record_id_col)
        self.chunk_size = chunk_size
        self.max_workers = max_workers

    def _process_chunk(self, chunk: pd.DataFrame, rule_name: str) -> pd.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        pairs = self.rules[rule_name](chunk)
        return pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"]) if pairs else pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def run_rule(self, rule_name: str) -> pd.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        chunks = [self.df[i:i + self.chunk_size] for i in range(0, len(self.df), self.chunk_size)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk, rule_name) for chunk in chunks]
            results = [f.result() for f in as_completed(futures)]
        return pd.concat(results, ignore_index=True)

# ------------------------- Mongo Blocking -------------------------
class MongoBlocking(BaseBlocking):
    def __init__(self, conn_str, collection, record_id_col="_id"):
        from pymongo import MongoClient
        logging.info(f"Initializing MongoBlocking with conn_str={conn_str}, collection={collection}")
        self.client = MongoClient(conn_str)
        self.db = self.client.get_default_database()
        self.collection = self.db[collection]
        self.id_col = record_id_col
        self.available_columns = self._get_collection_fields()
        self.attr_map = {}
        for col in self.available_columns:
            attr = match_attribute(col)
            if attr and attr not in self.attr_map:
                self.attr_map[attr] = col
        self.rules = self._generate_all_rules()

    def _get_collection_fields(self):
        sample_doc = self.collection.find_one()
        if not sample_doc:
            return []
        return list(sample_doc.keys())

    def _pairs_from_groups(self, groups, rule_name):
        pairs = []
        for g in groups:
            ids = g.get("ids", [])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((str(ids[i]), str(ids[j]), rule_name))
        return pairs

    def _group_by_field(self, field, transform=None, rule_name=None):
        pipeline = [
            {"$match": {field: {"$ne": None}}},
            {"$group": {"_id": f"${field}", "ids": {"$push": f"${self.id_col}"}}}
        ]
        if transform:
            pipeline.insert(0, {"$addFields": {field: transform}})
        groups = list(self.collection.aggregate(pipeline))
        return self._pairs_from_groups(groups, rule_name or field)

    def run_rule(self, rule_name: str) -> pd.DataFrame:
        if rule_name not in self.rules:
            logging.warning(f"Rule {rule_name} not found in rules: {list(self.rules.keys())}")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])
        pairs = self.rules[rule_name]()
        return pd.DataFrame(pairs, columns=["RecordID1", "RecordID2", "Rule"]) if pairs else pd.DataFrame(columns=["RecordID1", "RecordID2", "Rule"])

    def _generate_all_rules(self) -> dict:
        rules = {}
        if "email" in self.attr_map:
            c = self.attr_map["email"]
            def email_rule():
                return self._group_by_field(c, {"$toLower": {"$trim": {"input": f"${c}"}}}, "Email")
            rules["Email"] = email_rule

        if "phone" in self.attr_map:
            c = self.attr_map["phone"]
            def phone_rule():
                return self._group_by_field(c, {"$replaceAll": {"input": {"$replaceAll": {"input": {"$trim": {"input": f"${c}"}}, "find": "-", "replacement": ""}}, "find": " ", "replacement": ""}}, "Phone")
            rules["Phone"] = phone_rule

        if "postcode" in self.attr_map:
            c = self.attr_map["postcode"]
            def postcode_rule():
                return self._group_by_field(c, {"$toLower": {"$trim": {"input": f"${c}"}}}, "Postcode")
            rules["Postcode"] = postcode_rule

        if "dob" in self.attr_map:
            c = self.attr_map["dob"]
            def dob_rule():
                return self._group_by_field(c, None, "DOB")
            rules["DOB"] = dob_rule

        logging.info(f"Generated Mongo rules: {list(rules.keys())}")
        return rules

    def run_all(self, parallel=True, max_workers=8) -> dict:
        logging.info(f"Running all Mongo rules (parallel={parallel}, max_workers={max_workers})")
        res = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rule = {executor.submit(self.run_rule, rule): rule for rule in self.rules}
                for future in as_completed(future_to_rule):
                    rule = future_to_rule[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            res[rule] = df
                            logging.info(f"Rule {rule} generated {len(df)} pairs")
                    except Exception as e:
                        logging.error(f"Skipping {rule}: {str(e)}")
        else:
            for rule in self.rules:
                try:
                    df = self.run_rule(rule)
                    if not df.empty:
                        res[rule] = df
                        logging.info(f"Rule {rule} generated {len(df)} pairs")
                except Exception as e:
                    logging.error(f"Skipping {rule}: {str(e)}")
        return res

    def merge_all(self) -> pd.DataFrame:
        per_rule = self.run_all()
        if not per_rule:
            logging.info("No pairs generated by any rule")
            return pd.DataFrame(columns=["RecordID1", "RecordID2", "RulesUsed"])
        combined = pd.concat(per_rule.values(), ignore_index=True)
        agg = combined.groupby(["RecordID1", "RecordID2"])["Rule"].apply(lambda s: ",".join(sorted(set(s)))).reset_index()
        logging.info(f"Merged {len(agg)} unique pairs")
        return agg.rename(columns={"Rule":"RulesUsed"})

    def generate_rule_report(self, per_rule_dfs: dict, show_top_n: int = 10, save_html_path: str = None):
        if not per_rule_dfs:
            logging.info("No rule results to report.")
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
            logging.info("No candidate pairs found by any rule.")
            return None, None

        stats_df = pd.DataFrame(stats, columns=["Rule", "Pairs", "UniqueRecords", "AvgBlockSize"])
        stats_df = stats_df.sort_values("Pairs", ascending=False).reset_index(drop=True)
        stats_df["PairsPct"] = 100 * stats_df["Pairs"] / stats_df["Pairs"].sum()

        logging.info("\n=== Blocking Rules Summary ===")
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
            logging.info(f"Saved interactive plot to {save_html_path}")

        return stats_df, chart

# ------------------------- Blocking Factory -------------------------
class BlockingFactory:
    @staticmethod
    def auto_create(
        conn_str: Optional[str] = None,
        view_name: Optional[str] = None,
        record_id_col: str = "RecordID",
        df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        collection: Optional[str] = None,
        enhanced: bool = False,
        high_performance: bool = False,
        comparison_thresholds: Optional[Dict] = None,
        chunk_size: int = 10000,
        max_workers: int = 8,
    ):
        logging.info(f"BlockingFactory.auto_create called with df={df is not None}, conn_str={conn_str}, view_name={view_name}, collection={collection}")
        if df is not None:
            if isinstance(df, pl.DataFrame):
                logging.info("Creating PolarsBlocking")
                return PolarsBlocking(df, record_id_col)
            elif isinstance(df, pd.DataFrame):
                if enhanced:
                    logging.info("Creating EnhancedPandasBlocking")
                    return EnhancedPandasBlocking(df, record_id_col, comparison_thresholds)
                elif high_performance:
                    logging.info("Creating HighPerformanceBlocking")
                    return HighPerformanceBlocking(df, record_id_col, chunk_size=chunk_size, max_workers=max_workers)
                else:
                    logging.info("Creating PandasBlocking")
                    return PandasBlocking(df, record_id_col)
        elif collection:
            logging.info("Creating MongoBlocking")
            return MongoBlocking(conn_str, collection, record_id_col)
        elif conn_str and view_name:
            if "trino://" in conn_str:
                dialect = "trino"
            elif "mysql+pymysql://" in conn_str:
                dialect = "mysql"
            elif "postgresql://" in conn_str:
                dialect = "postgresql"
            else:
                dialect = "hive"
            logging.info(f"Creating SQLBlocking with dialect={dialect}")
            return SQLBlocking(conn_str, view_name, record_id_col, dialect=dialect)
        raise ValueError("Must provide either df, or conn_str and view_name, or conn_str and collection")

# ------------------------- Pipeline Runner -------------------------
def run_blocking_pipeline(
    conn_str: Optional[str] = None,
    view_name: Optional[str] = None,
    df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    collection: Optional[str] = None,
    record_id_col: str = "RecordID",
    parallel: bool = True,
    max_workers: int = 8,
    show_top_n: int = 10,
    save_html_path: Optional[str] = None,
    enhanced: bool = False,
    high_performance: bool = False,
    comparison_thresholds: Optional[Dict] = None,
    create_clusters: bool = True,
    min_cluster_size: int = 2,
    chunk_size: int = 10000,
) -> Tuple:
    logging.info(f"Starting blocking pipeline with df={df is not None}, conn_str={conn_str}, collection={collection}")
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
    try:
        per_rule = blocker.run_all(parallel=parallel, max_workers=max_workers)
    except TypeError:
        per_rule = blocker.run_all()
    if high_performance and hasattr(blocker, "generate_performance_report"):
        result = blocker.generate_performance_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
    elif enhanced and hasattr(blocker, "generate_enhanced_report"):
        result = blocker.generate_enhanced_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
    else:
        result = blocker.generate_rule_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
    stats_df, chart = result if result else (None, None)
    clusters_df = None
    if create_clusters and per_rule:
        merged = blocker.merge_all()
        if not merged.empty:
            G = nx.Graph()
            for _, row in merged.to_pandas().iterrows():
                G.add_edge(row["RecordID1"], row["RecordID2"], rules=row["RulesUsed"])
            components = list(nx.connected_components(G))
            clusters = []
            for i, component in enumerate(components):
                if len(component) >= min_cluster_size:
                    for record_id in component:
                        clusters.append({"ClusterID": i + 1, "RecordID": record_id})
            clusters_df = pd.DataFrame(clusters)
            if clusters_df is not None and not clusters_df.empty:
                logging.info(f"Created {len(clusters_df['ClusterID'].unique())} clusters with {len(clusters_df)} records")
    return blocker, per_rule, stats_df, chart, clusters_df

# ------------------------- Sample Data Generator -------------------------
def create_sample_data(n_records=100):
    np.random.seed(42)
    data = []
    base_names = [
        ("John", "Smith"), ("Jane", "Doe"), ("Michael", "Johnson"),
        ("Sarah", "Williams"), ("David", "Brown"), ("Lisa", "Davis"),
        ("Robert", "Miller"), ("Jennifer", "Wilson"), ("William", "Moore"),
        ("Elizabeth", "Taylor")
    ]
    for i in range(n_records):
        if i < len(base_names):
            first, last = base_names[i]
        else:
            first, last = base_names[i % len(base_names)]
        if i % 3 == 0:
            first_var, last_var = first, last
        elif i % 3 == 1:
            first_var = first + " Jr" if len(first) < 6 else first
            last_var = last + "son" if len(last) < 6 else last
        else:
            first_var = first.replace('a', 'e') if 'a' in first else first
            last_var = last.replace('i', 'y') if 'i' in last else last
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

# ------------------------- Module Exports -------------------------
__all__ = [
    'normalize_date',
    'BlockingFactory',
    'run_blocking_pipeline',
    'EnhancedPandasBlocking',
    'HighPerformanceBlocking',
    'PandasBlocking',
    'PolarsBlocking',
    'SQLBlocking',
    'MongoBlocking',
    'create_sample_data'
]
