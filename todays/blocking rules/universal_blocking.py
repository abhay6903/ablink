# universal_blocking.py
import re
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import jellyfish
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt

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
    def auto_create(conn_str=None, view_name=None, record_id_col="RecordID", df=None, collection=None):
        # Case 1: Pandas DataFrame
        if df is not None:
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

        if save_html_path:
            chart.save(save_html_path)
            print(f"Saved interactive plot to {save_html_path}")

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
):
    """
    Unified pipeline for blocking:
      - Detects backend (SQL, Pandas, Mongo) automatically
      - Runs all blocking rules
      - Generates summary report + Altair chart
    Returns: (blocker, per_rule, stats_df, chart)
    """
    # Step 1: auto-detect backend
    blocker = BlockingFactory.auto_create(
        conn_str=conn_str,
        view_name=view_name,
        record_id_col=record_id_col,
        df=df,
        collection=collection,
    )

    # Step 2: run blocking
    if hasattr(blocker, "run_all"):
        try:
            per_rule = blocker.run_all(parallel=parallel, max_workers=max_workers)
        except TypeError:
            # for Pandas/Mongo which don’t support parallel
            per_rule = blocker.run_all()
    else:
        raise RuntimeError("Blocker does not implement run_all()")

    # Step 3: generate report
    if hasattr(blocker, "generate_rule_report"):
        result = blocker.generate_rule_report(per_rule, show_top_n=show_top_n, save_html_path=save_html_path)
        if result is None:
            stats_df, chart = None, None
        else:
            stats_df, chart = result
    else:
        stats_df, chart = None, None

    return blocker, per_rule, stats_df, chart

   
