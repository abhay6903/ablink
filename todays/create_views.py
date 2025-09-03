# universal_create_views.py

from sqlalchemy import create_engine, inspect, text
from sqlalchemy import types as satypes
from pymongo import MongoClient
import duckdb
import pandas as pd

def choose_tables(tables: list) -> list:
    """Ask user which tables/collections they want to create views for."""
    print("\n📌 Available tables/collections:")
    for i, t in enumerate(tables, 1):
        print(f"{i}. {t}")

    choice = input("\n👉 Enter table numbers (comma-separated) or 'all' for all: ").strip()

    if choice.lower() == "all":
        return tables
    else:
        idxs = [int(x.strip()) for x in choice.split(",")]
        return [tables[i - 1] for i in idxs if 0 < i <= len(tables)]


def create_views(conn_str: str, db_name: str = None, schema: str = None):
    """
    Automatically detects DB type from conn_str and creates *_copy views 
    for selected tables/collections, then prints sample rows.
    """
    
    # -------------------------------
    # MongoDB
    # -------------------------------
    if conn_str.startswith("mongodb://") or conn_str.startswith("mongodb+srv://"):
        if not db_name:
            db_name = input("Enter MongoDB database name: ")
        client = MongoClient(conn_str)
        db = client[db_name]
        collections = db.list_collection_names()

        chosen = choose_tables(collections)

        for coll in chosen:
            view_name = f"{coll}_copy"
            try:
                db.drop_collection(view_name)  # drop if exists
            except:
                pass
            db.command({
                "create": view_name,
                "viewOn": coll,
                "pipeline": []  # identity view
            })
            print(f"\n✅ Created MongoDB view: {view_name}")
            sample = list(db[view_name].find().limit(5))
            print(pd.DataFrame(sample))
        return
    
    # -------------------------------
    # DuckDB
    # -------------------------------
    if conn_str.endswith(".duckdb"):
        con = duckdb.connect(conn_str)
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]

        chosen = choose_tables(tables)

        for table_name in chosen:
            view_name = f"{table_name}_copy"
            sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table_name}"
            con.execute(sql)
            print(f"\n✅ Created DuckDB view: {view_name}")
            df = con.execute(f"SELECT * FROM {view_name} LIMIT 5").fetchdf()
            print(df)
        con.close()
        return
    
    # -------------------------------
    # SQLite
    # -------------------------------
    if conn_str.endswith(".db") or conn_str.endswith(".sqlite"):
        engine = create_engine(f"sqlite:///{conn_str}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        chosen = choose_tables(tables)

        with engine.connect() as conn:
            for table in chosen:
                view_name = f"{table}_copy"
                sql = f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT * FROM {table}"
                conn.execute(text(sql))
                print(f"\n✅ Created SQLite view: {view_name}")
                df = pd.read_sql(f"SELECT * FROM {view_name} LIMIT 5", conn)
                print(df)
        return
    
    # -------------------------------
    # SQLAlchemy Engines (MySQL, Postgres, Hive, Pinot, TPCH, System)
    # -------------------------------
    try:
        engine = create_engine(conn_str)
        inspector = inspect(engine)
        if not schema:
            schema = input("Enter schema name (leave empty if none): ") or None
        tables = inspector.get_table_names(schema=schema)

        chosen = choose_tables(tables)

        with engine.connect() as conn:
            for table in chosen:
                view_name = f"{table}_view"
                sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table}"
                conn.execute(text(sql))
                print(f"\n✅ Created SQL view: {view_name}")
                df = pd.read_sql(f"SELECT * FROM {view_name} LIMIT 5", conn)
                print(df)
        return
    except Exception as e:
        raise ValueError(f"❌ Could not detect DB type or create views. Error: {str(e)}")


if __name__ == "__main__":
    conn_str = input("🔑 Enter your connection string: ")
    create_views(conn_str)


def create_views_auto(conn_str: str, db_name: str = None, schema: str = None, tables: list = None, suffix: str = "_view", sanitize_empty_as_null: bool = False) -> list:
    """
    Non-interactive view creation for automation. Creates views for provided tables
    or for all tables if none provided. Returns list of created view names.
    """
    created_views = []

    # MongoDB
    if conn_str.startswith("mongodb://") or conn_str.startswith("mongodb+srv://"):
        if not db_name:
            raise ValueError("db_name is required for MongoDB in non-interactive mode")
        client = MongoClient(conn_str)
        db = client[db_name]
        collections = db.list_collection_names()
        target = tables or collections
        for coll in target:
            view_name = f"{coll}{suffix}"
            try:
                db.drop_collection(view_name)
            except Exception:
                pass
            db.command({"create": view_name, "viewOn": coll, "pipeline": []})
            created_views.append(view_name)
        return created_views

    # DuckDB
    if conn_str.endswith(".duckdb"):
        con = duckdb.connect(conn_str)
        try:
            all_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            target = tables or all_tables
            for table_name in target:
                view_name = f"{table_name}{suffix}"
                sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {table_name}"
                con.execute(sql)
                created_views.append(view_name)
            return created_views
        finally:
            con.close()

    # SQLite
    if conn_str.endswith(".db") or conn_str.endswith(".sqlite"):
        engine = create_engine(f"sqlite:///{conn_str}")
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        target = tables or all_tables
        with engine.connect() as conn:
            for table in target:
                view_name = f"{table}{suffix}"
                if sanitize_empty_as_null:
                    cols = inspector.get_columns(table)
                    select_parts = []
                    for c in cols:
                        col = c['name']
                        col_type = c.get('type')
                        if isinstance(col_type, satypes.String) or isinstance(col_type, satypes.Text):
                            # Clean sanitization: remove newlines, trim, empty->NULL
                            expr = f"NULLIF(TRIM(REPLACE(REPLACE({col}, '\\n', ''), '\\r', '')), '') AS {col}"
                            select_parts.append(expr)
                        else:
                            select_parts.append(col)
                    select_clause = ", ".join(select_parts)
                    sql = f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT {select_clause} FROM {table}"
                else:
                    sql = f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT * FROM {table}"
                conn.execute(text(sql))
                created_views.append(view_name)
        return created_views

    
    # SQL (MySQL, Postgres, Hive, Trino, etc.) via SQLAlchemy
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    dialect = engine.dialect.name.lower()
    all_tables = inspector.get_table_names(schema=schema)
    target = tables or all_tables

    with engine.connect() as conn:
        for table in target:
            view_name = f"{table}{suffix}"
            base_table = f"{schema}.{table}" if schema else table

            # Build sanitized column expressions
            cols = inspector.get_columns(table, schema=schema)
            select_parts = []
            for c in cols:
                col = c['name']
                col_type = str(c.get('type')).lower()
                if sanitize_empty_as_null and any(t in col_type for t in ["char", "text", "string", "varchar"]):
                    expr = f"""
                        NULLIF(
                            TRIM(
                                REPLACE(REPLACE(REPLACE(REPLACE({col}, '\\n',' '), '\\r',' '), '\\t',' '), '%','')
                            ),
                            ''
                        ) AS {col}
                    """
                    select_parts.append(expr)
                else:
                    select_parts.append(f"{col} AS {col}")

            select_clause = ",\n                ".join(select_parts)

            # Add RecordID depending on dialect/version
            if dialect in ["postgresql", "postgres", "hive", "trino"]:
                sql = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS RecordID,
                    {select_clause}
                FROM {base_table} t
                """
            elif dialect == "mysql":
                server_version = conn.exec_driver_sql("SELECT VERSION()").scalar()
                major_version = int(server_version.split(".")[0])
                if major_version >= 8:
                    sql = f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT 
                        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS RecordID,
                        {select_clause}
                    FROM {base_table} t
                    """
                else:
                    sql = f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT 
                        (@rownum := @rownum + 1) AS RecordID,
                        {select_clause}
                    FROM {base_table} t, (SELECT @rownum := 0) r
                    """
            else:
                raise ValueError(f"Dialect {dialect} not supported for RecordID injection")

            conn.execute(text(sql))
            created_views.append(view_name)
