import polars as pl
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from sqlalchemy import create_engine, text
from typing import Optional, Dict, List
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')
pio.renderers.default = 'notebook_connected'
alt.data_transformers.disable_max_rows()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ValueDistributionReport:
    def __init__(self, connection_string: str, base_table_name: str, view_suffix: str = "_view", use_altair: bool = True):
        self.engine = create_engine(connection_string, connect_args={'http_scheme': 'http', 'auth': None})
        self.base_table_name = base_table_name
        self.view_name = f"{base_table_name}{view_suffix}"
        self.use_altair = use_altair
        self.df = None
        self.color_palettes = {
            'seaborn': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
        }
        self.current_palette = self.color_palettes['seaborn']
        logging.info(f"Initialized ValueDistributionReport for {self.view_name}")

    def _fetch_and_sanitize_data(self) -> pl.DataFrame:
        """Fetch data from Trino and sanitize using Polars."""
        try:
            query = f"SELECT * FROM {self.base_table_name} LIMIT 1000000"
            df_pandas = pd.read_sql(text(query), self.engine)
            df = pl.from_pandas(df_pandas)
            logging.info(f"Fetched {len(df)} rows from {self.base_table_name}")

            # Sanitize string columns
            string_cols = [col for col, dtype in df.schema.items() if dtype == pl.Utf8 and col != 'RecordID']
            for col in string_cols:
                df = df.with_columns(
                    pl.col(col).str.replace_all(r'[\n\r\t%]', ' ').str.strip_chars()
                    .str.replace_all(r'\s+', ' ').replace({'': None, 'nan': None, 'None': None, r'\N': None})
                )

            # Add RecordID if not present
            if 'RecordID' not in df.columns:
                df = df.with_row_count(name='RecordID', offset=1)

            logging.info(f"Sanitized data for {self.view_name}, columns: {df.columns}")
            return df
        except Exception as e:
            logging.error(f"Failed to fetch/sanitize data: {str(e)}")
            raise

    def _get_column_type(self, series: pl.Series, col_name: str = '') -> str:
        """Determine column type for visualization."""
        col_name_lower = col_name.lower()
        place_name_hints = ['place', 'city', 'village', 'state', 'district', 'country', 'postcode', 'zip', 'birth_place']
        if any(h in col_name_lower for h in place_name_hints):
            return 'categorical'

        if series.dtype in (pl.Date, pl.Datetime):
            return 'datetime'
        if series.dtype == pl.Duration:
            return 'timedelta'
        if series.dtype.is_numeric():
            return 'continuous' if series.n_unique() > 20 else 'categorical'
        return 'categorical'

    def _coerce_timedelta_to_numeric(self, series: pl.Series) -> tuple:
        """Convert timedelta series to numeric for plotting."""
        cleaned = series.drop_nulls()
        if cleaned.is_empty():
            return cleaned, 'Days'
        try:
            days = cleaned.dt.total_seconds() / 86400.0
            if days.max() is not None and abs(days.max()) < 0.05:
                hours = cleaned.dt.total_seconds() / 3600.0
                return hours, 'Hours'
            return days, 'Days'
        except Exception:
            seconds = cleaned.dt.total_seconds()
            return seconds, 'Seconds'

    def _null_stats_text(self, series: pl.Series) -> tuple:
        """Calculate null statistics."""
        total = len(series)
        nulls = series.is_null().sum()
        pct = (nulls / total * 100.0) if total > 0 else 0.0
        return nulls, total, f"From total records {pct:.1f}% (count {nulls:,}) values were null."

    def _prepare_categorical_data(self, series: pl.Series, col_name: str, top_n: int) -> pl.DataFrame:
        """Prepare categorical data for visualization, handling date columns."""
        logging.info(f"Preparing categorical data for {col_name}, dtype: {series.dtype}")
        if series.dtype in (pl.Date, pl.Datetime):
            series = series.cast(pl.Utf8).fill_null('')
            logging.info(f"Converted {col_name} from date to string")
        s = series.drop_nulls()
        if s.dtype == pl.Utf8:
            s = s.str.replace_all(r'[\n\r\t%]', ' ').str.strip_chars().str.replace_all(r'\s+', ' ').replace({'': None, 'nan': None, 'None': None, r'\N': None}).str.to_titlecase()
        vc = s.value_counts()
        if vc.is_empty():
            return pl.DataFrame({col_name: [], 'Count': []})
        vc = vc.sort('count', descending=True).head(max(top_n, 20))
        return vc.rename({'count': 'Count'}).rename({vc.columns[0]: col_name})

    def _altair_categorical_chart(self, data: pl.DataFrame, col_name: str, null_text: str):
        """Create Altair categorical chart."""
        logging.info(f"Creating categorical chart for {col_name}, data columns: {data.columns}")
        display_data = data.head(20).to_pandas()
        if display_data.empty:
            return alt.Chart(pd.DataFrame({'msg': ['No data']})).mark_text(size=14).encode(text='msg:N').properties(height=80)

        hover = alt.selection_point(on='mouseover', fields=[col_name], nearest=True, empty='none')
        show_param = alt.param(name='show_values', value=False, bind=alt.binding_checkbox(name='Show values'))

        base = alt.Chart(display_data).add_params(hover, show_param)
        bars = base.mark_bar().encode(
            x=alt.X(f'{col_name}:N', sort='-y', title=col_name),
            y=alt.Y('Count:Q'),
            color=alt.condition(hover, alt.value('#1f77b4'), alt.value('#7fb3d5')),
            tooltip=[alt.Tooltip(f'{col_name}:N', title=col_name), alt.Tooltip('Count:Q')]
        )
        labels = base.mark_text(dy=-5, color='black').encode(
            x=alt.X(f'{col_name}:N', sort='-y'),
            y='Count:Q',
            text=alt.Text('Count:Q', format=','),
            opacity=alt.condition(show_param, alt.value(1.0), alt.value(0.0))
        )
        chart = (bars + labels).properties(height=360, title=f"{col_name} — {null_text}").resolve_scale(y='independent')
        return chart

    def _altair_hist_chart(self, clean_series: pl.Series, col_name: str, x_label: str, null_text: str):
        """Create Altair histogram for continuous/timedelta/datetime data."""
        logging.info(f"Creating histogram for {col_name}, x_label: {x_label}, dtype: {clean_series.dtype}")
        if len(clean_series) > 50000:
            clean_series = clean_series.sample(n=50000, seed=42)
        
        df_for_chart = clean_series.to_frame(name=x_label)
        
        is_datetime = clean_series.dtype in (pl.Date, pl.Datetime)

        if is_datetime:
            x_encoding = alt.X(f'{x_label}:T', timeUnit='yearmonth', title=x_label)
        else:
            x_encoding = alt.X(f'{x_label}:Q', title=x_label, bin=alt.Bin(maxbins=50))
            
        hist = alt.Chart(df_for_chart).mark_bar().encode(
            x=x_encoding,
            y=alt.Y('count()', title='Count')
        ).properties(height=360, title=f"{col_name} — {null_text}").interactive()
        return hist

    def _altair_percentile_chart(self, series: pl.Series, col_name: str, x_label: str):
        """Create Altair percentile chart for numerical and date-based columns."""
        logging.info(f"Creating percentile chart for {col_name}")
        
        clean_series = series.drop_nulls()
        if clean_series.is_empty() or len(clean_series) < 2:
            return alt.Chart(pd.DataFrame({'msg': [f'Not enough data in {col_name} for percentile analysis']})) \
                .mark_text(size=12).encode(text='msg:N').properties(height=80, title=f"Percentile Distribution for {col_name}")

        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        percentile_labels = ["1st", "5th", "10th", "25th (Q1)", "50th (Median)", "75th (Q3)", "90th", "95th", "99th"]
        
        original_dtype = clean_series.dtype
        is_datetime = original_dtype in (pl.Date, pl.Datetime)
        
        if is_datetime:
            numeric_series = clean_series.cast(pl.Int64)
            quantile_values_numeric = [numeric_series.quantile(p) for p in percentiles]
            values = pl.Series(quantile_values_numeric, dtype=pl.Float64).cast(pl.Int64).cast(original_dtype)
        else:
            values = [clean_series.quantile(p) for p in percentiles]

        percentile_data = pd.DataFrame({'Percentile': percentile_labels, 'Value': values})

        if is_datetime:
            percentile_data['Value'] = pd.to_datetime(percentile_data['Value']).dt.strftime('%Y-%m-%d')
            x_encoding = alt.X('Value:N', title=x_label, sort=None)
            tooltip_encoding = [alt.Tooltip('Percentile:N'), alt.Tooltip('Value:N', title='Value')]
            text_format = None
        else:
            x_encoding = alt.X('Value:Q', title=x_label)
            tooltip_encoding = [alt.Tooltip('Percentile:N'), alt.Tooltip('Value:Q', format=',.2f')]
            text_format = ',.2f'

        base = alt.Chart(percentile_data).properties(title=f"Value Percentiles for {col_name}", height=250)
        
        bars = base.mark_bar(opacity=0.8).encode(
            y=alt.Y('Percentile:N', sort=percentile_labels, title="Percentile"),
            x=x_encoding,
            tooltip=tooltip_encoding
        )
        
        text = base.mark_text(align='left', baseline='middle', dx=4, fontSize=10).encode(
                text=alt.Text('Value:Q', format=text_format) if not is_datetime else alt.Text('Value:N'),
                y=alt.Y('Percentile:N', sort=percentile_labels),
                x=x_encoding,
                color=alt.value('black')
            )

        return bars + text

    def _create_enhanced_visualization(self, series: pl.Series, col_name: str, col_type: str, top_n: int = 15):
        """Create visualization for a column, including a percentile chart for numerical types."""
        logging.info(f"Creating visualization for {col_name}, type: {col_type}")
        if self.use_altair:
            _, _, null_text = self._null_stats_text(series)
            
            main_chart = None
            percentile_chart = None
            
            if col_type in ['continuous', 'datetime', 'timedelta']:
                x_label = col_name
                if series.dtype == pl.Duration:
                    clean_vals, x_label = self._coerce_timedelta_to_numeric(series)
                else:
                    clean_vals = series.drop_nulls()
                main_chart = self._altair_hist_chart(clean_vals, col_name, x_label, null_text)
                percentile_chart = self._altair_percentile_chart(clean_vals, col_name, x_label)
            else: # Categorical
                display_data = self._prepare_categorical_data(series, col_name, top_n)
                main_chart = self._altair_categorical_chart(display_data, col_name, null_text)
            
            try:
                from IPython.display import display
                if main_chart:
                    display(main_chart)
                if percentile_chart:
                    display(percentile_chart)
            except Exception as e:
                logging.error(f"Chart display failed for {col_name}: {str(e)}")
                print(f"Chart generation failed for {col_name}: {str(e)}")

    def create_overview_dashboard(self, max_cols: int = None) -> None:
        """Create overview dashboard for all columns."""
        if self.df is None:
            self.df = self._fetch_and_sanitize_data()
        columns_to_show = self.df.columns if max_cols is None else self.df.columns[:max_cols]

        if self.use_altair:
            charts = []
            row = []
            for col in columns_to_show:
                if col.lower() == 'recordid':
                    continue

                col_type = self._get_column_type(self.df[col], col)
                _, _, null_text = self._null_stats_text(self.df[col])
                
                if col_type in ['continuous', 'datetime', 'timedelta']:
                    x_label = col
                    if self.df[col].dtype == pl.Duration:
                        clean_vals, x_label = self._coerce_timedelta_to_numeric(self.df[col])
                    else:
                        clean_vals = self.df[col].drop_nulls()
                    ch = self._altair_hist_chart(clean_vals, col_name=col, x_label=x_label, null_text=null_text).properties(width=250, height=220)
                else:
                    display_data = self._prepare_categorical_data(self.df[col], col, top_n=8)
                    ch = self._altair_categorical_chart(display_data, col, null_text).properties(width=250, height=220)

                row.append(ch)
                if len(row) == 3:
                    charts.append(alt.hconcat(*row))
                    row = []
            if row:
                charts.append(alt.hconcat(*row))

            dashboard = alt.vconcat(*charts).properties(title=f"🎨 Value Distribution Overview Dashboard — Table: {self.view_name}")
            try:
                from IPython.display import display
                display(dashboard)
            except Exception as e:
                logging.error(f"Dashboard display failed: {str(e)}")
                print(f"Dashboard generation failed: {str(e)}")
        else:
            logging.warning("Altair disabled; no fallback visualization implemented")

    def analyze_all_columns(self, top_n: int = 10, show_individual_plots: bool = True, max_columns: int = None, sample_size: int = None) -> Dict:
        """Analyze all columns and generate visualizations."""
        if self.df is None:
            self.df = self._fetch_and_sanitize_data()
        if sample_size:
            self.df = self.df.sample(n=sample_size, seed=42)
        if max_columns:
            self.df = self.df.select(self.df.columns[:max_columns])

        results = {}
        for col in self.df.columns:
            if col.lower() == 'recordid':
                continue
            
            col_type = self._get_column_type(self.df[col], col)
            print(f"Processing column: {col}, type: {col_type}")
            
            if col_type == 'timedelta':
                vc = self.df[col].drop_nulls().cast(pl.Utf8).value_counts().rename({"count": "Count"})
            else:
                vc = self.df[col].drop_nulls().value_counts().rename({"count": "Count"})
            vc = vc.rename({vc.columns[0]: "value"})
            
            print(f"vc columns after rename: {vc.columns}")
            top_vals = vc.head(top_n).to_pandas()
            bottom_vals = vc.tail(top_n).to_pandas()
            results[col] = {"top_values": top_vals, "bottom_values": bottom_vals, "type": col_type}

            if show_individual_plots:
                self._create_enhanced_visualization(self.df[col], col, col_type, top_n)

        self.create_overview_dashboard(max_cols=max_columns)
        return results

    def create_complete_report(self, connection_string: str = None, base_table_name: str = None) -> Dict:
        """Generate a complete value distribution report."""
        if connection_string:
            self.engine = create_engine(connection_string, connect_args={'http_scheme': 'http', 'auth': None})
        if base_table_name:
            self.base_table_name = base_table_name
            self.view_name = f"{base_table_name}{self.view_suffix}"
        logging.info(f"🚀 Complete Value Distribution Report for {self.view_name}")
        return self.analyze_all_columns(top_n=15, show_individual_plots=True)