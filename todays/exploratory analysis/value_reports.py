import pandas as pd
import pandas.api.types as pdt
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from typing import Optional, Dict, List
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
import altair as alt
alt.data_transformers.disable_max_rows()   # 👈 removes the row cap
# Set matplotlib to display inline in notebooks
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        import matplotlib
        matplotlib.use('module://matplotlib_inline.backend_inline')
except Exception:
    pass
plt.style.use('default')

class ValueDistributionReport:
    def __init__(self, connection_string: str, use_plotly: bool = True, max_inline_bars: int = 20, use_altair: bool = True):
        self.engine = create_engine(connection_string)
        self.use_plotly = use_plotly
        self.max_inline_bars = max_inline_bars
        self.use_altair = use_altair
        
        # Color palettes
        self.color_palettes = {
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
            'pastel': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD3BA', '#E0BBE4'],
            'seaborn': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
        }
        
        self.current_palette = self.color_palettes['seaborn']
        
        # Configure matplotlib
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

    def _get_column_type(self, series: pd.Series, col_name: str = '') -> str:
        col_name_lower = (col_name or '').lower()
        # Strong name-based hints FIRST
        dob_name_hints = ['dob', 'date_of_birth', 'birth_date', 'birthday']
        place_name_hints = ['place', 'city', 'village', 'state', 'district', 'country', 'postcode', 'zip', 'birth_place']
        if any(h in col_name_lower for h in dob_name_hints):
            return 'dob'
        if any(h in col_name_lower for h in place_name_hints):
            return 'categorical'

        # Type-based detection
        if pdt.is_datetime64_any_dtype(series):
            return 'datetime'
        if pdt.is_timedelta64_dtype(series):
            return 'timedelta'
        if pdt.is_numeric_dtype(series):
            return 'continuous' if series.nunique(dropna=True) > 20 else 'categorical'

        # Object/string: read full column and decide
        s = self._normalize_nulls(series)
        try:
            parsed = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
            valid_ratio = parsed.notna().mean()
            # Only treat as datetime if majority parses as dates
            if valid_ratio >= 0.8 and parsed.notna().sum() >= 50:
                return 'datetime'
        except Exception:
            pass
        return 'categorical'

    def _coerce_timedelta_to_numeric(self, series: pd.Series) -> tuple:
        """
        Convert a timedelta series to a numeric representation for plotting.
        Returns (numeric_series, unit_label).
        """
        cleaned = series.dropna()
        if cleaned.empty:
            return cleaned, 'Days'
        try:
            days = cleaned.dt.total_seconds() / 86400.0
            # If days are too small, switch to hours
            if days.max() is not None and abs(float(days.max())) < 0.05:
                hours = cleaned.dt.total_seconds() / 3600.0
                return hours, 'Hours'
            return days, 'Days'
        except Exception:
            seconds = cleaned.dt.total_seconds()
            return seconds, 'Seconds'

    def _normalize_nulls(self, series: pd.Series) -> pd.Series:
        """Treat newline/whitespace-only and common null-like tokens as NaN for object dtype."""
        if series.dtype == object:
            s = series.astype(str)
            # Remove newlines, tabs, carriage returns
            s = s.str.replace(r"[\n\r\t]", " ", regex=True)
            # Normalize multiple spaces to single space
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()
            # Common null-like tokens
            null_tokens = {"", "null", "none", "nan", "na", "n/a", "-", "--", "."}
            s = s.where(~s.str.lower().isin(null_tokens))
            return s
        return series

    def _null_stats_text(self, series: pd.Series) -> tuple:
        total = int(series.shape[0]) if series is not None else 0
        nulls = int(series.isna().sum()) if series is not None else 0
        pct = (nulls / total * 100.0) if total > 0 else 0.0
        return nulls, total, f"From total records {pct:.1f}% (count {nulls:,}) values were null."

    def _prepare_categorical_data(self, series: pd.Series, col_name: str, top_n: int) -> pd.DataFrame:
        s = series.dropna()
        if s.dtype == object:
            s = s.astype(str).str.strip()
            s = s.str.replace(r"\\s+", " ", regex=True)
            s = s.str.title()
        vc = s.value_counts()
        if len(vc) == 0:
            return pd.DataFrame({col_name: [], 'Count': []})
        vc = vc.head(max(top_n, self.max_inline_bars))
        data = vc.reset_index()
        data.columns = [col_name, 'Count']
        return data

    def _plotly_categorical_chart(self, data: pd.DataFrame, col_name: str, null_text: str) -> None:
        display_data = data.head(self.max_inline_bars).copy()
        if display_data.empty:
            fig = go.Figure()
            fig.add_annotation(text='No data', x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=f'📊 {col_name} - Distribution', template='plotly_white', height=360)
            fig.show(renderer='notebook_connected')
            return
        fig = go.Figure([
            go.Bar(
                x=[str(x) for x in display_data[col_name]],
                y=display_data['Count'],
                marker=dict(color='#3498db', opacity=0.85),
                text=[f"{int(v):,}" for v in display_data['Count']],
                textposition='none',
                hovertemplate="<b>%{x}</b><br>Count: <b>%{y:,}</b><extra></extra>"
            )
        ])
        fig.update_layout(
            title=f'📊 {col_name} - Distribution',
            template='plotly_white',
            xaxis=dict(tickangle=-45),
            height=440,
            margin=dict(l=30, r=20, t=80, b=80),
            updatemenus=[{
                'type': 'buttons', 'direction': 'right', 'x': 1.0, 'y': 1.2, 'xanchor': 'right', 'yanchor': 'top',
                'buttons': [
                    {'label': 'Show Values', 'method': 'restyle', 'args': [{'textposition': 'outside'}]},
                    {'label': 'Hide Values', 'method': 'restyle', 'args': [{'textposition': 'none'}]}
                ]
            }]
        )
        try:
            ymax = float(display_data['Count'].max())
        except Exception:
            ymax = 0
        fig.add_annotation(text=null_text, xref='paper', yref='y', x=0.0, y=ymax * 1.05 if ymax else 1.0,
                           showarrow=False, align='left', bgcolor='rgba(255,255,255,0.7)')
        fig.show(renderer='notebook_connected')

    # -------- Altair lightweight charts (safe for local) --------
    def _altair_categorical_chart(self, data: pd.DataFrame, col_name: str, null_text: str):
        display_data = data.head(self.max_inline_bars).copy()
        if display_data.empty:
            return (alt.Chart(pd.DataFrame({'msg': ['No data']}))
                    .mark_text(size=14)
                    .encode(text='msg:N')
                    .properties(height=80))

        # Interactivity: hover highlight and toggle for value labels
        hover = alt.selection_point(on='mouseover', fields=[col_name], nearest=True, empty='none')
        show_param = alt.param(
                                name='show_values',  # 👈 safe signal name
                                value=False,
                                bind=alt.binding_checkbox(name='Show values') ) # label for UI can still have spaces)                                )


        base = alt.Chart(display_data).add_params(hover, show_param)
        bars = (
            base.mark_bar()
            .encode(
                x=alt.X(f'{col_name}:N', sort='-y', title=col_name),
                y=alt.Y('Count:Q'),
                color=alt.condition(hover, alt.value('#1f77b4'), alt.value('#7fb3d5')),
                tooltip=[alt.Tooltip(f'{col_name}:N', title=col_name), alt.Tooltip('Count:Q')]
            )
        )

        labels = (
            base.mark_text(dy=-5, color='black')
            .encode(
                x=alt.X(f'{col_name}:N', sort='-y'),
                y='Count:Q',
                text=alt.Text('Count:Q', format=',') ,
                opacity=alt.condition(show_param, alt.value(1.0), alt.value(0.0))

            )
        )

        chart = (bars + labels).properties(
            height=360,
            title=f"{col_name} — {null_text}"
        ).resolve_scale(y='independent')
        return chart

    def _altair_hist_chart(self, clean_series: pd.Series, col_name: str, x_label: str, null_text: str):
        if len(clean_series) > 50000:
            clean_series = clean_series.sample(50000, random_state=42)
        df_hist = pd.DataFrame({'value': clean_series})
        if df_hist.empty:
            return (alt.Chart(pd.DataFrame({'msg': ['No data']}))
                    .mark_text(size=14)
                    .encode(text='msg:N')
                    .properties(height=80))
        base = alt.Chart(df_hist)
        hist = (
            base.mark_bar()
            .encode(
                x=alt.X('value:Q', bin=alt.Bin(maxbins=30), title=x_label),
                y=alt.Y('count():Q', title='Count'),
                tooltip=[alt.Tooltip('count():Q', title='Count')]
            )
        )
        return hist.properties(height=360, title=f"{col_name} — {null_text}").interactive()

    def _plotly_continuous_chart(self, series: pd.Series, col_name: str, null_text: str) -> None:
        # Handle timedelta by converting to numeric units
        x_label = 'Value'
        if pdt.is_timedelta64_dtype(series):
            clean_data, x_label = self._coerce_timedelta_to_numeric(series)
        else:
            clean_data = series.dropna()
        if len(clean_data) == 0:
            fig = go.Figure()
            fig.add_annotation(text='No data', x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=f'📈 {col_name} - Distribution', template='plotly_white', height=400)
            fig.show(renderer='notebook_connected')
            return
        fig = go.Figure([
            go.Histogram(x=clean_data, nbinsx=30, marker=dict(color='#9b59b6', opacity=0.8))
        ])
        fig.update_layout(title=f'📈 {col_name} - Distribution', template='plotly_white', xaxis_title=x_label,
                          height=440, margin=dict(l=30, r=20, t=80, b=60))
        fig.add_annotation(text=null_text, xref='paper', yref='paper', x=0.0, y=1.08,
                           showarrow=False, align='left', bgcolor='rgba(255,255,255,0.7)')
        fig.show(renderer='notebook_connected')

    def _plotly_dob_chart(self, series: pd.Series, col_name: str, null_text: str) -> None:
        clean_data = pd.to_datetime(series, errors='coerce').dropna()
        if len(clean_data) == 0:
            fig = go.Figure()
            fig.add_annotation(text='No valid dates', x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=f'📅 {col_name} - No Valid Data', template='plotly_white', height=380)
            fig.show(renderer='notebook_connected')
            return
        current_year = pd.Timestamp.now().year
        ages = current_year - clean_data.dt.year
        hist = np.histogram(ages, bins=30)
        fig = go.Figure([
            go.Bar(x=[str(x) for x in hist[1][:-1]], y=hist[0], marker=dict(color='#2ecc71', opacity=0.85))
        ])
        fig.update_layout(title=f'👥 {col_name} - Age Distribution', template='plotly_white', height=440,
                          margin=dict(l=30, r=20, t=80, b=60))
        fig.add_annotation(text=null_text, xref='paper', yref='paper', x=0.0, y=1.08,
                           showarrow=False, align='left', bgcolor='rgba(255,255,255,0.7)')
        fig.show(renderer='notebook_connected')

    def _create_enhanced_visualization(self, series: pd.Series, col_name: str, col_type: str, top_n: int = 15):
        if self.use_altair:
            cleaned_for_nulls = self._normalize_nulls(series)
            _, _, null_text = self._null_stats_text(cleaned_for_nulls)
            if col_type == 'dob':
                # Convert to ages
                clean_data = pd.to_datetime(cleaned_for_nulls, errors='coerce').dropna()
                if clean_data.empty:
                    chart = alt.Chart(pd.DataFrame({'msg': ['No valid dates']})).mark_text(size=14).encode(text='msg:N')
                else:
                    ages = (pd.Timestamp.now().year - clean_data.dt.year)
                    chart = self._altair_hist_chart(ages, col_name='Age', x_label='Age', null_text=null_text)
            elif col_type in ['continuous', 'datetime', 'timedelta']:
                # Coerce timedelta to numeric units
                x_label = 'Value'
                if pdt.is_timedelta64_dtype(cleaned_for_nulls):
                    clean_vals, x_label = self._coerce_timedelta_to_numeric(cleaned_for_nulls)
                else:
                    clean_vals = cleaned_for_nulls.dropna()
                chart = self._altair_hist_chart(clean_vals, col_name, x_label, null_text)
            else:
                display_data = self._prepare_categorical_data(cleaned_for_nulls, col_name, top_n)
                chart = self._altair_categorical_chart(display_data, col_name, null_text)
            try:
                import IPython.display as disp
                disp.display(chart)
            except Exception:
                pass
        elif self.use_plotly:
            cleaned_for_nulls = self._normalize_nulls(series)
            _, _, null_text = self._null_stats_text(cleaned_for_nulls)
            if col_type == 'dob':
                self._plotly_dob_chart(series, col_name, null_text)
            elif col_type in ['continuous', 'datetime', 'timedelta']:
                self._plotly_continuous_chart(cleaned_for_nulls, col_name, null_text)
            else:
                display_data = self._prepare_categorical_data(cleaned_for_nulls, col_name, top_n)
                self._plotly_categorical_chart(display_data, col_name, null_text)
        else:
            # Simplified for matplotlib-only case
            pass

    def create_overview_dashboard(self, table_name: str, max_cols: int = None) -> None:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
            if df.empty:
                print(f"⚠️ Table '{table_name}' is empty.")
                return

            columns_to_show = df.columns if max_cols is None else df.columns[:max_cols]

            if self.use_altair:
                charts = []
                row = []
                for idx, col in enumerate(columns_to_show):
                    col_type = self._get_column_type(df[col], col)
                    cleaned_for_nulls = self._normalize_nulls(df[col])
                    _, _, null_text = self._null_stats_text(cleaned_for_nulls)

                    if col_type == 'dob':
                        clean_data = pd.to_datetime(cleaned_for_nulls, errors='coerce').dropna()
                        if clean_data.empty:
                            ch = alt.Chart(pd.DataFrame({'msg': [f'{col}: No valid dates']})).mark_text(size=12).encode(text='msg:N').properties(width=250, height=220)
                        else:
                            ages = (pd.Timestamp.now().year - clean_data.dt.year)
                            ch = self._altair_hist_chart(ages, col_name='Age', x_label='Age', null_text=null_text).properties(width=250, height=220)
                    elif col_type in ['continuous', 'datetime', 'timedelta']:
                        x_label = 'Value'
                        if pdt.is_timedelta64_dtype(cleaned_for_nulls):
                            clean_vals, x_label = self._coerce_timedelta_to_numeric(cleaned_for_nulls)
                        else:
                            clean_vals = cleaned_for_nulls.dropna()
                        ch = self._altair_hist_chart(clean_vals, col_name=col, x_label=x_label, null_text=null_text).properties(width=250, height=220)
                    else:
                        display_data = self._prepare_categorical_data(cleaned_for_nulls, col, top_n=8)
                        ch = self._altair_categorical_chart(display_data, col, null_text).properties(width=250, height=220)

                    row.append(ch)
                    if len(row) == 3:
                        charts.append(alt.hconcat(*row))
                        row = []
                if row:
                    charts.append(alt.hconcat(*row))

                dashboard = alt.vconcat(*charts).properties(title=f"🎨 Value Distribution Overview Dashboard — Table: {table_name}")
                try:
                    import IPython.display as disp
                    disp.display(dashboard)
                except Exception:
                    pass
            else:
                # Fallback to matplotlib (previous behavior) if Altair disabled
                n_cols = min(3, len(columns_to_show))
                n_rows = (len(columns_to_show) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                axes = np.array(axes).flatten()
                for i, col in enumerate(columns_to_show):
                    ax = axes[i]
                    col_type = self._get_column_type(df[col], col)
                    if col_type == 'dob':
                        clean_data = pd.to_datetime(df[col], errors='coerce').dropna()
                        if len(clean_data) > 0:
                            current_year = pd.Timestamp.now().year
                            ages = current_year - clean_data.dt.year
                            ax.hist(ages, bins=20, color=self.current_palette[i % len(self.current_palette)])
                            ax.set_title(f'👥 {col} (Ages)')
                        else:
                            ax.text(0.5, 0.5, 'No valid dates', ha='center', va='center')
                    elif col_type in ['continuous', 'datetime']:
                        clean_data = df[col].dropna()
                        ax.hist(clean_data, bins=20, color=self.current_palette[i % len(self.current_palette)])
                        ax.set_title(f'📈 {col}')
                    elif col_type == 'timedelta':
                        numeric, unit = self._coerce_timedelta_to_numeric(df[col])
                        if len(numeric) > 0:
                            ax.hist(numeric, bins=20, color=self.current_palette[i % len(self.current_palette)])
                            ax.set_title(f'⏱ {col} ({unit})')
                        else:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    else:
                        display_data = self._prepare_categorical_data(df[col], col, top_n=8)
                        if not display_data.empty:
                            bars = ax.bar(range(len(display_data)), display_data['Count'], color=self.current_palette[i % len(self.current_palette)])
                            ax.set_xticks(range(len(display_data)))
                            ax.set_xticklabels([str(x)[:8] + '...' if len(str(x)) > 8 else str(x) for x in display_data[col]], rotation=45, fontsize=8)
                            for bar, count in zip(bars, display_data['Count']):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}', ha='center', va='bottom', fontsize=7)
                            ax.set_title(f'📊 {col}')
                        else:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                for i in range(len(columns_to_show), len(axes)):
                    axes[i].set_visible(False)
                plt.suptitle(f'🎨 Value Distribution Overview Dashboard\nTable: {table_name}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
        except Exception as e:
            print(f"❌ Error creating dashboard: {str(e)}")

    def analyze_all_columns(self, table_name: str, top_n: int = 10, show_individual_plots: bool = True, max_columns: int = None, sample_size: int = None) -> Dict:
        try:
            query = f"SELECT * FROM {table_name}"
            if sample_size:
                query += f" ORDER BY RAND() LIMIT {sample_size}"
            df = pd.read_sql(query, self.engine)
            if df.empty:
                print(f"⚠️ Table '{table_name}' is empty.")
                return {}

            if max_columns:
                df = df.iloc[:, :max_columns]

            results = {}
            for col in df.columns:
                col_type = self._get_column_type(df[col], col)
                if col_type == 'dob':
                    clean_data = pd.to_datetime(df[col], errors='coerce').dropna()
                    if len(clean_data) > 0:
                        current_year = pd.Timestamp.now().year
                        ages = current_year - clean_data.dt.year
                        vc = ages.value_counts().sort_index()
                    else:
                        vc = pd.Series(dtype=int)
                else:
                    if col_type == 'timedelta':
                        # Use string representation buckets to avoid Timedelta formatting errors
                        series_str = df[col].dropna().astype('timedelta64[s]').astype(str)
                        vc = series_str.value_counts()
                    else:
                        vc = self._normalize_nulls(df[col]).value_counts(dropna=True)

                top_vals = vc.head(top_n).reset_index()
                bottom_vals = vc.tail(top_n).reset_index()
                results[col] = {"top_values": top_vals, "bottom_values": bottom_vals, "type": col_type}

                if show_individual_plots:
                    self._create_enhanced_visualization(df[col], col, col_type, top_n)

            self.create_overview_dashboard(table_name, max_cols=max_columns)
            return results
        except Exception as e:
            print(f"❌ Error processing table '{table_name}': {str(e)}")
            return {}

    def create_complete_report(self, table_name: str, connection_string: str = None):
        if connection_string:
            self.engine = create_engine(connection_string)
        print(f"🚀 Complete Value Distribution Report for {table_name}")
        results = self.analyze_all_columns(table_name=table_name, top_n=15, show_individual_plots=True)
        return results
