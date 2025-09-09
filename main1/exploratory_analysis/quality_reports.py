import polars as pl
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sqlalchemy import create_engine, text
import numpy as np
from typing import Optional, Dict
import warnings
import logging
warnings.filterwarnings('ignore')
pio.renderers.default = 'notebook_connected'
alt.data_transformers.enable('default', max_rows=50000)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataQualityReport:
    def __init__(self, connection_string: str, base_table_name: str, view_suffix: str = "_view"):
        self.engine = create_engine(connection_string, connect_args={'http_scheme': 'http', 'auth': None})
        self.base_table_name = base_table_name
        self.view_name = f"{base_table_name}{view_suffix}"
        self.df = None
        self.colors = {
            'excellent': '#00E676', 'good': '#2196F3', 'fair': '#FF9800', 'poor': '#FF5722', 'critical': '#E91E63',
            'gradient': ['#E91E63', '#FF5722', '#FF9800', '#2196F3', '#00E676']
        }
        self.template = "plotly_white"
        self.use_altair = True
        logging.info(f"Initialized DataQualityReport for {self.view_name}")

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
                    pl.col(col).str.strip_chars().str.replace_all(r'[\n\r\t%]', ' ')
                    .str.replace_all(r'\s+', ' ').replace({'': None, 'nan': None, 'None': None, r'\N': None})
                )

            # Handle DOB as date if present
            if 'dob' in df.columns:
                df = df.with_columns(pl.col('dob').str.to_date(format=None, strict=False))

            # Add RecordID if not present
            if 'RecordID' not in df.columns:
                df = df.with_row_count(name='RecordID', offset=1)

            logging.info(f"Sanitized data for {self.view_name}, columns: {df.columns}")
            return df
        except Exception as e:
            logging.error(f"Failed to fetch/sanitize data: {str(e)}")
            raise

    def _get_completeness_color(self, completeness: float) -> str:
        """Return color based on completeness percentage."""
        if completeness >= 95:
            return self.colors['excellent']
        elif completeness >= 80:
            return self.colors['good']
        elif completeness >= 60:
            return self.colors['fair']
        elif completeness >= 40:
            return self.colors['poor']
        else:
            return self.colors['critical']

    def _get_quality_level(self, completeness: float) -> str:
        """Categorize data quality based on completeness."""
        if completeness >= 95:
            return 'Excellent'
        elif completeness >= 80:
            return 'Good'
        elif completeness >= 60:
            return 'Fair'
        elif completeness >= 40:
            return 'Poor'
        else:
            return 'Critical'

    def get_completeness(self, plot: bool = True, height: int = 800, save_path: Optional[str] = None) -> pl.DataFrame:
        """Calculate and visualize column completeness."""
        if self.df is None:
            self.df = self._fetch_and_sanitize_data()

        if self.df.is_empty():
            logging.warning(f"Table '{self.view_name}' is empty.")
            return pl.DataFrame()

        total_rows = len(self.df)
        completeness = self.df.select(pl.all().is_not_null().mean() * 100)
        missing_counts = self.df.select(pl.all().is_null().sum())

        completeness_df = pl.DataFrame({
            'Column': completeness.columns,
            'Completeness (%)': completeness.row(0),
            'Missing Count': missing_counts.row(0),
            'Total Rows': [total_rows] * len(completeness.columns),
            'Quality Level': [self._get_quality_level(comp) for comp in completeness.row(0)]
        }).with_columns(pl.col('Completeness (%)').round(2))

        completeness_df = completeness_df.sort('Completeness (%)')

        if plot:
            self._create_interactive_plot(completeness_df, height, save_path)

        return completeness_df.sort('Completeness (%)', descending=True)

    def _create_interactive_plot(self, completeness_df: pl.DataFrame, height: int, save_path: Optional[str] = None):
        """Create interactive visualization with Altair."""
        if self.use_altair:
            try:
                base = completeness_df.to_pandas()
                bar = alt.Chart(base).mark_bar().encode(
                    x=alt.X('Completeness (%):Q', title='Completeness (%)', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Column:N', sort='-x'),
                    color=alt.value('#2196F3'),
                    tooltip=[
                        alt.Tooltip('Column:N'),
                        alt.Tooltip('Completeness (%):Q', format='.1f'),
                        alt.Tooltip('Missing Count:Q'),
                        alt.Tooltip('Quality Level:N')
                    ]
                ).properties(height=int(height * 0.65))
                pie_source = completeness_df.group_by('Quality Level').agg(count=pl.count()).to_pandas()
                pie = alt.Chart(pie_source).mark_arc(innerRadius=40).encode(
                    theta='count:Q',
                    color='Quality Level:N',
                    tooltip=['Quality Level:N', 'count:Q']
                ).properties(height=int(height * 0.35))
                chart = alt.vconcat(bar, pie).resolve_legend(color='independent').properties(
                    title=f"Data Quality Assessment Dashboard — Table: {self.view_name}"
                )
                from IPython.display import display
                display(chart)
                return
            except Exception as e:
                logging.error(f"Altair plot failed: {str(e)}")

        # Plotly fallback
        completeness_df_sorted = completeness_df.sort('Completeness (%)')
        colors = [self._get_completeness_color(comp) for comp in completeness_df_sorted['Completeness (%)']]
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.7, 0.3],
            column_widths=[0.75, 0.25],
            subplot_titles=('Column Completeness', 'Quality Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}], [{"colspan": 2}, None]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        fig.add_trace(
            go.Bar(
                y=completeness_df_sorted['Column'],
                x=completeness_df_sorted['Completeness (%)'],
                orientation='h',
                marker=dict(color=colors, line=dict(color='white', width=2), opacity=0.8),
                hovertemplate="<b>%{y}</b><br>Completeness: %{x:.1f}%<br>Missing: %{customdata[0]} rows<br>Quality: %{customdata[1]}<extra></extra>",
                customdata=completeness_df_sorted.select(['Missing Count', 'Quality Level']).to_numpy(),
                name="Completeness"
            ),
            row=1, col=1
        )
        quality_counts = completeness_df.group_by('Quality Level').agg(count=pl.count()).to_pandas()
        quality_colors = [self.colors[level.lower()] for level in quality_counts['Quality Level']]
        fig.add_trace(
            go.Pie(
                labels=quality_counts['Quality Level'],
                values=quality_counts['count'],
                marker=dict(colors=quality_colors, line=dict(color='white', width=2)),
                textinfo='label+percent',
                textfont=dict(size=11, color='white'),
                hovertemplate="<b>%{label}</b><br>Columns: %{value}<br>Percentage: %{percent}<extra></extra>"
            ),
            row=1, col=2
        )
        avg_completeness = completeness_df['Completeness (%)'].mean()
        min_completeness = completeness_df['Completeness (%)'].min()
        max_completeness = completeness_df['Completeness (%)'].max()
        total_columns = len(completeness_df)
        critical_columns = len(completeness_df.filter(pl.col('Completeness (%)') < 40))
        summary_text = (f"📊 <b>SUMMARY STATISTICS</b><br>"
                       f"Total Columns: <b>{total_columns}</b> | "
                       f"Average: <b>{avg_completeness:.1f}%</b> | "
                       f"Range: <b>{min_completeness:.1f}% - {max_completeness:.1f}%</b> | "
                       f"Critical Issues: <b>{critical_columns} columns</b>")
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=0.15,
            showarrow=False,
            font=dict(size=14, color="#2c3e50"),
            bgcolor="rgba(236, 240, 241, 0.8)",
            bordercolor="#bdc3c7",
            borderwidth=2,
            borderpad=10,
            row=2, col=1
        )
        fig.update_layout(
            title=dict(text=f"<b>🎯 Data Quality Assessment Dashboard</b><br><span style='font-size:16px; color:#7f8c8d'>Table: {self.view_name}</span>",
                       x=0.5, font=dict(size=24, color="#2c3e50"), pad=dict(b=24)),
            template=self.template,
            height=height,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            margin=dict(l=20, r=20, t=100, b=50)
        )
        fig.update_xaxes(title=dict(text="<b>Completeness (%)</b>", font=dict(size=14)), range=[0, 105], showgrid=True, gridcolor="rgba(0,0,0,0.1)", row=1, col=1)
        fig.update_yaxes(title=dict(text="<b>Columns</b>", font=dict(size=14)), showgrid=False, row=1, col=1)
        fig.update_layout(hovermode='closest')
        fig.show(config={"displayModeBar": False}, renderer='notebook_connected')
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1200, height=height, scale=2)
            logging.info(f"📁 Plot saved to: {save_path}")

    def create_animated_dashboard(self, height: int = 800, save_path: Optional[str] = None) -> None:
        """Create animated, interactive dashboard."""
        completeness_df = self.get_completeness(plot=False)
        if completeness_df.is_empty():
            return

        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.6, 0.4],
            column_widths=[0.7, 0.3],
            subplot_titles=('🎯 Column Completeness', '📊 Quality Distribution', '📈 Completeness Trends', ''),
            specs=[[{"type": "bar"}, {"type": "pie"}], [{"type": "scatter"}, {"type": "table"}]]
        )
        completeness_df_sorted = completeness_df.sort('Completeness (%)')
        colors = [self._get_completeness_color(comp) for comp in completeness_df_sorted['Completeness (%)']]
        fig.add_trace(
            go.Bar(
                y=completeness_df_sorted['Column'],
                x=completeness_df_sorted['Completeness (%)'],
                orientation='h',
                marker=dict(color=colors, line=dict(color='white', width=2), opacity=0.85),
                text=[f"{val:.1f}%" for val in completeness_df_sorted['Completeness (%)']],
                textposition='outside',
                textfont=dict(color='black', size=10, family='Arial Black'),
                hovertemplate="<b>📊 %{y}</b><br>✅ Completeness: <b>%{x:.1f}%</b><br>❌ Missing: <b>%{customdata[0]} rows</b><br>🏷️ Quality: <b>%{customdata[1]}</b><extra></extra>",
                customdata=completeness_df_sorted.select(['Missing Count', 'Quality Level']).to_numpy(),
                name="Completeness"
            ),
            row=1, col=1
        )
        quality_counts = completeness_df.group_by('Quality Level').agg(count=pl.count()).to_pandas()
        quality_colors = [self.colors[level.lower()] for level in quality_counts['Quality Level']]
        fig.add_trace(
            go.Pie(
                labels=quality_counts['Quality Level'],
                values=quality_counts['count'],
                marker=dict(colors=quality_colors, line=dict(color='white', width=3)),
                textinfo='label+percent',
                textfont=dict(size=12, color='white', family='Arial Black'),
                hole=0.4,
                hovertemplate="<b>🎯 %{label}</b><br>📊 Columns: <b>%{value}</b><br>📈 Percentage: <b>%{percent}</b><extra></extra>",
                pull=[0.1 if level == 'Critical' else 0 for level in quality_counts['Quality Level']]
            ),
            row=1, col=2
        )
        sorted_completeness = completeness_df.sort('Completeness (%)', descending=True)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sorted_completeness))),
                y=sorted_completeness['Completeness (%)'],
                mode='lines+markers',
                line=dict(color='#3498db', width=4, shape='spline'),
                marker=dict(size=8, color=sorted_completeness['Completeness (%)'], colorscale='RdYlGn', showscale=False, line=dict(color='white', width=2)),
                hovertemplate="<b>📊 Column Rank:</b> %{x}<br><b>✅ Completeness:</b> %{y:.1f}%<br><extra></extra>",
                name="Trend"
            ),
            row=2, col=1
        )
        report = self.generate_detailed_report()
        table_data = [
            ['📊 Total Columns', str(report['total_columns'])],
            ['📈 Average Completeness', f"{report['avg_completeness']:.1f}%"],
            ['✅ Excellent Columns', str(report['excellent_columns'])],
            ['⚠️ Issues Need Attention', str(report['poor_columns'] + report['critical_columns'])],
            ['🚨 Critical Issues', str(report['critical_columns'])]
        ]
        fig.add_trace(
            go.Table(
                header=dict(values=['<b>Metric</b>', '<b>Value</b>'], fill_color='#34495e', font=dict(color='white', size=12, family='Arial Black'), align='center'),
                cells=dict(values=list(zip(*table_data)), fill_color=[['#ecf0f1', '#d5dbdb'] * len(table_data)], font=dict(color='#2c3e50', size=11), align=['left', 'center'])
            ),
            row=2, col=2
        )
        fig.update_layout(
            title=dict(text=f"<b>📈 Interactive Data Quality Dashboard</b><br><span style='font-size:16px; color:#7f8c8d'>Table: {self.view_name}</span>",
                       x=0.5, font=dict(size=26, color="#2c3e50", family="Arial Black"), pad=dict(b=24)),
            template=self.template,
            height=height,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", font=dict(color="white", size=12, family="Arial"))
        )
        fig.update_xaxes(title=dict(text="<b>Completeness (%)</b>", font=dict(size=14)), range=[0, 105], showgrid=True, gridcolor="rgba(0,0,0,0.1)", row=1, col=1)
        fig.update_yaxes(title=dict(text="<b>Columns</b>", font=dict(size=14)), showgrid=False, row=1, col=1)
        fig.update_xaxes(title=dict(text="<b>Column Rank</b>", font=dict(size=12)), showgrid=True, gridcolor="rgba(0,0,0,0.1)", row=2, col=1)
        fig.update_yaxes(title=dict(text="<b>Completeness (%)</b>", font=dict(size=12)), range=[0, 105], showgrid=True, gridcolor="rgba(0,0,0,0.1)", row=2, col=1)
        fig.show(config={"displayModeBar": False}, renderer='notebook_connected')
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                logging.info(f"📁 Interactive dashboard saved to: {save_path}")

    def generate_detailed_report(self) -> dict:
        """Generate comprehensive data quality report."""
        completeness_df = self.get_completeness(plot=False)
        if completeness_df.is_empty():
            return {}

        report = {
            'table_name': self.view_name,
            'total_rows': completeness_df['Total Rows'][0],
            'total_columns': len(completeness_df),
            'avg_completeness': completeness_df['Completeness (%)'].mean(),
            'min_completeness': completeness_df['Completeness (%)'].min(),
            'max_completeness': completeness_df['Completeness (%)'].max(),
            'excellent_columns': len(completeness_df.filter(pl.col('Completeness (%)') >= 95)),
            'good_columns': len(completeness_df.filter((pl.col('Completeness (%)') >= 80) & (pl.col('Completeness (%)') < 95))),
            'fair_columns': len(completeness_df.filter((pl.col('Completeness (%)') >= 60) & (pl.col('Completeness (%)') < 80))),
            'poor_columns': len(completeness_df.filter((pl.col('Completeness (%)') >= 40) & (pl.col('Completeness (%)') < 60))),
            'critical_columns': len(completeness_df.filter(pl.col('Completeness (%)') < 40)),
            'worst_columns': completeness_df.sort('Completeness (%)').head(5).select(['Column', 'Completeness (%)']).to_dicts(),
            'best_columns': completeness_df.sort('Completeness (%)', descending=True).head(5).select(['Column', 'Completeness (%)']).to_dicts()
        }
        return report

    def print_summary(self):
        """Print formatted summary report."""
        report = self.generate_detailed_report()
        if not report:
            logging.warning(f"No data available for table: {self.view_name}")
            return

        print(f"\n{'🎯'*20}")
        print(f"📊 DATA QUALITY SUMMARY: {report['table_name'].upper()}")
        print(f"{'🎯'*20}")
        print(f"📋 Total Rows: {report['total_rows']:,}")
        print(f"📊 Total Columns: {report['total_columns']}")
        print(f"📈 Average Completeness: {report['avg_completeness']:.1f}%")
        print(f"📉 Completeness Range: {report['min_completeness']:.1f}% - {report['max_completeness']:.1f}%")
        print(f"\n🎯 QUALITY BREAKDOWN:")
        print(f"   ✅ Excellent (≥95%): {report['excellent_columns']} columns")
        print(f"   ✔️ Good (80-94%): {report['good_columns']} columns")
        print(f"   ⚠️ Fair (60-79%): {report['fair_columns']} columns")
        print(f"   ❌ Poor (40-59%): {report['poor_columns']} columns")
        print(f"   🚨 Critical (<40%): {report['critical_columns']} columns")
        if report['critical_columns'] > 0:
            print(f"\n🚨 ATTENTION NEEDED - Worst Columns:")
            for col in report['worst_columns']:
                print(f"   • {col['Column']}: {col['Completeness (%)']:.1f}%")
        print(f"\n⭐ TOP PERFORMERS:")
        for col in report['best_columns'][:3]:
            print(f"   • {col['Column']}: {col['Completeness (%)']:.1f}%")
        print(f"{'🎯'*20}\n")