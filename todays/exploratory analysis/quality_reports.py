import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import plotly.io as pio
import altair as alt
import numpy as np
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
pio.renderers.default = 'notebook_connected'
alt.data_transformers.enable('default', max_rows=50000)

class DataQualityReport:
    def __init__(self, connection_string: str):
        """
        Initialize the Data Quality Report with database connection.
        
        Args:
            connection_string (str): Database connection string
        """
        self.engine = create_engine(connection_string)
        
        # Beautiful gradient color scheme
        self.colors = {
            'excellent': '#00E676',    # Bright Green
            'good': '#2196F3',         # Material Blue
            'fair': '#FF9800',         # Orange
            'poor': '#FF5722',         # Deep Orange
            'critical': '#E91E63',     # Pink/Red
            'gradient': ['#E91E63', '#FF5722', '#FF9800', '#2196F3', '#00E676']
        }
        
        # Modern template
        self.template = "plotly_white"
        self.use_altair = True
    
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
    
    def get_completeness(self, 
                        table_name: str, 
                        plot: bool = True,
                        height: int = 800,
                        save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate and visualize column completeness for the given table.
        
        Args:
            table_name (str): Name of the database table
            plot (bool): Whether to generate visualization
            height (int): Height of the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            pd.DataFrame: Completeness statistics
        """
        try:
            # Load from table directly
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                print(f"Warning: Table '{table_name}' is empty.")
                return pd.DataFrame()
            
            # Calculate completeness metrics
            total_rows = len(df)
            completeness = df.notnull().mean() * 100
            missing_counts = df.isnull().sum()
            
            # Create comprehensive results DataFrame
            completeness_df = pd.DataFrame({
                'Column': completeness.index,
                'Completeness (%)': completeness.values,
                'Missing Count': missing_counts.values,
                'Total Rows': total_rows,
                'Quality Level': [self._get_quality_level(comp) for comp in completeness.values]
            }).round(2)
            
            # Sort by completeness (ascending) for better visualization
            completeness_df = completeness_df.sort_values('Completeness (%)')
            
            if plot:
                self._create_interactive_plot(completeness_df, table_name, height, save_path)
            
            return completeness_df.sort_values('Completeness (%)', ascending=False)
            
        except Exception as e:
            print(f"Error processing table '{table_name}': {str(e)}")
            return pd.DataFrame()
    
    def _create_interactive_plot(self, 
                               completeness_df: pd.DataFrame, 
                               table_name: str,
                               height: int,
                               save_path: Optional[str] = None):
        """Create a beautiful, interactive visualization with hover effects (Altair first, Plotly fallback)."""
        
        if self.use_altair:
            try:
                base = completeness_df.copy()
                bar = (
                    alt.Chart(base)
                    .mark_bar()
                    .encode(
                        x=alt.X('Completeness (%):Q', title='Completeness (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Column:N', sort='-x'),
                        color=alt.value('#2196F3'),
                        tooltip=[
                            alt.Tooltip('Column:N'),
                            alt.Tooltip('Completeness (%):Q', format='.1f'),
                            alt.Tooltip('Missing Count:Q'),
                            alt.Tooltip('Quality Level:N')
                        ]
                    )
                    .properties(height=int(height * 0.65))
                )
                pie_source = base.groupby('Quality Level').size().reset_index(name='count')
                pie = (
                    alt.Chart(pie_source)
                    .mark_arc(innerRadius=40)
                    .encode(
                        theta='count:Q',
                        color='Quality Level:N',
                        tooltip=['Quality Level:N', 'count:Q']
                    )
                    .properties(height=int(height * 0.35))
                )
                chart = alt.vconcat(bar, pie).resolve_legend(color='independent').properties(
                    title=f"Data Quality Assessment Dashboard — Table: {table_name}"
                )
                import IPython.display as disp
                disp.display(chart)
                return
            except Exception:
                pass
        
        # Create subplots (Plotly fallback)
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.7, 0.3],
            column_widths=[0.75, 0.25],
            subplot_titles=('Column Completeness', 'Quality Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Assign colors based on completeness
        colors = [self._get_completeness_color(comp) for comp in completeness_df['Completeness (%)']]
        
        # Main horizontal bar chart with enhanced hover
        fig.add_trace(
            go.Bar(
                y=completeness_df['Column'],
                x=completeness_df['Completeness (%)'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                hovertemplate="<b>%{y}</b><br>" +
                             "Completeness: %{x:.1f}%<br>" +
                             "Missing: %{customdata[0]} rows<br>" +
                             "Quality: %{customdata[1]}" +
                             "<extra></extra>",
                customdata=np.column_stack((completeness_df['Missing Count'], 
                                          completeness_df['Quality Level'])),
                name="Completeness",
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="white",
                    font=dict(color="white", size=12)
                )
            ),
            row=1, col=1
        )
        
        # Quality distribution pie chart
        quality_counts = completeness_df['Quality Level'].value_counts()
        quality_colors = [self.colors[level.lower()] for level in quality_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=quality_counts.index,
                values=quality_counts.values,
                marker=dict(colors=quality_colors, line=dict(color='white', width=2)),
                textinfo='label+percent',
                textfont=dict(size=11, color='white'),
                hovertemplate="<b>%{label}</b><br>" +
                             "Columns: %{value}<br>" +
                             "Percentage: %{percent}" +
                             "<extra></extra>",
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="white",
                    font=dict(color="white", size=12)
                )
            ),
            row=1, col=2
        )
        
        # Summary statistics as annotations
        avg_completeness = completeness_df['Completeness (%)'].mean()
        min_completeness = completeness_df['Completeness (%)'].min()
        max_completeness = completeness_df['Completeness (%)'].max()
        total_columns = len(completeness_df)
        critical_columns = len(completeness_df[completeness_df['Completeness (%)'] < 40])
        
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
        
        # Update layout with modern styling
        fig.update_layout(
            title=dict(
                text=f"<b>🎯 Data Quality Assessment Dashboard</b><br><span style='font-size:16px; color:#7f8c8d'>Table: {table_name}</span>",
                x=0.5,
                font=dict(size=24, color="#2c3e50"),
                pad=dict(b=24)
            ),
            template=self.template,
            height=height,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            margin=dict(l=20, r=20, t=100, b=50)
        )
        
        # Style the bar chart
        fig.update_xaxes(
            title=dict(text="<b>Completeness (%)</b>", font=dict(size=14)),
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=1,
            row=1, col=1
        )
        
        fig.update_yaxes(
            title=dict(text="<b>Columns</b>", font=dict(size=14)),
            showgrid=False,
            row=1, col=1
        )
        
        # Add hover mode for better interactivity
        fig.update_layout(hovermode='closest')
        
        # Show the interactive plot
        fig.show(config={"displayModeBar": False}, renderer='notebook_connected')
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1200, height=height, scale=2)
            print(f"📁 Plot saved to: {save_path}")
    
    def create_animated_dashboard(self, table_name: str, height: int = 800, save_path: Optional[str] = None) -> None:
        """
        Create an animated, interactive dashboard with multiple visualizations.
        
        Args:
            table_name (str): Name of the table to analyze
            height (int): Height of the dashboard figure
            save_path (str, optional): Path to save the dashboard as HTML
        """
        completeness_df = self.get_completeness(table_name, plot=False)
        
        if completeness_df.empty:
            return
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.6, 0.4],
            column_widths=[0.7, 0.3],
            subplot_titles=('🎯 Column Completeness', '📊 Quality Distribution', 
                          '📈 Completeness Trends', ''),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Sort data for better visualization
        completeness_df_sorted = completeness_df.sort_values('Completeness (%)')
        colors = [self._get_completeness_color(comp) for comp in completeness_df_sorted['Completeness (%)']]
        
        # 1. Enhanced horizontal bar chart
        fig.add_trace(
            go.Bar(
                y=completeness_df_sorted['Column'],
                x=completeness_df_sorted['Completeness (%)'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2),
                    opacity=0.85
                ),
                text=[f"{val:.1f}%" for val in completeness_df_sorted['Completeness (%)']],
                textposition='outside',
                textfont=dict(color='black', size=10, family='Arial Black'),
                hovertemplate="<b>📊 %{y}</b><br>" +
                             "✅ Completeness: <b>%{x:.1f}%</b><br>" +
                             "❌ Missing: <b>%{customdata[0]} rows</b><br>" +
                             "🏷️ Quality: <b>%{customdata[1]}</b>" +
                             "<extra></extra>",
                customdata=np.column_stack((completeness_df_sorted['Missing Count'], 
                                          completeness_df_sorted['Quality Level'])),
                name="Completeness"
            ),
            row=1, col=1
        )
        
        # 2. Quality distribution pie chart with gradient
        quality_counts = completeness_df['Quality Level'].value_counts()
        quality_colors = [self.colors[level.lower()] for level in quality_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=quality_counts.index,
                values=quality_counts.values,
                marker=dict(
                    colors=quality_colors,
                    line=dict(color='white', width=3)
                ),
                textinfo='label+percent',
                textfont=dict(size=12, color='white', family='Arial Black'),
                hole=0.4,  # Donut chart
                hovertemplate="<b>🎯 %{label}</b><br>" +
                             "📊 Columns: <b>%{value}</b><br>" +
                             "📈 Percentage: <b>%{percent}</b>" +
                             "<extra></extra>",
                pull=[0.1 if level == 'Critical' else 0 for level in quality_counts.index]  # Emphasize critical
            ),
            row=1, col=2
        )
        
        # 3. Trend line chart
        sorted_completeness = completeness_df.sort_values('Completeness (%)', ascending=False)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sorted_completeness))),
                y=sorted_completeness['Completeness (%)'],
                mode='lines+markers',
                line=dict(color='#3498db', width=4, shape='spline'),
                marker=dict(
                    size=8,
                    color=sorted_completeness['Completeness (%)'],
                    colorscale='RdYlGn',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                hovertemplate="<b>📊 Column Rank:</b> %{x}<br>" +
                             "<b>✅ Completeness:</b> %{y:.1f}%<br>" +
                             "<extra></extra>",
                name="Trend"
            ),
            row=2, col=1
        )
        
        # 4. Summary table
        summary_data = self.generate_detailed_report(table_name)
        
        table_data = [
            ['📊 Total Columns', str(summary_data['total_columns'])],
            ['📈 Average Completeness', f"{summary_data['avg_completeness']:.1f}%"],
            ['✅ Excellent Columns', str(summary_data['excellent_columns'])],
            ['⚠️ Issues Need Attention', str(summary_data['poor_columns'] + summary_data['critical_columns'])],
            ['🚨 Critical Issues', str(summary_data['critical_columns'])]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#34495e',
                    font=dict(color='white', size=12, family='Arial Black'),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['#ecf0f1', '#d5dbdb'] * len(table_data)],
                    font=dict(color='#2c3e50', size=11),
                    align=['left', 'center']
                )
            ),
            row=2, col=2
        )
        
        # Update layout with modern styling
        fig.update_layout(
            title=dict(
                text=f"<b>📈 Interactive Data Quality Dashboard</b><br><span style='font-size:16px; color:#7f8c8d'>Table: {table_name}</span>",
                x=0.5,
                font=dict(size=26, color="#2c3e50", family="Arial Black"),
                pad=dict(b=24)
            ),
            template=self.template,
            height=height,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                font=dict(color="white", size=12, family="Arial")
            )
        )
        
        # Style individual subplots
        fig.update_xaxes(
            title=dict(text="<b>Completeness (%)</b>", font=dict(size=14)),
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            row=1, col=1
        )
        
        fig.update_yaxes(
            title=dict(text="<b>Columns</b>", font=dict(size=14)),
            showgrid=False,
            row=1, col=1
        )
        
        fig.update_xaxes(
            title=dict(text="<b>Column Rank</b>", font=dict(size=12)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            row=2, col=1
        )
        
        fig.update_yaxes(
            title=dict(text="<b>Completeness (%)</b>", font=dict(size=12)),
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            row=2, col=1
        )
        
        fig.show(config={"displayModeBar": False}, renderer='notebook_connected')
        
        # Save if needed
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                print(f"📁 Interactive dashboard saved to: {save_path}")
    
    def generate_detailed_report(self, table_name: str) -> dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            table_name (str): Name of the table to analyze
            
        Returns:
            dict: Detailed quality metrics
        """
        completeness_df = self.get_completeness(table_name, plot=False)
        
        if completeness_df.empty:
            return {}
        
        # Calculate detailed metrics
        report = {
            'table_name': table_name,
            'total_rows': completeness_df['Total Rows'].iloc[0],
            'total_columns': len(completeness_df),
            'avg_completeness': completeness_df['Completeness (%)'].mean(),
            'min_completeness': completeness_df['Completeness (%)'].min(),
            'max_completeness': completeness_df['Completeness (%)'].max(),
            'excellent_columns': len(completeness_df[completeness_df['Completeness (%)'] >= 95]),
            'good_columns': len(completeness_df[(completeness_df['Completeness (%)'] >= 80) & 
                                              (completeness_df['Completeness (%)'] < 95)]),
            'fair_columns': len(completeness_df[(completeness_df['Completeness (%)'] >= 60) & 
                                              (completeness_df['Completeness (%)'] < 80)]),
            'poor_columns': len(completeness_df[(completeness_df['Completeness (%)'] >= 40) & 
                                              (completeness_df['Completeness (%)'] < 60)]),
            'critical_columns': len(completeness_df[completeness_df['Completeness (%)'] < 40]),
            'worst_columns': completeness_df.nsmallest(5, 'Completeness (%)')[['Column', 'Completeness (%)']].to_dict('records'),
            'best_columns': completeness_df.nlargest(5, 'Completeness (%)')[['Column', 'Completeness (%)']].to_dict('records')
        }
        
        return report
    
    def print_summary(self, table_name: str):
        """Print a formatted summary report with emojis."""
        report = self.generate_detailed_report(table_name)
        
        if not report:
            print(f"❌ No data available for table: {table_name}")
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
        print(f"   ✔️  Good (80-94%): {report['good_columns']} columns") 
        print(f"   ⚠️  Fair (60-79%): {report['fair_columns']} columns")
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

# Example usage for Jupyter Notebook:
"""
# Install required packages first:
# !pip install plotly

# Initialize the report
dq_report = DataQualityReport("your_connection_string_here")

# Method 1: Simple interactive completeness report
completeness_data = dq_report.get_completeness("your_table_name")

# Method 2: Full interactive dashboard
dq_report.create_animated_dashboard("your_table_name")

# Method 3: Print detailed summary
dq_report.print_summary("your_table_name")
"""