import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Day Parting Converter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, light CSS with your brand colors
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #334155;
    }
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    h1 {
        color: #47d495 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(71, 212, 149, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #6f58c9;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(71, 212, 149, 0.2);
        box-shadow: 0 4px 20px rgba(71, 212, 149, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #47d495;
        box-shadow: 0 8px 30px rgba(71, 212, 149, 0.15);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #47d495, #6f58c9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 10px rgba(71, 212, 149, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(71, 212, 149, 0.4) !important;
    }
    
    .stSelectbox > div > div {
        background-color: white !important;
        border: 2px solid rgba(71, 212, 149, 0.3) !important;
        border-radius: 8px !important;
        color: #334155 !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #f8fafc !important;
        border: 2px solid rgba(71, 212, 149, 0.3) !important;
        border-radius: 8px !important;
        color: #334155 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #47d495 !important;
        box-shadow: 0 0 0 3px rgba(71, 212, 149, 0.1) !important;
    }
    
    .stExpander {
        background-color: white !important;
        border: 1px solid rgba(111, 88, 201, 0.2) !important;
        border-radius: 8px !important;
    }
    
    .success-banner {
        background: linear-gradient(90deg, rgba(71, 212, 149, 0.1) 0%, rgba(152, 193, 217, 0.05) 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #47d495;
        margin: 1rem 0;
        color: #334155;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(238, 108, 77, 0.05) 0%, rgba(152, 193, 217, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(238, 108, 77, 0.3);
        margin: 1rem 0;
        color: #334155;
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    .badge-excellent {
        background: #47d495;
        color: white;
    }
    
    .badge-good {
        background: #98c1d9;
        color: white;
    }
    
    .badge-poor {
        background: #ee6c4d;
        color: white;
    }
    
    .stDataFrame {
        border: 1px solid rgba(71, 212, 149, 0.2);
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1>⚡ Day Parting Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform your hourly PPC data into actionable day-parted insights</p>', unsafe_allow_html=True)

# Day part mappings
DAY_PARTS = {
    'Early Hours (0-5)': list(range(0, 6)),
    'Pre-Work/Commute (6-8)': list(range(6, 9)),
    'Work AM (9-11)': list(range(9, 12)),
    'Lunch (12-13)': list(range(12, 14)),
    'Work Afternoon (14-18)': list(range(14, 19)),
    'Evening/Night (19-23)': list(range(19, 24))
}

def get_performance_badge(metric_name, value, benchmarks):
    """Generate performance badges based on industry benchmarks"""
    if metric_name == 'CTR':
        if value >= benchmarks['ctr_excellent']:
            return f'<span class="performance-badge badge-excellent">🎯 Excellent CTR</span>'
        elif value >= benchmarks['ctr_good']:
            return f'<span class="performance-badge badge-good">✅ Good CTR</span>'
        else:
            return f'<span class="performance-badge badge-poor">⚠️ Low CTR</span>'
    elif metric_name == 'CPC':
        if value <= benchmarks['cpc_excellent']:
            return f'<span class="performance-badge badge-excellent">💰 Low CPC</span>'
        elif value <= benchmarks['cpc_good']:
            return f'<span class="performance-badge badge-good">📊 Average CPC</span>'
        else:
            return f'<span class="performance-badge badge-poor">💸 High CPC</span>'
    return ''

def process_hourly_data(df):
    """Convert hourly data to day parting with enhanced analytics"""
    df = df.copy()
    
    # Convert percentages to decimals for calculation
    if 'CTR' in df.columns:
        if df['CTR'].dtype == 'object':
            df['CTR'] = df['CTR'].str.replace('%', '').astype(float) / 100
    
    results = []
    hourly_performance = []
    
    for day_part, hours in DAY_PARTS.items():
        day_part_data = df[df['Hour of the day'].isin(hours)]
        
        if day_part_data.empty:
            continue
            
        # Store hourly data for charts
        for _, row in day_part_data.iterrows():
            hourly_performance.append({
                'Hour': row['Hour of the day'],
                'Day Part': day_part,
                'Cost': row['Cost'],
                'Clicks': row['Clicks'],
                'CTR': row['CTR'] * 100 if isinstance(row['CTR'], float) else float(str(row['CTR']).replace('%', '')),
                'CPC': row['Avg. CPC'] if 'Avg. CPC' in row else row.get('Avg CPC', 0)
            })
        
        # Aggregate metrics
        total_cost = day_part_data['Cost'].sum()
        total_clicks = day_part_data['Clicks'].sum()
        total_impressions = day_part_data['Impr.'].sum()
        total_conversions = day_part_data['Conversions'].sum()
        
        # Calculate derived metrics
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        avg_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
        cost_per_conv = (total_cost / total_conversions) if total_conversions > 0 else 0
        
        results.append({
            'Day Parting': day_part,
            'Cost/Conv': cost_per_conv,
            'Clicks': int(total_clicks),
            'Impr': int(total_impressions),
            'CTR': ctr,
            'CTR_Display': f"{ctr:.2f}%",
            'Avg CPC': avg_cpc,
            'Cost': total_cost,
            'Conversions': total_conversions,
            'Hours': f"{min(hours)}-{max(hours)}"
        })
    
    return pd.DataFrame(results), pd.DataFrame(hourly_performance)

def create_performance_charts(day_part_df, hourly_df):
    """Create enhanced visualizations"""
    
    # Color scheme
    colors = {
        'primary': '#47d495',
        'secondary': '#98c1d9',
        'accent': '#ee6c4d',
        'purple': '#6f58c9',
        'dark': '#111111'
    }
    
    # 1. Day Part Performance Overview
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost by Day Part', 'CTR by Day Part', 'Clicks Distribution', 'Cost per Conversion'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cost bars
    fig1.add_trace(
        go.Bar(x=day_part_df['Day Parting'], y=day_part_df['Cost'],
               name='Cost', marker_color=colors['primary'],
               text=[f'£{x:.0f}' for x in day_part_df['Cost']],
               textposition='outside'),
        row=1, col=1
    )
    
    # CTR line
    fig1.add_trace(
        go.Bar(x=day_part_df['Day Parting'], y=day_part_df['CTR'],
               name='CTR', marker_color=colors['accent'],
               text=[f'{x:.1f}%' for x in day_part_df['CTR']],
               textposition='outside'),
        row=1, col=2
    )
    
    # Clicks distribution (bar instead of pie)
    fig1.add_trace(
        go.Bar(x=day_part_df['Day Parting'], y=day_part_df['Clicks'],
               name='Clicks', marker_color=colors['secondary'],
               text=[f'{x:,}' for x in day_part_df['Clicks']],
               textposition='outside'),
        row=2, col=1
    )
    
    # Cost per conversion
    fig1.add_trace(
        go.Bar(x=day_part_df['Day Parting'], y=day_part_df['Cost/Conv'],
               name='Cost/Conv', marker_color=colors['purple'],
               text=[f'£{x:.0f}' for x in day_part_df['Cost/Conv']],
               textposition='outside'),
        row=2, col=2
    )
    
    fig1.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,250,252,0.5)',
        font=dict(color='#334155', family='Arial'),
        title_text="📊 Day Parting Performance Dashboard",
        title_x=0.5,
        title_font_size=20,
        title_font_color=colors['primary']
    )
    
    # Update all subplot backgrounds
    fig1.update_xaxes(showgrid=True, gridcolor='rgba(71, 212, 149, 0.2)')
    fig1.update_yaxes(showgrid=True, gridcolor='rgba(71, 212, 149, 0.2)')
    
    # 2. Hourly Trend Analysis
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=hourly_df['Hour'], y=hourly_df['Cost'],
        mode='lines+markers',
        name='Cost',
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=8, color=colors['primary'])
    ))
    
    fig2.add_trace(go.Scatter(
        x=hourly_df['Hour'], y=hourly_df['CTR'],
        mode='lines+markers',
        name='CTR (%)',
        yaxis='y2',
        line=dict(color=colors['accent'], width=3),
        marker=dict(size=8, color=colors['accent'])
    ))
    
    fig2.update_layout(
        title="⏰ Hourly Performance Trends",
        title_font_color=colors['primary'],
        title_font_size=18,
        xaxis=dict(title="Hour of Day", color='#334155', showgrid=True, gridcolor='rgba(71, 212, 149, 0.2)'),
        yaxis=dict(title="Cost (£)", side='left', color=colors['primary']),
        yaxis2=dict(title="CTR (%)", side='right', overlaying='y', color=colors['accent']),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,250,252,0.5)',
        font=dict(color='#334155', family='Arial'),
        height=400,
        hovermode='x unified'
    )
    
    return fig1, fig2

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    input_method = st.selectbox(
        "🔗 Choose input method:",
        ["📋 Paste data", "📁 Upload CSV file"],
        key="input_method"
    )

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style='color: #47d495; margin: 0;'>💡 Quick Tip</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #98c1d9;'>
        Export from Google Ads: Dimensions → Time → Hour of day
        </p>
    </div>
    """, unsafe_allow_html=True)

if input_method == "📋 Paste data":
    with st.expander("📋 Expected format", expanded=False):
        st.code("""Hour of the day	Cost / conv.	Clicks	Impr.	CTR	Avg. CPC	Cost	Conversions
0	35.74	27	274	9.85%	9.27	250.16	7
1	147.42	21	229	9.17%	7.02	147.42	1
...""", language="text")
    
    pasted_data = st.text_area(
        "📥 Paste your Google Ads data here:",
        height=200,
        placeholder="Paste your hourly performance data here (tab-separated with headers)...",
        key="data_input"
    )
    
    if pasted_data:
        try:
            lines = pasted_data.strip().split('\n')
            headers = lines[0].split('\t')
            data_rows = []
            
            for line in lines[1:]:
                row = line.split('\t')
                if len(row) == len(headers):
                    data_rows.append(row)
            
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns
            numeric_cols = ['Hour of the day', 'Cost / conv.', 'Clicks', 'Impr.', 'Avg. CPC', 'Cost', 'Conversions']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.rename(columns={'Cost / conv.': 'Cost/Conv', 'Avg. CPC': 'Avg. CPC'})
            
            st.markdown(f"""
            <div class="success-banner">
                <h4 style='margin: 0; color: #47d495;'>✅ Data loaded successfully!</h4>
                <p style='margin: 0.5rem 0 0 0; color: #98c1d9;'>Processed {len(df)} hours of campaign data</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error parsing data: {str(e)}")
            df = None

else:  # Upload CSV
    uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"""
            <div class="success-banner">
                <h4 style='margin: 0; color: #47d495;'>✅ File uploaded successfully!</h4>
                <p style='margin: 0.5rem 0 0 0; color: #98c1d9;'>Loaded {len(df)} rows from {uploaded_file.name}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
            df = None

# Process and display results
if 'df' in locals() and df is not None:
    
    # Show original data preview
    with st.expander("👀 Preview original data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    try:
        day_part_df, hourly_df = process_hourly_data(df)
        
        if not day_part_df.empty:
            st.markdown("---")
            
            # Performance insights
            total_cost = day_part_df['Cost'].sum()
            total_conversions = day_part_df['Conversions'].sum()
            best_ctr_period = day_part_df.loc[day_part_df['CTR'].idxmax(), 'Day Parting']
            best_cost_conv_period = day_part_df.loc[day_part_df['Cost/Conv'].idxmin(), 'Day Parting']
            
            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #47d495; margin: 0; font-size: 1.5rem;'>£{total_cost:,.0f}</h3>
                    <p style='color: #98c1d9; margin: 0.2rem 0 0 0; font-size: 0.9rem;'>Total Spend</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #47d495; margin: 0; font-size: 1.5rem;'>{total_conversions:,.0f}</h3>
                    <p style='color: #98c1d9; margin: 0.2rem 0 0 0; font-size: 0.9rem;'>Total Conversions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #ee6c4d; margin: 0; font-size: 1.2rem;'>{best_ctr_period.split('(')[0]}</h3>
                    <p style='color: #98c1d9; margin: 0.2rem 0 0 0; font-size: 0.9rem;'>Highest CTR Period</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #6f58c9; margin: 0; font-size: 1.2rem;'>{best_cost_conv_period.split('(')[0]}</h3>
                    <p style='color: #98c1d9; margin: 0.2rem 0 0 0; font-size: 0.9rem;'>Best Cost/Conv Period</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            fig1, fig2 = create_performance_charts(day_part_df, hourly_df)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Enhanced results table
            st.markdown("### 📊 Day Parting Analysis")
            
            # Format the dataframe for better display
            display_df = day_part_df.copy()
            display_df['Cost'] = display_df['Cost'].apply(lambda x: f"£{x:,.2f}")
            display_df['Cost/Conv'] = display_df['Cost/Conv'].apply(lambda x: f"£{x:,.2f}")
            display_df['Avg CPC'] = display_df['Avg CPC'].apply(lambda x: f"£{x:.2f}")
            display_df['CTR_Display'] = display_df['CTR'].apply(lambda x: f"{x:.2f}%")
            display_df = display_df.drop(['CTR'], axis=1)
            display_df = display_df.rename(columns={'CTR_Display': 'CTR'})
            
            # Reorder columns
            display_df = display_df[['Day Parting', 'Hours', 'Cost', 'Clicks', 'Impr', 'CTR', 'Avg CPC', 'Cost/Conv', 'Conversions']]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Strategic insights
            st.markdown("""
            <div class="insight-box">
                <h4 style='color: #ee6c4d; margin: 0 0 1rem 0;'>🎯 Strategic Recommendations</h4>
                <ul style='color: #98c1d9; margin: 0;'>
                    <li><strong>Budget Allocation:</strong> Focus spend on highest converting periods</li>
                    <li><strong>Bid Adjustments:</strong> Increase bids during peak performance windows</li>
                    <li><strong>Ad Scheduling:</strong> Consider pausing underperforming time slots</li>
                    <li><strong>Creative Testing:</strong> Test different messaging for different day parts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = day_part_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="day_parting_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                try:
                    from io import BytesIO
                    buffer = BytesIO()
                    day_part_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=buffer,
                        file_name="day_parting_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.info("💡 Install openpyxl for Excel export")
            
            with col3:
                with st.expander("📋 Copy-paste format"):
                    copyable_text = day_part_df.to_csv(sep='\t', index=False)
                    st.text_area(
                        "Tab-separated (Excel/Sheets ready):",
                        value=copyable_text,
                        height=200
                    )
                
        else:
            st.warning("⚠️ No data to process. Check your input format.")
            
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")

# Footer with usage instructions
with st.expander("ℹ️ Usage Guide", expanded=False):
    st.markdown("""
    ### 🚀 How to use this tool:
    
    **1. Export from Google Ads:**
    - Navigate to Campaigns → Dimensions → Time → Hour of day
    - Select your date range and download the data
    
    **2. Input your data:**
    - Paste directly or upload CSV file
    - Ensure all required columns are included
    
    **3. Analyse results:**
    - View interactive charts and insights
    - Download processed data in multiple formats
    - Use recommendations for campaign optimisation
    
    **📈 Day Part Definitions:**
    - **Early Hours (0-5):** Midnight to 5 AM - Low volume, often mobile
    - **Pre-Work/Commute (6-8):** 6-8 AM - Mobile heavy, commuter searches  
    - **Work AM (9-11):** 9-11 AM - Peak B2B, high intent
    - **Lunch (12-13):** 12-1 PM - Quick mobile searches
    - **Work Afternoon (14-18):** 2-6 PM - Extended business hours
    - **Evening/Night (19-23):** 7-11 PM - Consumer research, leisure
    """)
