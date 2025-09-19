import streamlit as st
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="Day Parting Converter",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Hourly to Day Parting Converter")
st.markdown("Convert Google Ads hourly data into day-parted segments for easier analysis")

# Day part mappings
DAY_PARTS = {
    'Early Hours (0-5)': list(range(0, 6)),
    'Pre-Work/Commute (6-8)': list(range(6, 9)),
    'Work AM (9-11)': list(range(9, 12)),
    'Lunch (12-13)': list(range(12, 14)),
    'Work Afternoon (14-18)': list(range(14, 19)),
    'Evening/Night (19-23)': list(range(19, 24))
}

def process_hourly_data(df):
    """Convert hourly data to day parting"""
    
    # Ensure proper column names and data types
    df = df.copy()
    
    # Convert percentages to decimals for calculation
    if 'CTR' in df.columns:
        if df['CTR'].dtype == 'object':  # Percentage strings
            df['CTR'] = df['CTR'].str.replace('%', '').astype(float) / 100
    
    results = []
    
    for day_part, hours in DAY_PARTS.items():
        # Filter data for this day part
        day_part_data = df[df['Hour of the day'].isin(hours)]
        
        if day_part_data.empty:
            continue
            
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
            'CTR': f"{ctr:.2f}%",
            'Avg CPC': avg_cpc,
            'Cost': total_cost,
            'Conversions': total_conversions
        })
    
    return pd.DataFrame(results)

# Input methods
input_method = st.radio(
    "Choose input method:",
    ["Paste data", "Upload CSV file"]
)

if input_method == "Paste data":
    st.subheader("Paste your hourly data")
    
    # Show expected format
    with st.expander("Expected format"):
        st.code("""Hour of the day	Cost / conv.	Clicks	Impr.	CTR	Avg. CPC	Cost	Conversions
0	35.74	27	274	9.85%	9.27	250.16	7
1	147.42	21	229	9.17%	7.02	147.42	1
...""")
    
    # Text area for pasting data
    pasted_data = st.text_area(
        "Paste your data here (tab-separated with headers):",
        height=200,
        placeholder="Paste your Google Ads hourly data here..."
    )
    
    if pasted_data:
        try:
            # Convert pasted data to DataFrame
            lines = pasted_data.strip().split('\n')
            
            # Parse headers and data
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
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Cost / conv.': 'Cost/Conv',
                'Impr.': 'Impr.',
                'Avg. CPC': 'Avg CPC'
            })
            
            st.success(f"✅ Loaded {len(df)} hours of data")
            
        except Exception as e:
            st.error(f"Error parsing data: {str(e)}")
            df = None

else:  # Upload CSV
    st.subheader("Upload CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows from CSV")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            df = None

# Process and display results
if 'df' in locals() and df is not None:
    
    # Show original data preview
    with st.expander("Preview original data"):
        st.dataframe(df.head(10))
    
    # Convert to day parting
    try:
        day_part_df = process_hourly_data(df)
        
        if not day_part_df.empty:
            st.subheader("📈 Day Parting Results")
            
            # Display results table
            st.dataframe(
                day_part_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Cost/Conv': st.column_config.NumberColumn(format="%.2f"),
                    'Avg CPC': st.column_config.NumberColumn(format="%.2f"),
                    'Cost': st.column_config.NumberColumn(format="%.2f"),
                    'Conversions': st.column_config.NumberColumn(format="%.1f")
                }
            )
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = day_part_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="day_parting_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download (requires openpyxl)
                try:
                    from io import BytesIO
                    buffer = BytesIO()
                    day_part_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=buffer,
                        file_name="day_parting_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.info("Install openpyxl for Excel export: pip install openpyxl")
            
            # Copy-paste friendly format
            with st.expander("Copy-paste friendly format"):
                copyable_text = day_part_df.to_csv(sep='\t', index=False)
                st.text_area(
                    "Tab-separated format (ready for Excel/Sheets):",
                    value=copyable_text,
                    height=200
                )
                
        else:
            st.warning("No data to process. Check your input format.")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Make sure your data has the expected columns: Hour of the day, Cost, Clicks, Impr., CTR, Avg. CPC, Conversions")

# Usage instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Export data from Google Ads:**
       - Go to Campaigns > Dimensions > Time > Hour of day
       - Select your desired date range and metrics
       - Download or copy the data
    
    2. **Input your data:**
       - Either paste directly into the text area
       - Or upload as a CSV file
    
    3. **Get results:**
       - View the day-parted analysis
       - Download as CSV or Excel
       - Copy tab-separated format for pasting into other tools
    
    **Day Part Definitions:**
    - Early Hours (0-5): Hours 0-5
    - Pre-Work/Commute (6-8): Hours 6-8  
    - Work AM (9-11): Hours 9-11
    - Lunch (12-13): Hours 12-13
    - Work Afternoon (14-18): Hours 14-18
    - Evening/Night (19-23): Hours 19-23
    """)
