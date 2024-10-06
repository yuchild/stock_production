import yfinance as yf
import streamlit as st
import pandas as pd

from src.functions import dl_tf_pd, load


col1, col2, col3 = st.columns([1,2,1])

col1.markdown("""
## Next Time Stock Interval Time Frame Prediction App
""")
# Text input
symbol = col1.text_input("Enter your stock symbol in caps:", value='NVDA').upper()

# Display the selected option
col2.write(f"You selected stock: {symbol}")

# Dropdown menu for intervals
interval_options = ['5m', '15m','30m', '1h', '1d', '1wk']
selected_interval_option = col1.selectbox("Choose an interval option:", interval_options, index=0)

# Display the selected interval option
col2.write(f"You selected interval: {selected_interval_option}")

# Dropdown menu for period option
period_options = ['1mo', '3mo','1y', 'max']
selected_period_option = Col1.selectbox("Choose an period option:", period_options, index=0)

# Display the selected period option
col2.write(f"You selected period: {selected_period_option}")

# Run download, transform, and modeling
time_stamp, summary_table = dl_tf_pd(symbol, 
                                     selected_interval_option, 
                                     selected_period_option, 
                                     skip_dl=False,
                                    )

# Display the current time
col1.write(f"Current Time (EST): {time_stamp}")

# Display results summary table
col2.table(summary_table)

# Load stock dataframe
stock_df = load(symbol, selected_interval_option)

# display line chart of Close values
col3.line_chart(stock_df.adj_close.iloc[-200])

# display line chart of Volume values
col3.line_chart(stock_df.volume.iloc[-200])




