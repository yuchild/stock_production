import yfinance as yf
import streamlit as st
import pandas as pd

from src.functions import dl_tf_pd, load

st.write("""
# Next Time Frame Prediction App
""")
# Text input
symbol = st.text_input("Enter your stock symbol in caps:").upper()

# Display the selected option
st.write(f"You selected stock: {symbol}")

# Dropdown menu for intervals
interval_options = ['5m', '15m','30m', '1h', '1d', '1wk']
selected_interval_option = st.selectbox("Choose an interval option:", interval_options)

# Display the selected interval option
st.write(f"You selected interval: {selected_interval_option}")

# Dropdown menu for period option
period_options = ['1mo', '3mo','1y', 'max']
selected_period_option = st.selectbox("Choose an period option:", period_options)

# Display the selected period option
st.write(f"You selected interval: {selected_period_option}")

# Run download, transform, and modeling
time_stamp, summary_table = dl_tf_pd(symbol, 
                                       selected_interval_option, 
                                       selected_period_option, 
                                       skip_dl=False,
                                      )

# Display the current time
st.write(f"Current Time (EST): {time_stamp}")

# Display results summary table
st.table(summary_table)

# Load stock dataframe
stock_df = load(symbol, selected_interval_option)

# display line chart of Close values
st.line_chart(stock_df.Close)

# display line chart of Volume values
st.line_chart(stock_df.Volume)