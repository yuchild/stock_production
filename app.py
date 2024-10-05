import yfinance as yf
import streamlit as st
import pandas as pd

from datetime import datetime
import pytz

from src import functions as f

%load_ext autoreload
%autoreload 2

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

# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')

# Get current time in Eastern Time Zone
eastern_time = datetime.now(eastern)

# Format the time to include hour, minute, and seconds
time_stamp = eastern_time.strftime('%Y-%m-%d %H:%M:%S')

# Display the current time
st.write(f"Current Time (EST): {time_stamp}")

stock = yf.Ticker(symbol)

stock_df = stock.history(interval=selected_interval_option,
                         period=selected_period_option,
                         auto_adjust=False,
                         prepost=True, # include aftermarket hours
                        )


st.line_chart(stock_df.Close)
st.line_chart(stock_df.Volume)