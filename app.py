import yfinance as yf
import streamlit as st
import pandas as pd

from datetime import datetime

st.write("""
# Next Time Frame Prediction App
""")

symbol = 'GOOG'

yf_obj = yf.Ticker(symbol)

# Get current date
date_stamp = datetime.now().strftime('%Y-%m-%d')

df = yf_obj.history(period='1d', start='2010-01-01', end=date_stamp)

st.line_chart(df.Close)
st.line_chart(df.Volume)