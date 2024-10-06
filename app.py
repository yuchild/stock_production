import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px

from src.functions import dl_tf_pd, load

# Container 1
with st.container():
    st.markdown(f"""
    ##Next Stock Interval Time Frame Prediction App
    
    """)
    
    
# Container 2
with st.container(): 
    col1, col2, col3 = st.columns([1,2,3])
    # Text input
    symbol = col1.text_input("Enter your stock symbol in caps:", value='NVDA').upper()

    # Display the selected option
    col1.write(f"You selected stock: {symbol}\n")

    # Dropdown menu for intervals
    interval_options = ['5m', '15m','30m', '1h', '1d', '1wk']
    selected_interval_option = col2.selectbox("Choose an interval option:", interval_options, index=0)

    # Display the selected interval option
    col2.write(f'You selected interval: {selected_interval_option}\n')

    # Dropdown menu for period option
    period_options = ['1mo', '3mo','1y', 'max']
    selected_period_option = col3.selectbox("Choose an period option:", period_options, index=0)

    # Display the selected period option
    col3.write(f'You selected period: {selected_period_option}')

    
# Container 3
with st.container():
    # Run download, transform, and modeling
    time_stamp, summary_table = dl_tf_pd(symbol, 
                                         selected_interval_option, 
                                         selected_period_option, 
                                         skip_dl=False,
                                        )

    # Display the current time
    st.write(f"Current Time (EST): {time_stamp}\n")

    # Display results summary table
    st.table(summary_table.T)
    
    
# Container 4
with st.container():
    col1, col2 = st.columns([1,2])
    
    # Load stock dataframe
    stock_df = load(symbol, selected_interval_option).iloc[-200:]

    # Reset the index to turn 'DatetimeIndex' from index to a 'date' column
    stock_df = stock_df.reset_index().rename(columns={'Datetime': 'date'})
    
    # display line chart of Close values
    col1.write(f'{symbol} Price, last 200 intervals')    
    fig = px.line(stock_df, x='date',  y='adj_close')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True
    )
    col1.plotly_chart(fig)

    # display line chart of Volume values
    col2.write(f'{symbol} Volume, last 200 intervals')
    fig = px.line(stock_df, x='date',  y='volume')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Volume',
        xaxis_rangeslider_visible=True
    )
    col2.plotly_chart(fig)



