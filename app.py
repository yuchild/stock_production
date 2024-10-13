import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px

from src.functions import dl_tf_pd, load

# Container 1
with st.container():
    st.markdown(f"""
#### Next Stock Interval Time Frame Prediction App 
TL;DR

Select your stock and time interval for predictions. The current time is used for the latest data sample, which is then passed through the model to predict the next interval's movement.

For example, if you select a 5-minute interval at 10:14 AM EST, the model will predict the stock's movement between 10:15 AM and 10:20 AM.

To maximize prediction accuracy, run the model as close to the end of the current interval as possible, using the most up-to-date information.

Note: The model predicts stock movement (up, static, or down) in the next interval, not the exact price.

    """)
    
    
# Container 2
with st.container(): 
    col1, col2 = st.columns([1,1])
    # Text input
    symbol = col1.text_input("Enter stock symbol in caps:", value='NVDA').upper()

    # Display the selected option
    col1.write(f"You selected stock: {symbol}\n")

    # Dropdown menu for intervals
    interval_options = ['5m', '15m','30m', '1h', '1d', '1wk']
    selected_interval_option = col2.selectbox("Choose an interval option:", interval_options, index=0)

    # Display the selected interval option
    col2.write(f'You selected interval: {selected_interval_option}\n')
    
# Container 3
with st.container():
    # Run download, transform, and modeling
    time_stamp, summary_table = dl_tf_pd(symbol, 
                                         selected_interval_option, 
                                         skip_dl=False,
                                        )

    # Display the current time
    st.write(f"Current Time (EST): {time_stamp}\n")

    # Display results summary table
    col_headers = summary_table.model.tolist()
    summary_table = summary_table.round(6)
    summary_table_transposed = summary_table.T.iloc[1:]
    summary_table_transposed.columns = col_headers
    st.table(summary_table_transposed)
    
    
# Container 4
with st.container():
    col1, col2 = st.columns([1,1])
    
    # Load stock dataframe
    stock_df = load(symbol, selected_interval_option).iloc[-200:]

    # Reset the index to turn 'DatetimeIndex' from index to a 'date' column
    stock_df = stock_df.reset_index().rename(columns={'Date': 'date',
                                                      'Datetime': 'date',
                                                     })
    
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



