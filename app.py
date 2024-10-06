import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px

from src.functions import dl_tf_pd, load

# Container 1
with st.container(): 
    col1, col2 = st.columns([0.5,2])

    col1.markdown("""
    ### Next Time Stock Interval Time Frame Prediction App
    """)
    # Text input
    symbol = col1.text_input("Enter your stock symbol in caps:", value='NVDA').upper()

    # Display the selected option
    col1.write(f"You selected stock: {symbol}\n")

    # Dropdown menu for intervals
    interval_options = ['5m', '15m','30m', '1h', '1d', '1wk']
    selected_interval_option = col1.selectbox("Choose an interval option:", interval_options, index=0)

    # Display the selected interval option
    col1.write(f'You selected interval: {selected_interval_option}\n')

    # Dropdown menu for period option
    period_options = ['1mo', '3mo','1y', 'max']
    selected_period_option = col1.selectbox("Choose an period option:", period_options, index=0)

    # Display the selected period option
    col1.write(f'You selected period: {selected_period_option}')

    # Run download, transform, and modeling
    time_stamp, summary_table = dl_tf_pd(symbol, 
                                         selected_interval_option, 
                                         selected_period_option, 
                                         skip_dl=False,
                                        )
    # Load stock dataframe
    stock_df = load(symbol, selected_interval_option)
    
    # Reset the index to turn 'date' from index to a column
    stock_df = stock_df.reset_index().rename(columns={'index': 'date'})

    # display line chart of Close values
    col2.write(f'{symbol} Price, Last 200 Intervals')    
    fig = px.line(stock_df, x='date',  y='adj_close')
    plotly_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis_rangeslider_visible=True
    )
    col2.plotly_chart(plotly_fig)

    # display line chart of Volume values
    col2.write(f'{symbol} Volume, Last 200 Intervals')
    fig = px.line(stock_df, x='date',  y='volume')
    plotly_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis_rangeslider_visible=True
    )
    col2.plotly_chart(plotly_fig)

# Container 2 
with st.container():
    # Display the current time
    col2.write(f"Current Time (EST): {time_stamp}")

    # Display results summary table
    col2.table(summary_table)





