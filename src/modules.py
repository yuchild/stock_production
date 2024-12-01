#!usr/bin/env python3


import pandas as pd
import numpy as np
from IPython.display import display
from yfinance import Ticker
from pykalman import KalmanFilter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import cross_val_predict

from prophet import Prophet

import logging

# Set logging level for cmdstanpy to WARNING to suppress INFO messages
# logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Disable cmdstanpy logging
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

# from cmdstanpy import install_cmdstan
# install_cmdstan()

import os

# Suppress cmdstanpy output
os.environ["STAN_NUM_THREADS"] = "1"  # If using threading, limit to 1
os.environ["CMDSTANPY_SILENT"] = "1"

# Set the number of threads to use
os.environ['STAN_BACKEND'] = 'CMDSTANPY'  # Ensure the correct backend
os.environ['STAN_NUM_THREADS'] = '4'  # Number of cores



from datetime import datetime, timedelta
import pytz

################################################
# functions for downloading and loading tables #
################################################

def download(symbol, interval):
    
    stock = Ticker(symbol)
    
    if interval in {'5m','15m','30m','1h',}:
        interval_period_map = {'5m':58,
                               '15m':58,
                               '30m':58,
                               '1h':728,
                              }
        today = datetime.today().date()
        start = today - timedelta(days=interval_period_map[interval])
        stock_df = stock.history(interval=interval,
                                 start=str(start),
                                 end=None,
                                 # period=period,
                                 auto_adjust=False,
                                 prepost=True, # include aftermarket hours
                                )
        
    else:
        stock_df = stock.history(interval=interval,
                         period='max',
                         auto_adjust=False,
                         prepost=True, # include aftermarket hours
                        )
    
    stock_df.columns = stock_df.columns.str.lower().str.replace(' ', '_')
    stock_df.to_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')
    
def load(symbol, interval):
    return pd.read_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')

def load_model_df(symbol, interval):
    return pd.read_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')

###########################################
# functions for use to transform features #
###########################################

# candle parts percentages
def candle_parts_pcts(o, c, h, l):
    full = h - l
    if full == 0:
        # If full is zero, return 0 for all components to avoid division by zero
        return 0, 0, 0
    body = abs(o - c)
    if o > c:
        top_wick = h - o
        bottom_wick = c - l
    else:
        top_wick = h - c
        bottom_wick = o - l
    return top_wick / full, body / full, bottom_wick / full

# previous close and open gap % of pervious candle size
def gap_up_down_pct(o, pc, ph, pl):
    if (o == pc) or (ph == pl):
        return 0
    else:
        return (o - pc) / (ph - pl)
    
    
# z-score calculation
def zscore(x, mu, stdev):
    if stdev == 0:
        return 0
    else:
        return (x - mu) / stdev

# DEPRECATED
# direction calculation: 
# def direction(pctc, mean, stdev):
    
#     pct_pos = mean + 0.43073 / 2 * stdev
#     pct_neg = mean - 0.43073 / 2 * stdev
    
#     if pctc >= pct_pos:
#         return 1
#     elif pctc <= pct_neg:
#         return 2
#     else:
#         return 0

# compute kelly criterion
def kelly_c(p, l=1, g=2.5):     
    return list(map(lambda x:(x / l - (1 - x) / g), p))

#################################
# functions for modeling output #
#################################

def transform(symbol, interval):
    
    if load(symbol, interval).shape[0] > 0:
        df = load(symbol, interval)
        
    else:
        download(symbol, interval)
        df = load(symbol,interval)
    
    # sma and z-score windows
    if interval not in {'1d', '1wk', '1mo',}:
        n_sma = 21
        n_z = 7
    elif interval in {'1wk'}:
        n_sma = 29
        n_z = 9
    elif interval in {'1mo'}:
        n_sma = 17
        n_z = 7
    else:
        n_sma = 29
        n_z = 11
    
    # Kalman filtering (noise reduction algorithm) 
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=0.01
                     )

    state_means, _ = kf.filter(df['adj_close'].values)
    state_means = pd.Series(state_means.flatten(), index=df.index)
    df['kma'] = state_means
    df[f'sma{n_sma}'] = df['adj_close'].rolling(window=n_sma).mean().copy()
    df[f'kma_sma{n_sma}_diff'] = (df['kma'] - df[f'sma{n_sma}']).copy()
    df[f'kma_sma{n_sma}_diff_stdev{n_z}'] = df[f'kma_sma{n_sma}_diff'].rolling(window=n_z).std().copy()
    df[f'kma_sma{n_sma}_diff_mu{n_z}'] = df[f'kma_sma{n_sma}_diff'].rolling(window=n_z).mean().copy()

    # Calculate Kalman Filter vs SMA41 difference z-score
    df[f'kma_sma{n_sma}_diff_z{n_z}'] = df.apply(lambda row: zscore(row[f'kma_sma{n_sma}_diff'], row[f'kma_sma{n_sma}_diff_mu{n_z}'], row[f'kma_sma{n_sma}_diff_stdev{n_z}']), axis=1, result_type='expand').copy()

    # sma 20, 10, 5
    df['sma20'] = df['adj_close'].rolling(window=20).mean().copy()
    df['sma10'] = df['adj_close'].rolling(window=10).mean().copy()
    df['sma5'] = df['adj_close'].rolling(window=5).mean().copy()
    
    df['sma_long_diff'] = (df['sma20'] - df['sma10']).copy()
    df[f'sma_long_diff_mu{n_z}'] = df['sma_long_diff'].rolling(window=n_z).mean().copy()
    df[f'sma_long_diff_stdev{n_z}'] = df['sma_long_diff'].rolling(window=n_z).std().copy()  
    
    df[f'sma_long_diff_z{n_z}'] = df.apply(lambda row: zscore(row['sma_long_diff'], row[f'sma_long_diff_mu{n_z}'], row[f'sma_long_diff_stdev{n_z}']), axis=1, result_type='expand').copy()
    
    df['sma_short_diff'] = (df['sma10'] - df['sma5']).copy()
    df[f'sma_short_diff_mu{n_z}'] = df['sma_short_diff'].rolling(window=n_z).mean().copy()
    df[f'sma_short_diff_stdev{n_z}'] = df['sma_short_diff'].rolling(window=n_z).std().copy()   
    
    df[f'sma_short_diff_z{n_z}'] = df.apply(lambda row: zscore(row['sma_short_diff'], row[f'sma_short_diff_mu{n_z}'], row[f'sma_short_diff_stdev{n_z}']), axis=1, result_type='expand').copy()
    
    #update 1 day table: candle parts %'s
    df[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = df.apply(lambda row: candle_parts_pcts(row['open'], row['close'], row['high'],  row['low']), axis=1, result_type='expand').copy()

    #stdev of candel parts
    df[f'top_stdev{n_z}'] = df['pct_top_wick'].rolling(window=7).std().copy() 
    df[f'body_stdev{n_z}'] = df['pct_body'].rolling(window=7).std().copy() 
    df[f'bottom_stdev{n_z}'] = df['pct_bottom_wick'].rolling(window=7).std().copy()

    #mean of candel parts
    df[f'top_mu{n_z}'] = df['pct_top_wick'].rolling(window=7).mean().copy() 
    df[f'body_mu{n_z}'] = df['pct_body'].rolling(window=7).mean().copy() 
    df[f'bottom_mu{n_z}'] = df['pct_bottom_wick'].rolling(window=7).mean().copy()

    #z-score of candel parts
    df[f'top_z{n_z}'] = df.apply(lambda row: zscore(row['pct_top_wick'], row[f'top_mu{n_z}'], row[f'top_stdev{n_z}']), axis=1, result_type='expand').copy()
    df[f'body_z{n_z}'] = df.apply(lambda row: zscore(row['pct_body'], row[f'body_mu{n_z}'], row[f'body_stdev{n_z}']), axis=1, result_type='expand').copy()
    df[f'bottom_z{n_z}'] = df.apply(lambda row: zscore(row['pct_bottom_wick'], row[f'bottom_mu{n_z}'], row[f'bottom_stdev{n_z}']), axis=1, result_type='expand').copy()

    #stdev of volume
    df[f'vol_stdev{n_z}'] = df['volume'].rolling(window=n_z).std().copy() 
    
    #mean of volume
    df[f'vol_mu{n_z}'] = df['volume'].rolling(window=n_z).mean().copy() 

    #z-score of candel parts
    df[f'vol_z{n_z}'] = df.apply(lambda row: zscore(row['volume'], row[f'vol_mu{n_z}'], row[f'vol_stdev{n_z}']), axis=1, result_type='expand').copy()

    #update 1 day table: % gap btwn current open relative to previous candle size
    df['pc'] = df['close'].shift(1).copy()
    df['ph'] = df['high'].shift(1).copy()
    df['pl'] = df['low'].shift(1).copy()
    df['pct_gap_up_down'] = df.apply(lambda row: gap_up_down_pct(row['open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()
    
    df[f'pct_gap_up_down_mu{n_z}'] = df['pct_gap_up_down'].rolling(window=n_z).mean().copy()
    df[f'pct_gap_up_down_stdev{n_z}'] = df['pct_gap_up_down'].rolling(window=n_z).std().copy()
    df[f'pct_gap_up_down_z{n_z}'] = df.apply(lambda row: zscore(row['pct_gap_up_down'], row[f'pct_gap_up_down_mu{n_z}'], row[f'pct_gap_up_down_stdev{n_z}']), axis=1, result_type='expand').copy()

    #stdev of adjusted close
    df['ac_stdev5'] = df['adj_close'].rolling(window=5).std().copy() 
    df['ac_stdev7'] = df['adj_close'].rolling(window=7).std().copy() 
    df['ac_stdev11'] = df['adj_close'].rolling(window=11).std().copy() 

    #mean of adjusted close
    df['ac_mu5'] = df['adj_close'].rolling(window=5).mean().copy() 
    df['ac_mu7'] = df['adj_close'].rolling(window=7).mean().copy() 
    df['ac_mu11'] = df['adj_close'].rolling(window=11).mean().copy() 

    #z-score of adjusted close
    df['ac_z5'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu5'], row['ac_stdev5']), axis=1, result_type='expand').copy()
    df['ac_z7'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu7'], row['ac_stdev7']), axis=1, result_type='expand').copy()
    df['ac_z11'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu11'], row['ac_stdev11']), axis=1, result_type='expand').copy()
    
#     def next_prophet_prediction(window):
#         # Ensure the window has enough data points
#         if len(window) < n_sma:
#             return np.nan  # Return NaN if window length is insufficient

#         try:
#             # Prepare window DataFrame for Prophet and remove timezone
#             df_window = pd.DataFrame({
#                 'ds': window.index.tz_localize(None), # Remove timezone
#                 'y': window.values
#             }).dropna(subset=['y']).copy()

#             # Initialize and fit the model
#             model = Prophet(n_changepoints=5,   # Reduce from default (25)
#                             daily_seasonality=False, 
#                             weekly_seasonality=True, 
#                             seasonality_mode='additive',
#                            )  
#             model.fit(df_window)

#             # Create a DataFrame for the next day forecast
#             future = model.make_future_dataframe(periods=1)
#             forecast = model.predict(future)

#             # Return only the next day’s prediction
#             return forecast['yhat'].iloc[-1]
#         except Exception as e:
#             print(f"Error in Prophet prediction: {e}")
#             return np.nan  # Return NaN if there's an error in prediction
    
#     # Prophet next time period predictor
#     df[f'next_pred{n_sma}'] = df['adj_close'].rolling(window=n_sma).apply(next_prophet_prediction)
#     df[f'adj_next_pred{n_sma}_diff'] = (df['adj_close'] - df[f'next_pred{n_sma}']).copy()
#     df[f'adj_next_pred{n_sma}_diff_mu{n_z}'] = df[f'adj_next_pred{n_sma}_diff'].rolling(window=n_z).mean()
#     df[f'adj_next_pred{n_sma}_diff_stdev{n_z}'] = df[f'adj_next_pred{n_sma}_diff'].rolling(window=n_z).std()
    
#     # calculate z-score for adj_close prophet prediction difference
#     df[f'adj_next_pred{n_sma}_diff_z{n_z}'] = df.apply(lambda row: zscore(row[f'adj_next_pred{n_sma}_diff'], row[f'adj_next_pred{n_sma}_diff_mu{n_z}'], row[f'adj_next_pred{n_sma}_diff_stdev{n_z}']), axis=1, result_type='expand').copy()
    
    #target column: direction: -1, 0, 1
    df['adj_close_pctc'] = df['adj_close'].pct_change(fill_method=None)
    #     mean = df['adj_close_pctc'].mean()
    #     stdev = df['adj_close_pctc'].std()
    #     df['direction'] = df.apply(lambda row: direction(row['adj_close_pctc'], mean, stdev), axis=1, result_type='expand').copy() 
    df['direction'] = pd.qcut(df['adj_close_pctc'], q=3, labels=[2, 0, 1])
    df['direction'] = df['direction'].shift(-1).copy() # shift up to predict next time interval 

    # day of month, week, hour of day
    df['day_of_month'] = df.index.day        # Day of the month (1-31)
    df['day_of_week'] = df.index.weekday     # Day of the week (0 = Monday, 6 = Sunday)
    df['hour_of_day'] = df.index.hour        # Hour of the day (0-23)
  
    # categorical features
    categorical_features = ['day_of_month',
                            'day_of_week',
                            'hour_of_day']
    
    # Change data types of categorical columns to 'category'
    for column in categorical_features:
        df[column] = df[column].astype('category')
    
    # clustering... select columns ending with 'z##'
    z_columns = ['ac_z5', 'ac_z7', 'ac_z7', f'top_z{n_z}', f'body_z{n_z}', f'bottom_z{n_z}', f'vol_z{n_z}', f'pct_gap_up_down_z{n_z}', f'kma_sma{n_sma}_diff_z{n_z}', f'sma_short_diff_z{n_z}', f'sma_long_diff_z{n_z}', # f'adj_next_pred{n_sma}_diff_z{n_z}'
                ]
    
    # drop nulls for kmeans fit
    data_z = df[z_columns].dropna() 
    
    optimal_k = 2  # Replace with the optimal number from the elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data_z['cluster'] = kmeans.fit_predict(data_z)
    
    # Add the 'cluster' column back to the original DataFrame
    df['cluster'] = data_z['cluster']
    
    # save 1d file for model building
    df[[f'top_z{n_z}',
        f'body_z{n_z}',
        f'bottom_z{n_z}',
        f'vol_z{n_z}',
        f'pct_gap_up_down_z{n_z}',
        'ac_z5',
        'ac_z7',
        'ac_z11',
        f'kma_sma{n_sma}_diff_z{n_z}',
        f'sma_short_diff_z{n_z}',
        f'sma_long_diff_z{n_z}',
        # f'adj_next_pred{n_sma}_diff_stdev{n_z}',
        'day_of_month',
        'day_of_week',
        'hour_of_day',
        'cluster',
        'direction',
       ]
      ].to_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')

    
def model(symbol, interval):
    # Load data
    data = load_model_df(symbol, interval)
    data.dropna(inplace=True, axis=0)
    X = data.drop(columns=['direction'], axis=1)
    y = data['direction']
    
    # Print column names to check for issues
    # print("Columns in X before preprocessing:")
    # print(X.columns)
    
    # Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]

    # Check if categorical_features are present in X
    categorical_features = ['day_of_month', 'day_of_week', 'hour_of_day', 'cluster',]
    missing_features = [col for col in categorical_features if col not in X.columns]
    if missing_features:
        print(f"Missing categorical features: {missing_features}")

    # Make categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])
   
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # This will include all other columns in the transformed output
    )
    
    # Define your models
    models = {
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, learning_rate=0.06, max_depth=4, n_estimators=175),
        'GradientBoosting': GradientBoostingClassifier(random_state=42, learning_rate=0.11, max_depth=5, n_estimators=83, validation_fraction=0.09, n_iter_no_change=17,subsample=0.8,max_features='sqrt'),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=12, min_samples_split=9, n_estimators=200),
        # 'LightGBM': LGBMClassifier(random_state=42,force_col_wise=True),
        'KNN': KNeighborsClassifier(n_neighbors=8, p=1, weights='uniform')
    }

    
    # Create a pipeline that first preprocesses the data and then trains the model
    pipelines = {}
    for model_name, model in models.items():
        pipelines[model_name] = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', model)])
    
    #     X_validation_dfs = {}
    #     y_validation_series = {}
    models = {}
    classification_reports = {}
    
    
    # Create a function to get the column names after transformation
    def get_feature_names_out(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out())
            else:
                feature_names.extend(columns)
        return feature_names

    for model_name, pipeline in pipelines.items():
        # Apply the pipeline's preprocessor to the data
        X_transformed = pipeline.named_steps['preprocessor'].fit_transform(X)

        # Get feature names after transformation
        feature_names = get_feature_names_out(pipeline.named_steps['preprocessor'])

        # Convert the sparse matrix to a dense array and then to a DataFrame with proper column names
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        # Store current prediction data 
        curr_prediction = X_transformed.iloc[-1].copy()

        # Drop last row, model can't see this because it is used for prediction
        X_transformed = X_transformed.iloc[:-1]

        # Now perform train_test_split on the transformed data
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y[:-1], test_size=0.2, random_state=42, stratify=y[:-1])
        
        # Fit and evaluate the model
        model = pipeline.named_steps['classifier']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # store model in models dictionary
        models[model_name] = model

        # Evaluate the model
        # print(f"Model: {model.__class__.__name__}")
        # print(classification_report(y_test, y_pred, zero_division=0))
        
        classification_reports[model.__class__.__name__] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    return curr_prediction, models, feature_names, classification_reports


def make_prediction(models, curr_prediction, feature_names):
    # Ensure curr_prediction is a DataFrame with the correct feature names
    curr_prediction_df = pd.DataFrame([curr_prediction], columns=feature_names)
    
    predictions = {}
    prediction_probas = {}
    
    for model_name, model in models.items():
        # Make predictions using the reshaped and named DataFrame
        prediction = model.predict(curr_prediction_df)
        predictions[model_name] = prediction[0]
        prediction_probas[model_name] = model.predict_proba(curr_prediction_df)
    
    return predictions, prediction_probas


def predictions_summary(predictions, prediction_probas, classification_reports):
    model_map = {'XGBoost':'XGBClassifier', 
                 'GradientBoosting':'GradientBoostingClassifier', 
                 'RandomForest':'RandomForestClassifier', 
                 'KNN':'KNeighborsClassifier',
                }

    prediction_map = {0: 'static',
                      1: 'up',
                      2: 'down',
                     }
    
    prediction_str = []
    precision = []
    recall = []
    f1 = []
    support = []

    for model, prediction in predictions.items():
        prediction_str.append(prediction_map[prediction])
        precision.append(classification_reports[model_map[model]][str(prediction)]['precision'])
        recall.append(classification_reports[model_map[model]][str(prediction)]['recall'])
        f1.append(classification_reports[model_map[model]][str(prediction)]['f1-score'])
        support.append([classification_reports[model_map[model]]['1']['support'],
                        classification_reports[model_map[model]]['0']['support'],
                        classification_reports[model_map[model]]['2']['support'],
                       ]
                      )
        
    return pd.DataFrame({'model': predictions.keys(),
                         'prediction': prediction_str,
                         'kelly_1:2.5': kelly_c(precision), # kelly_c(np.max(np.array(list(prediction_probas.values())).T, axis=0))[0],
                         'prob_up': np.array(list(prediction_probas.values())).T[1][0],
                         'prob_static': np.array(list(prediction_probas.values())).T[0][0],
                         'prob_down': np.array(list(prediction_probas.values())).T[2][0],
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'support': support,
                        }
                       )

def dl_tf_pd(symbol, interval, skip_dl=False):
    # Define Eastern Time Zone
    eastern = pytz.timezone('US/Eastern')

    # Get current time in Eastern Time Zone
    eastern_time = datetime.now(eastern)

    # Format the time to include hour, minute, and seconds
    time_stamp = eastern_time.strftime('%Y-%m-%d %H:%M:%S')

    # print(f'{time_stamp}\n')
    
    if not skip_dl:
        download(symbol, interval)
        transform(symbol, interval)
        curr_prediction, models, feature_names, classification_reports = model(symbol, interval)
        predictions, prediction_probas = make_prediction(models, curr_prediction, feature_names)
        return time_stamp, predictions_summary(predictions, prediction_probas, classification_reports)
    else:
        transform(symbol, interval, period)
        curr_prediction, models, feature_names, classification_reports = model(symbol, interval)
        predictions, prediction_probas = make_prediction(models, curr_prediction, feature_names)
        return time_stamp, predictions_summary(predictions, prediction_probas, classification_reports)

def predictions(symbol):
    symbol = symbol.upper()
    intervals = ['5m',
                 '15m',
                 '1h',
                 '1d',
                 '1wk',
                 '1mo',
                ]

    for interval in intervals:

        time_stamp, summary_table = dl_tf_pd(symbol, interval, skip_dl=False) # skip_dl=True on redo's
        print(f'{symbol} {interval} Interval Timestamp: {time_stamp} ')
        col_headers = summary_table.model.tolist()
        col_headers
        summary_table_transposed = summary_table.T.iloc[1:]
        summary_table_transposed.columns = col_headers
        display(summary_table_transposed)

if __name__ == '__main__':
    ...