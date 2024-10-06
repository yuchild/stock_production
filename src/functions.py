#!usr/bin/env python3



import pandas as pd
import numpy as np
from yfinance import Ticker
from pykalman import KalmanFilter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

from datetime import datetime
import pytz

import pprint

def download(symbol, interval, period):
    stock = Ticker(symbol)
    stock_df = stock.history(interval=interval,
                             period=period,
                             auto_adjust=False,
                             prepost=True, # include aftermarket hours
                            )
    stock_df.columns = stock_df.columns.str.lower().str.replace(' ', '_')
    stock_df.to_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')
    
def load(symbol, interval):
    return pd.read_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')

def load_model_df(symbol, interval):
    return pd.read_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')

#########################################
# functions for use to transform tables #
#########################################

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

# direction calculation:
def direction(pctc, mean, stdev):
    
    pct_pos = mean + 0.43073 / 2.17 * stdev
    pct_neg = mean - 0.43073 / 2.17 * stdev
    if pctc >= pct_pos:
        return 1
    elif pctc <= pct_neg:
        return 2
    else:
        return 0

# compute kelly criterion
def kelly_c(p, l=1, g=2.5): 
    return p / l - (1 - p) / g

#################################
# functions for modeling output #
#################################

def transform(symbol, interval, period):
    
    if load(symbol, interval).shape[0] > 0:
        df = load(symbol, interval)
        
    else:
        download(symbol, interval, period)
        df = load(symbol,interval)
    
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
    df['sma40'] = df['adj_close'].rolling(window=40).mean().copy()
    df['kma_sma40_diff'] = (df['kma'] - df['sma40']).copy()
    df['kma_sma40_diff_stdev21'] = df['kma_sma40_diff'].rolling(window=21).std().copy()
    df['kma_sma40_diff_mu21'] = df['kma_sma40_diff'].rolling(window=21).mean().copy()

    # Calculate Kalman Filter vs SMA40 difference z-score
    df['kma_sma40_diff_z21'] = df.apply(lambda row: zscore(row['kma_sma40_diff'], row['kma_sma40_diff_mu21'], row['kma_sma40_diff_stdev21']), axis=1, result_type='expand').copy()

    #update 1 day table: candle parts %'s
    df[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = df.apply(lambda row: candle_parts_pcts(row['open'], row['close'], row['high'],  row['low']), axis=1, result_type='expand').copy()

    #stdev of adjusted close
    df['top_stdev21'] = df['pct_top_wick'].rolling(window=21).std().copy() 
    df['body_stdev21'] = df['pct_body'].rolling(window=21).std().copy() 
    df['bottom_stdev21'] = df['pct_bottom_wick'].rolling(window=21).std().copy()

    #mean of adjusted close
    df['top_mu21'] = df['pct_top_wick'].rolling(window=21).mean().copy() 
    df['body_mu21'] = df['pct_body'].rolling(window=21).mean().copy() 
    df['bottom_mu21'] = df['pct_bottom_wick'].rolling(window=21).mean().copy()

    #z-score of adjusted close
    df['top_z21'] = df.apply(lambda row: zscore(row['pct_top_wick'], row['top_mu21'], row['top_stdev21']), axis=1, result_type='expand').copy()
    df['body_z21'] = df.apply(lambda row: zscore(row['pct_body'], row['body_mu21'], row['body_stdev21']), axis=1, result_type='expand').copy()
    df['bottom_z21'] = df.apply(lambda row: zscore(row['pct_bottom_wick'], row['bottom_mu21'], row['bottom_stdev21']), axis=1, result_type='expand').copy()

    #update 1 day table: % gap btwn current open relative to previous candle size
    df['pc'] = df['close'].shift(1).copy()
    df['ph'] = df['high'].shift(1).copy()
    df['pl'] = df['low'].shift(1).copy()
    df['pct_gap_up_down'] = df.apply(lambda row: gap_up_down_pct(row['open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()

    #stdev of adjusted close
    df['ac_stdev5'] = df['adj_close'].rolling(window=5).std().copy() 
    df['ac_stdev8'] = df['adj_close'].rolling(window=8).std().copy() 
    df['ac_stdev13'] = df['adj_close'].rolling(window=13).std().copy()

    #mean of adjusted close
    df['ac_mu5'] = df['adj_close'].rolling(window=5).mean().copy() 
    df['ac_mu8'] = df['adj_close'].rolling(window=8).mean().copy() 
    df['ac_mu13'] = df['adj_close'].rolling(window=13).mean().copy()

    #z-score of adjusted close
    df['ac_z5'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu5'], row['ac_stdev5']), axis=1, result_type='expand').copy()
    df['ac_z8'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu8'], row['ac_stdev8']), axis=1, result_type='expand').copy()
    df['ac_z13'] = df.apply(lambda row: zscore(row['adj_close'], row['ac_mu13'], row['ac_stdev13']), axis=1, result_type='expand').copy()

    #target column: direction: -1, 0, 1
    df['adj_close_pctc'] = df['adj_close'].pct_change()
    mean = df['adj_close_pctc'].mean()
    stdev = df['adj_close_pctc'].std()
    df['direction'] = df.apply(lambda row: direction(row['adj_close_pctc'], mean, stdev), axis=1, result_type='expand').copy() 

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
    
    # save 1d file for model building
    df[['top_z21', 
        'body_z21', 
        'bottom_z21',
        'top_z21',
        'body_z21',
        'bottom_z21',
        'pct_gap_up_down',
        'ac_z5',
        'ac_z8',
        'ac_z13',
        'kma_sma40_diff_z21',
        'adj_close',
        'day_of_month',
        'day_of_week',
        'hour_of_day',
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
    categorical_features = ['day_of_month', 'day_of_week', 'hour_of_day']
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
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y[:-1], test_size=0.2, random_state=42)

        # cols
        cols = X_train.columns
                
        # store model in models dictionary
        models[model_name] = model
        
        
        # Fit and evaluate the model
        model = pipeline.named_steps['classifier']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


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
                         'kelly_1:2.5': kelly_c(np.max(np.array(list(prediction_probas.values())).T, axis=0))[0],
                         'prob_up': np.array(list(prediction_probas.values())).T[1][0],
                         'prob_static': np.array(list(prediction_probas.values())).T[0][0],
                         'prob_down': np.array(list(prediction_probas.values())).T[2][0],
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'support': support,
                        }
                       )

def dl_tf_pd(symbol, interval, period, skip_dl=False):
    # Define Eastern Time Zone
    eastern = pytz.timezone('US/Eastern')

    # Get current time in Eastern Time Zone
    eastern_time = datetime.now(eastern)

    # Format the time to include hour, minute, and seconds
    time_stamp = eastern_time.strftime('%Y-%m-%d %H:%M:%S')

    # print(f'{time_stamp}\n')
    
    if not skip_dl:
        download(symbol, interval, period)
        transform(symbol, interval, period)
        curr_prediction, models, feature_names, classification_reports = model(symbol, interval)
        predictions, prediction_probas = make_prediction(models, curr_prediction, feature_names)
        return time_stamp, predictions_summary(predictions, prediction_probas, classification_reports)
    else:
        transform(symbol, interval, period)
        curr_prediction, models, feature_names, classification_reports = model(symbol, interval)
        predictions, prediction_probas = make_prediction(models, curr_prediction, feature_names)
        return time_stamp, predictions_summary(predictions, prediction_probas, classification_reports)



if __name__ == '__main__':
    ...