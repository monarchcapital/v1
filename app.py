import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV # Import for tuning
from scipy.stats import uniform, randint # Import for parameter distributions

# Prophet requires a different data format and installation, will add in a later phase
# from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime # Import datetime for logging
from sklearn.preprocessing import StandardScaler # Import StandardScaler

import os # For file system operations

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

# Changed title to Monarch
st.title("👑 Monarch: Your Stock Price Oracle")

# Added an interesting description
st.markdown("""
Welcome to **Monarch**, a sophisticated stock price prediction platform.
This application employs a suite of advanced machine learning models and technical indicators
to analyze historical market data and forecast potential future price movements.
Leverage Monarch's analytical capabilities to gain deeper insights into stock trends,
evaluate model performance through backtesting, and explore future price projections.
""")

# --- Prediction Logging Functions ---
def save_prediction(ticker, prediction_for_date, predicted_value, actual_close_price, model_name, prediction_generation_date):
    """
    Saves a prediction to a CSV file.
    prediction_generation_date: The date/time when this prediction was generated.
    prediction_for_date: The date for which the prediction was made.
    actual_close_price: The actual close price for prediction_for_date (can be None/np.nan for future predictions).
    """
    predictions_dir = "monarch_predictions_data" # A dedicated directory for predictions
    os.makedirs(predictions_dir, exist_ok=True)
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions_log.csv")

    new_data = pd.DataFrame([{
        'prediction_generation_date': prediction_generation_date.strftime("%Y-%m-%d %H:%M:%S"),
        'prediction_for_date': prediction_for_date.strftime("%Y-%m-%d"),
        'ticker': ticker,
        'model_used': model_name,
        'predicted_value': predicted_value,
        'actual_close': actual_close_price if actual_close_price is not None else np.nan # Handle None for future actuals
    }])

    try:
        if not os.path.exists(file_path):
            new_data.to_csv(file_path, index=False)
        else:
            existing_df = pd.read_csv(file_path)
            existing_df['prediction_generation_date'] = pd.to_datetime(existing_df['prediction_generation_date'])
            existing_df['prediction_for_date'] = pd.to_datetime(existing_df['prediction_for_date'])

            # Check for exact duplicate (same generation date, same target date, same ticker, same model)
            # Compare dates only, not time, for generation date to allow for multiple runs on same day
            is_duplicate = ((existing_df['prediction_generation_date'].dt.date == prediction_generation_date.date()) &
                            (existing_df['prediction_for_date'].dt.date == prediction_for_date.date()) &
                            (existing_df['ticker'] == ticker) &
                            (existing_df['model_used'] == model_name)).any()

            if not is_duplicate:
                new_data.to_csv(file_path, mode='a', header=False, index=False)
                st.success(f"Prediction for {ticker} on {prediction_for_date.strftime('%Y-%m-%d')} saved.")
            else:
                st.info(f"Prediction for {ticker} on {prediction_for_date.strftime('%Y-%m-%d')} using {model_name} already logged for today's generation.")
    except Exception as e:
        st.error(f"Error saving prediction for {ticker} on {prediction_for_date.strftime('%Y-%m-%d')}: {e}")

def load_past_predictions(ticker):
    """Loads past predictions for a given ticker."""
    predictions_dir = "monarch_predictions_data"
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions_log.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df['prediction_for_date'] = pd.to_datetime(df['prediction_for_date'])
            df['prediction_generation_date'] = pd.to_datetime(df['prediction_generation_date'])
            return df
        except Exception as e:
            st.error(f"Error loading past predictions for {ticker}: {e}. File might be corrupted.")
            return pd.DataFrame()
    return pd.DataFrame()

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, use .ns at end of indan stocks ticker):", value="AAPL").upper()

# Training/Backtest Date Range
today = date.today()
default_end_bt = today - timedelta(days=1) # End yesterday
default_start_bt = default_end_bt - timedelta(days=365) # Start one year ago

start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt)
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt)

if start_bt >= end_bt:
    st.sidebar.error("Training Start Date (t1) must be before Training End Date (t2)")
    st.stop() # Stop execution if dates is invalid

model_choices = ['Random Forest', 'Linear Regression', 'SVR', 'XGBoost', 'Gradient Boosting', 'KNN', 'Decision Tree']
model_choice = st.sidebar.selectbox("Select Main Model (for Close Price):", model_choices)

# Add checkbox for hyperparameter tuning
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning (may increase training time)", value=False)


n_future = st.sidebar.slider("Predict Future Days (after t2):", min_value=1, max_value=60, value=15)

# Multi-select for model comparison
compare_models = st.sidebar.multiselect("Select Models to Compare:", model_choices, default=model_choices[:3])
# Changed max_value to 5000
train_days_comparison = st.sidebar.slider("Recent Data Period for Comparison (days):", min_value=60, max_value=5000, value=300, step=10)

# --- Technical Indicator Inputs and Explanations ---
st.sidebar.subheader("⚙️ Technical Indicator Settings")

st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**Moving Average (MA):** Smooths out price data to create a single flowing line, making it easier to spot trends. *Common values: 10, 20, 50, 200 days.*")
ma_input = st.sidebar.text_input("Moving Average Windows (comma-separated):", value="10,20")

# Parse MA input
try:
    ma_windows_list = [int(x.strip()) for x in ma_input.split(',') if x.strip()]
    if not ma_windows_list:
        ma_windows_list = [10, 20] # Default if input is empty
    if any(w <= 0 for w in ma_windows_list):
         st.sidebar.error("Moving Average windows must be positive integers.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid input for Moving Average windows. Please enter comma-separated integers.")
    st.stop()

st.sidebar.markdown("---") # Separator for clarity

# Added user input for Volatility windows
st.sidebar.markdown("**Volatility (Standard Deviation):** Measures how much the price is jumping around. Higher volatility means wilder price swings! *Common values: 10, 20 days.*")
std_input = st.sidebar.text_input("Volatility (Std Dev) Windows (comma-separated):", value="10")

# Parse Std Dev input
try:
    std_windows_list = [int(x.strip()) for x in std_input.split(',') if x.strip()]
    if not std_windows_list:
        std_windows_list = [10] # Default if input is empty
    if any(w <= 0 for w in std_windows_list):
         st.sidebar.error("Volatility windows must be positive integers.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid input for Volatility windows. Please enter comma-separated integers.")
    st.stop()


st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**Relative Strength Index (RSI):** A speed and momentum indicator. Helps spot if a stock is getting 'too expensive' (overbought >70) or 'too cheap' (oversold <30). *Common value: 14 days.*")
rsi_window = st.sidebar.number_input("RSI Window:", min_value=1, value=14)

st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It's a bit like spotting changes in the stock's speed and direction. *Common values: 12, 26, 9 days.*")
macd_short_window = st.sidebar.number_input("MACD Short Window:", min_value=1, value=12)
macd_long_window = st.sidebar.number_input("MACD Long Window:", min_value=1, value=26)
macd_signal_window = st.sidebar.number_input("MACD Signal Window:", min_value=1, value=9)

st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**Bollinger Bands (BB):** These bands hug the price and expand/contract with volatility. They help identify potential price extremes and momentum shifts. *Common values: 20 day window, 2.0 std dev.*")
bb_window = st.sidebar.number_input("Bollinger Bands Window:", min_value=1, value=20)
bb_std_dev = st.sidebar.number_input("Bollinger Bands Std Dev Multiplier:", min_value=0.1, value=2.0)

st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**Average True Range (ATR):** Measures market volatility by calculating the average range of price movement over a period. Helps gauge how much the price *could* move. *Common value: 14 days.*")
atr_window = st.sidebar.number_input("ATR Window:", min_value=1, value=14)

st.sidebar.markdown("---") # Separator for clarity

st.sidebar.markdown("**Stochastic Oscillator:** Compares a stock's closing price to its price range over a given period. Useful for spotting overbought (>80) or oversold (<20) conditions and potential trend reversals. *Common values: 14 day %K, 3 day %D.*")
stoch_window = st.sidebar.number_input("Stochastic %K Window:", min_value=1, value=14)
stoch_smooth_window = st.sidebar.number_input("Stochastic %D (Smooth) Window:", min_value=1, value=3)

st.sidebar.markdown("---") # Separator for clarity


lag_features_list = [1, 5, 10] # Keep lag features fixed for now

# Initialize variables for model comparison results
best_model_name = "N/A"
best_model_rmse = float('inf')
best_model_pct_rmse = float('inf')
df_comparison = pd.DataFrame()

# --- Data Loading ---
# Added ttl to the cache to force refresh after 1 hour
@st.cache_data(show_spinner=True, ttl=timedelta(hours=1))
def download_data(ticker):
    """Downloads historical stock data using yfinance."""
    try:
        # Fetch data with a specific period to ensure recent data is attempted
        # Using period="max" and then filtering might be better to get all history
        # but let's ensure we get data up to today.
        data = yf.download(ticker, period="max", progress=False) # Added progress=False to reduce output
        if data.empty:
            st.error(f"Could not download data for ticker {ticker}. Please check the ticker symbol.")
            return pd.DataFrame()

        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)

        # Clean column names: handle tuple format, convert to string, lowercase, and strip whitespace
        # This is crucial for handling potential MultiIndex columns from yfinance
        cleaned_columns = []
        for col in data.columns:
            if isinstance(col, tuple):
                # If it's a tuple, take the first element (the actual column name)
                cleaned_col = str(col[0]).lower().strip()
            else:
                # Otherwise, just process the string name
                cleaned_col = str(col).lower().strip()
            cleaned_columns.append(cleaned_col)

        data.columns = cleaned_columns


        # Define the required columns in lowercase for checking
        required_cols_lower = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Check if all required columns are present after cleaning
        if not all(col in data.columns for col in required_cols_lower):
             missing = [col for col in required_cols_lower if col not in data.columns]
             st.error(f"Downloaded data for {ticker} is missing required columns: {missing}. Cannot proceed.")
             return pd.DataFrame()

        # Rename columns to the standard format
        data.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)


        # Select and return the required columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data.dropna(inplace=True)
        # Ensure Date is datetime type
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()


# Updated create_features to accept std_windows_list and keep Open for prediction
def create_features(df, lag_features, ma_windows, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window):
    """
    Creates time-series features for stock price prediction.
    Includes lagged closing prices, moving averages, volatility, OBV, RSI, MACD, Bollinger Bands, ATR, and Stochastic Oscillator.
    Accepts lists of windows for MAs and Std Dev, and single windows for other indicators.
    Keeps 'Open' and 'Volatility' for potential prediction targets.
    """
    df_features = df.copy()
    # Ensure column names are strings before creating features
    df_features.columns = df_features.columns.astype(str)

    df_features['Day'] = np.arange(len(df_features)) # Keep Day as a potential feature

    # Add Lagged Features (using Close for now, could add Open lags later if needed)
    for lag in lag_features:
        df_features[f'Close_Lag_{lag}'] = df_features['Close'].shift(lag)
        # Add Open Lags as well for Open price prediction
        df_features[f'Open_Lag_{lag}'] = df_features['Open'].shift(lag)


    # Add Moving Averages (using the provided list)
    for window in ma_windows:
        df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()

    # Add Volatility (Standard Deviation) using the provided list
    for window in std_windows_list:
         df_features[f'Volatility_{window}'] = df_features['Close'].rolling(window=window).std()

    # Add On-Balance Volume (OBV)
    # Calculate daily price change
    df_features['Price_Change'] = df_features['Close'].diff()

    # Assign volume based on price change direction using .loc for robustness
    df_features['Volume_Direction'] = 0 # Initialize with 0
    df_features.loc[df_features['Price_Change'] > 0, 'Volume_Direction'] = df_features['Volume']
    df_features.loc[df_features['Price_Change'] < 0, 'Volume_Direction'] = -df_features['Volume']


    # Calculate the cumulative sum of Volume_Direction to get OBV
    df_features['OBV'] = df_features['Volume_Direction'].cumsum()

    # Calculate Relative Strength Index (RSI)
    delta = df_features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=rsi_window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_window-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_features['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Moving Average Convergence Divergence (MACD)
    exp1 = df_features['Close'].ewm(span=macd_short_window, adjust=False).mean()
    exp2 = df_features['Close'].ewm(span=macd_long_window, adjust=False).mean()
    df_features['MACD'] = exp1 - exp2
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=macd_signal_window, adjust=False).mean()
    df_features['MACD_Hist'] = df_features['MACD'] - df_features['MACD_Signal']

    # Calculate Bollinger Bands using explicit steps and ensuring 1D array for Series creation
    df_features['BB_Middle'] = df_features['Close'].rolling(window=bb_window).mean()
    rolling_std = df_features['Close'].rolling(window=bb_window).std()

    # Get the values from the rolling_std Series and multiply by bb_std_dev
    std_multiplier_values = rolling_std.values * bb_std_dev

    # Ensure the result is a Series before adding/subtracting
    # Use .ravel() just in case, although it should be 1D at this point
    std_multiplier_series = pd.Series(std_multiplier_values.ravel(), index=df_features.index)

    df_features['BB_Upper'] = df_features['BB_Middle'] + std_multiplier_series
    df_features['BB_Lower'] = df_features['BB_Middle'] - std_multiplier_series


    # Add Average True Range (ATR)
    # Calculate True Range
    df_features['High_Low'] = df_features['High'] - df_features['Low']
    df_features['High_PrevClose'] = np.abs(df_features['High'] - df_features['Close'].shift(1))
    df_features['Low_PrevClose'] = np.abs(df_features['Low'] - df_features['Close'].shift(1))
    df_features['True_Range'] = df_features[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df_features['ATR'] = df_features['True_Range'].ewm(span=atr_window, adjust=False).mean()


    # Add Stochastic Oscillator
    # Calculate %K
    lowest_low = df_features['Low'].rolling(window=stoch_window).min()
    highest_high = df_features['High'].rolling(window=stoch_window).max()
    # Avoid division by zero if highest_high equals lowest_low
    df_features['%K'] = ((df_features['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)) * 100
    # Calculate %D (3-day SMA of %K)
    df_features['%D'] = df_features['%K'].rolling(window=stoch_smooth_window).mean()

    # Drop the temporary columns used for calculations if they are not needed as features
    df_features.drop(columns=['Price_Change', 'Volume_Direction', 'High_Low', 'High_PrevClose', 'Low_PrevClose', 'True_Range'], errors='ignore', inplace=True)
    # Drop the temporary Lowest_Low and Highest_High columns if they were created
    df_features.drop(columns=['Lowest_Low', 'Highest_High'], errors='ignore', inplace=True)


    # Drop rows with NaN values created by lagging and rolling windows
    # Keep original data columns for iterative feature calculation
    # We need to keep 'Open', 'High', 'Low', 'Close', 'Volume' for creating features iteratively
    cols_to_keep_for_iterative = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols_only = [col for col in df_features.columns if col not in cols_to_keep_for_iterative]
    # Only dropna on the feature columns, keep rows if original data is present
    df_features_cleaned = df_features.dropna(subset=feature_cols_only).copy()


    return df_features_cleaned


def get_model(name):
    """Returns the selected regression model."""
    if name == 'Random Forest':
        # Added default hyperparameters for better starting point
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == 'Linear Regression':
        return LinearRegression()
    elif name == 'SVR':
        # Added default hyperparameters for better starting point
        return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    elif name == 'XGBoost':
        # Added default hyperparameters and early stopping parameters
        # Added tree_method='hist' for potentially better performance and compatibility
        # Added verbosity=0 to suppress training output
        return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42, early_stopping_rounds=10, eval_metric='rmse', use_label_encoder=False, tree_method='hist', verbosity=0)
    elif name == 'Gradient Boosting':
        # Added default hyperparameters for better starting point
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif name == 'KNN':
        # Added default hyperparameters for better starting point
        return KNeighborsRegressor(n_neighbors=5) # Number of neighbors can be tuned
    elif name == 'Decision Tree':
        # Added default hyperparameters to limit depth
        return DecisionTreeRegressor(random_state=42, max_depth=10) # Limit max_depth to prevent overfitting
    # Add Prophet later
    # elif model_name == 'Prophet':
    #     return Prophet()
    else:
        return LinearRegression()

# Modified train_model to train for multiple targets
def train_models(df_train, model_choice, perform_tuning=False, messages_list=None):
    """
    Trains models for 'Close', 'Open', and 'Volatility' on the provided training data,
    optionally with hyperparameter tuning.
    Returns a dictionary of trained models, scalers, and feature columns for each target.
    """
    if messages_list is None:
        messages_list = [] # Initialize if not provided

    if df_train.empty:
        messages_list.append("Training data is empty. Cannot train models.")
        return None

    # Define features (X) and potential targets (y)
    # Exclude original price/volume columns from features, except for those we might predict ('Open', 'Volatility')
    # We need to be careful here: when predicting 'Close', 'Open' and 'Volatility' can be features.
    # When predicting 'Open', 'Close' and 'Volatility' can be features.
    # When predicting 'Volatility', 'Close' and 'Open' can be features.
    # Let's define a base set of features excluding the targets we are currently predicting.

    # Base features exclude Date
    base_feature_cols = [col for col in df_train.columns if col not in ['Date']]

    # Remove target columns from features for each respective model training
    feature_cols_close = [col for col in base_feature_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] + [f'Volatility_{w}' for w in std_windows_list]]
    feature_cols_open = [col for col in base_feature_cols if col not in ['Open', 'Close', 'High', 'Low', 'Volume'] + [f'Volatility_{w}' for w in std_windows_list]]
    # For volatility prediction, we can use price-based features, but exclude Volatility itself as a direct feature
    feature_cols_volatility = [col for col in base_feature_cols if col not in [f'Volatility_{w}' for w in std_windows_list] + ['Close', 'Open', 'High', 'Low', 'Volume']]


    # Ensure feature lists are not empty
    if not feature_cols_close or not feature_cols_open or not feature_cols_volatility:
         messages_list.append("Feature columns are empty after excluding targets. Cannot train models.")
         messages_list.append(f"Debug: feature_cols_close {feature_cols_close}")
         messages_list.append(f"Debug: feature_cols_open {feature_cols_open}")
         messages_list.append(f"Debug: feature_cols_volatility {feature_cols_volatility}")
         return None


    # Prepare data for each target
    X_close = df_train[feature_cols_close].copy()
    y_close = df_train['Close'].values.ravel()

    X_open = df_train[feature_cols_open].copy()
    y_open = df_train['Open'].values.ravel()

    # We will predict the first volatility feature in the list as the target volatility
    volatility_target_col = f'Volatility_{std_windows_list[0]}' if std_windows_list else None
    if not volatility_target_col or volatility_target_col not in df_train.columns:
         messages_list.append("Volatility target column not found. Cannot train volatility model.")
         X_volatility = pd.DataFrame() # Empty dataframe to skip training
         y_volatility = np.array([])
         feature_cols_volatility_train = []
    else:
        X_volatility = df_train[feature_cols_volatility].copy()
        y_volatility = df_train[volatility_target_col].values.ravel()
        feature_cols_volatility_train = feature_cols_volatility # Use this for returning

    # Check if there are enough samples for training for each target
    if len(X_close) == 0 or len(y_close) == 0 or len(X_close) != len(y_close):
         messages_list.append(f"Insufficient/mismatched data for Close price training ({len(X_close)} features, {len(y_close)} target values). Cannot train Close model.")
         model_close, scaler_close, feature_cols_close_train = None, None, []
    else:
        # Scale the features for Close
        scaler_close = StandardScaler()
        X_close_scaled = scaler_close.fit_transform(X_close)
        model_close, best_params_close = _train_single_model(X_close_scaled, y_close, model_choice, perform_tuning, f"{model_choice} (Close)", messages_list)
        feature_cols_close_train = feature_cols_close # Use this for returning


    if len(X_open) == 0 or len(y_open) == 0 or len(X_open) != len(y_open):
         messages_list.append(f"Insufficient/mismatched data for Open price training ({len(X_open)} features, {len(y_open)} target values). Cannot train Open model.")
         model_open, scaler_open, feature_cols_open_train = None, None, []
    else:
        # Scale the features for Open
        scaler_open = StandardScaler()
        X_open_scaled = scaler_open.fit_transform(X_open)
        model_open, best_params_open = _train_single_model(X_open_scaled, y_open, model_choice, perform_tuning, f"{model_choice} (Open)", messages_list)
        feature_cols_open_train = feature_cols_open # Use this for returning


    if len(X_volatility) == 0 or len(y_volatility) == 0 or len(X_volatility) != len(y_volatility):
         messages_list.append(f"Insufficient/mismatched data for Volatility training ({len(X_volatility)} features, {len(y_volatility)} target values). Cannot train Volatility model.")
         model_volatility, scaler_volatility, feature_cols_volatility_train = None, None, []
    else:
        # Scale the features for Volatility
        scaler_volatility = StandardScaler()
        X_volatility_scaled = scaler_volatility.fit_transform(X_volatility)
        model_volatility, best_params_volatility = _train_single_model(X_volatility_scaled, y_volatility, model_choice, perform_tuning, f"{model_choice} (Volatility)", messages_list)
        # feature_cols_volatility_train is already set above


    # Return a dictionary containing models, scalers, and feature lists for each target
    return {
        'Close': {'model': model_close, 'scaler': scaler_close, 'features': feature_cols_close_train},
        'Open': {'model': model_open, 'scaler': scaler_open, 'features': feature_cols_open_train},
        'Volatility': {'model': model_volatility, 'scaler': scaler_volatility, 'features': feature_cols_volatility_train, 'target_col': volatility_target_col} # Include target col for volatility
    }


# Helper function to train a single model for a specific target
def _train_single_model(X_scaled, y, model_name, perform_tuning, display_name, messages_list):
    """Trains a single model for a given target."""
    base_model = get_model(model_name)
    trained_model = None
    best_params = None

    # Messages are now written to the parent expander context
    if perform_tuning:
        messages_list.append(f"Performing Randomized Hyperparameter Tuning for {display_name}...")
        param_distributions = {
            'Random Forest': {
                'n_estimators': randint(50, 200),
                'max_depth': [None] + list(randint(5, 20).rvs(10)), # Include None for max_depth
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10)
            },
            'SVR': {
                'C': uniform(1, 1000),
                'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(10)), # Include scale and auto
                'epsilon': uniform(0.01, 1)
            },
            'XGBoost': {
                'n_estimators': randint(100, 1000),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 1.0),
                'colsample_bytree': uniform(0.6, 1.0)
            },
            'Gradient Boosting': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10)
            },
            'KNN': {
                'n_neighbors': randint(3, 20)
            },
             'Decision Tree': {
                'max_depth': [None] + list(randint(3, 15).rvs(5)), # Include None for max_depth
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10)
            }
            # Linear Regression does not typically have hyperparameters to tune in this context
        }

        if model_name in param_distributions:
            scoring_metric = 'neg_mean_squared_error'
            n_iter_search = 20
            cv_folds = 5

            # Check if there are enough samples for cross-validation
            if len(X_scaled) < cv_folds + 1:
                 messages_list.append(f"Not enough data for {cv_folds}-fold cross-validation tuning for {display_name} ({len(X_scaled)} samples). Skipping tuning.")
                 perform_tuning = False # Disable tuning for this model

            if perform_tuning: # Re-check if tuning is still enabled
                random_search = RandomizedSearchCV(base_model, param_distributions=param_distributions[model_name],
                                                   n_iter=n_iter_search, scoring=scoring_metric, cv=cv_folds,
                                                   random_state=42, n_jobs=-1, verbose=0)

                try:
                    random_search.fit(X_scaled, y)
                    trained_model = random_search.best_estimator_
                    best_params = random_search.best_params_
                    messages_list.append(f"✅ Best parameters found for {display_name}: {best_params}")
                except Exception as e:
                    messages_list.append(f"Error during hyperparameter tuning for {display_name}: {e}. Training with default parameters.")
                    # Fallback to training with default parameters
                    try:
                        base_model.fit(X_scaled, y)
                        trained_model = base_model
                    except Exception as e_fallback:
                        messages_list.append(f"Error during fallback training for {display_name}: {e_fallback}. Cannot train model.")
                        trained_model = None


            else: # Tuning was skipped due to insufficient data
                 messages_list.append(f"Training {display_name} with default parameters (tuning skipped).")
                 try:
                    # For XGBoost, train with early stopping if tuning is off
                    if model_name == 'XGBoost':
                         train_size = int(len(X_scaled) * 0.8)
                         if len(X_scaled) - train_size > 0:
                             X_train_split, X_val_split = X_scaled[:train_size], X_scaled[train_size:]
                             y_train_split, y_val_split = y[:train_size], y[train_size:]
                             base_model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
                             trained_model = base_model
                         else:
                              messages_list.append(f"Not enough data for validation set for {display_name} early stopping. Training on full data.")
                              base_model.fit(X_scaled, y)
                              trained_model = base_model
                    else:
                       base_model.fit(X_scaled, y)
                       trained_model = base_model
                 except Exception as e:
                     messages_list.append(f"Error during model training with default parameters for {display_name}: {e}. Cannot train model.")
                     trained_model = None


        else: # Model not in param_distributions
            messages_list.append(f"Hyperparameter tuning not configured for {display_name}. Training with default parameters.")
            try:
                # For XGBoost, train with early stopping if tuning is off
                if model_name == 'XGBoost':
                     train_size = int(len(X_scaled) * 0.8)
                     if len(X_scaled) - train_size > 0:
                        X_train_split, X_val_split = X_scaled[:train_size], X_scaled[train_size:]
                        y_train_split, y_val_split = y[:train_size], y[train_size:]
                        base_model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
                        trained_model = base_model
                     else:
                          messages_list.append(f"Not enough data for validation set for {display_name} early stopping. Training on full data.")
                          base_model.fit(X_scaled, y)
                          trained_model = base_model
                else:
                    base_model.fit(X_scaled, y)
                    trained_model = base_model
            except Exception as e:
                messages_list.append(f"Error during model training with default parameters for {display_name}: {e}. Cannot train model.")
                trained_model = None

    else: # perform_tuning is False, train with default parameters
        messages_list.append(f"Training {display_name} with default parameters.")
        try:
            # For XGBoost, train with early stopping if tuning is off
            if model_name == 'XGBoost':
                 train_size = int(len(X_scaled) * 0.8)
                 if len(X_scaled) - train_size > 0:
                    X_train_split, X_val_split = X_scaled[:train_size], X_scaled[train_size:]
                    y_train_split, y_val_split = y[:train_size], y[train_size:]
                    base_model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
                    trained_model = base_model
                 else:
                      messages_list.append(f"Not enough data for validation set for {display_name} early stopping. Training on full data.")
                      base_model.fit(X_scaled, y)
                      trained_model = base_model

            else:
                base_model.fit(X_scaled, y)
                trained_model = base_model
        except Exception as e:
            messages_list.append(f"Error during model training with default parameters for {display_name}: {e}. Cannot train model.")
            trained_model = None

    return trained_model, best_params


# Modified generate_predictions to handle multiple targets
def generate_predictions(df_data, models_scalers_features):
    """
    Generates predictions for 'Close', 'Open', and 'Volatility' using the trained models and scalers.
    Returns a dictionary of prediction DataFrames for each target.
    """
    if df_data.empty or models_scalers_features is None:
        return {}

    predictions = {}

    for target, info in models_scalers_features.items():
        model = info.get('model')
        scaler = info.get('scaler')
        feature_cols_train = info.get('features')
        target_col = info.get('target_col') # For Volatility target column name

        if model is None or scaler is None or feature_cols_train is None:
            st.warning(f"Skipping prediction for {target} due to missing model, scaler, or feature list.")
            predictions[target] = pd.DataFrame()
            continue

        # Select and reorder columns to match the training data features
        # Ensure columns are strings before reindexing
        df_data.columns = df_data.columns.astype(str)
        # Use reindex to handle potential missing columns in df_data compared to training features
        X = df_data.reindex(columns=feature_cols_train, fill_value=0).copy()

        # Check if X is empty after selecting columns
        if X.empty:
            st.warning(f"No features available for {target} prediction after selecting training columns. Cannot generate predictions.")
            predictions[target] = pd.DataFrame()
            continue

        # Ensure X has the same number of columns as the training data after scaling
        expected_num_features = len(feature_cols_train)
        if hasattr(scaler, 'n_features_in_') and X.shape[1] != scaler.n_features_in_:
            st.warning(f"Mismatch in number of features for {target} prediction ({X.shape[1]}) and training scaler ({scaler.n_features_in_}). Cannot generate predictions.")
            predictions[target] = pd.DataFrame()
            continue
        elif X.shape[1] != expected_num_features:
             st.warning(f"Mismatch in number of features for {target} prediction ({X.shape[1]}) and expected training features ({expected_num_features}). Cannot generate predictions.")
             predictions[target] = pd.DataFrame()
             continue


        X_scaled = scaler.transform(X)

        try:
            predicted_values = model.predict(X_scaled)
        except Exception as e:
            st.warning(f"Error during {target} prediction: {e}")
            predictions[target] = pd.DataFrame()
            continue

        result_df = pd.DataFrame({
            'Date': df_data['Date'].values,
            f'Predicted {target}': predicted_values.ravel()
        })

        # Add actual values if available in df_data (only for 'Close', 'Open', 'Volatility')
        if target == 'Close' and 'Close' in df_data.columns:
             result_df['Actual Close'] = df_data['Close'].values.ravel()
        elif target == 'Open' and 'Open' in df_data.columns:
             result_df['Actual Open'] = df_data['Open'].values.ravel()
        elif target == 'Volatility' and target_col and target_col in df_data.columns:
             result_df[f'Actual {target}'] = df_data[target_col].values.ravel()


        # Calculate difference only if actual is available
        actual_col_name = f'Actual {target}'
        predicted_col_name = f'Predicted {target}'
        if actual_col_name in result_df.columns and not result_df[actual_col_name].isnull().all():
             result_df['Difference'] = result_df[actual_col_name] - result_df[predicted_col_name]
        else:
             result_df['Difference'] = np.nan # No difference if no actual price

        # Ensure columns are numeric before calculating metrics
        for col in [actual_col_name, predicted_col_name, 'Difference']:
             if col in result_df.columns:
                  result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        # Drop rows where Actual or Predicted are NaN (for metrics and display accuracy)
        # Only drop NaNs if there's an actual column to compare against
        if actual_col_name in result_df.columns:
             result_df.dropna(subset=[actual_col_name, predicted_col_name], inplace=True)


        predictions[target] = result_df

    return predictions

# Load data
data_load_state = st.text(f"Loading data for {ticker}...")
data = download_data(ticker)
data_load_state.text(f"Loaded data for {ticker}, total {len(data)} rows.")

if data.empty:
    st.stop()

# Determine the maximum window size for feature calculation
# Include std_windows_list in the calculation
all_windows = lag_features_list + ma_windows_list + std_windows_list + [rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window]
# Filter out None values and ensure windows are positive integers
valid_windows = [w for w in all_windows if isinstance(w, int) and w > 0]
max_window = max(valid_windows) if valid_windows else 1 # Ensure at least 1 if no valid windows defined


# Create features for the entire dataset, keeping original columns for iterative forecasting
df_features_full = create_features(data.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)

st.write(f"Debug: Number of rows after feature creation: {len(df_features_full)}")

# Filter data for the training period based on sidebar inputs (start_bt to end_bt)
df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()

if df_train_period.empty:
    st.warning("No data available in the selected training/backtest date range after feature creation.")
    st.stop()

# List to collect all training messages
all_training_messages = []

# Train models for Close, Open, and Volatility on the selected training period, with optional tuning
# We train the main model here first
models_scalers_features_main = train_models(df_train_period.copy(), model_choice, perform_tuning=perform_tuning, messages_list=all_training_messages)

if models_scalers_features_main is None or not models_scalers_features_main.get('Close', {}).get('model'):
    st.warning("Main Close price model training failed. Please check data and parameters.")
    st.stop()

# --- Output 5: Model Comparison (Moved up to calculate best model before next day prediction) ---
st.subheader("📊 Selected Models Comparison on Recent Data")

# Ensure enough data for comparison after feature creation
# Use the train_days_comparison slider for this section
compare_days = min(train_days_comparison, len(df_features_full)) # Use up to train_days_comparison or available data
df_compare_full = df_features_full.tail(compare_days).copy() # Use .copy() to avoid SettingWithCopyWarning

comparison_results = []
best_model_name = "N/A"
best_model_rmse = float('inf')
best_model_pct_rmse = float('inf') # Initialize best percentage RMSE

if df_compare_full.empty:
     st.warning("Not enough data after feature creation for model comparison.")
else:
    # Calculate the average actual close price for the comparison period
    average_actual_compare = df_compare_full['Close'].mean()

    fig_compare = go.Figure()
    # Actual line in black
    fig_compare.add_trace(go.Scatter(x=df_compare_full['Date'], y=df_compare_full['Close'], mode='lines', name='Actual', line=dict(color='black', width=3)))

    # Define a list of colors for predicted lines
    predicted_colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']


    for i, model_name_compare in enumerate(compare_models): # Renamed variable to avoid conflict
        try:
            # Train each comparison model on the 'compare_days' data for the 'Close' target
            # We will tune comparison models only if main tuning is enabled, for consistency and because tuning is slow
            # We only need the Close model for this comparison section
            # Train only the Close model for comparison purposes
            compare_model_info = train_models(df_compare_full.copy(), model_name_compare, perform_tuning=perform_tuning, messages_list=all_training_messages) # Pass messages_list

            if compare_model_info and compare_model_info.get('Close', {}).get('model'):
                model_compare = compare_model_info['Close']['model']
                scaler_compare = compare_model_info['Close']['scaler']
                feature_cols_compare_train = compare_model_info['Close']['features']

                # Generate predictions for the comparison period (only Close needed here)
                compared_pred_dfs = generate_predictions(df_compare_full.copy(), {'Close': {'model': model_compare, 'scaler': scaler_compare, 'features': feature_cols_compare_train}}) # Pass only Close info

                if 'Close' in compared_pred_dfs and not compared_pred_dfs['Close'].empty:
                    compared_pred_df = compared_pred_dfs['Close']
                    # Ensure actual and predicted are numeric for metrics
                    compared_pred_df['Actual Close'] = pd.to_numeric(compared_pred_df['Actual Close'], errors='coerce')
                    compared_pred_df['Predicted Close'] = pd.to_numeric(compared_pred_df['Predicted Close'], errors='coerce')
                    compared_pred_df.dropna(subset=['Actual Close', 'Predicted Close'], inplace=True)

                    if not compared_pred_df.empty: # Check again after dropping NaNs
                        y_compare_valid = compared_pred_df['Actual Close'].values.ravel()
                        y_pred = compared_pred_df['Predicted Close'].values.ravel()

                        mae = mean_absolute_error(y_compare_valid, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_compare_valid, y_pred))

                        # Calculate percentage error metrics
                        pct_mae = (mae / average_actual_compare) * 100 if average_actual_compare > 0 else np.nan
                        pct_rmse = (rmse / average_actual_compare) * 100 if average_actual_compare > 0 else np.nan


                        comparison_results.append({
                            'Model': model_name_compare,
                            'MAE': mae,
                            'RMSE': rmse,
                            '%-MAE': pct_mae, # Added percentage MAE
                            '%-RMSE': pct_rmse # Added percentage RMSE
                        })

                        # Check for the best model based on RMSE (can also consider %-RMSE)
                        if rmse < best_model_rmse:
                            best_model_rmse = rmse
                            best_model_name = model_name_compare
                        if pct_rmse < best_model_pct_rmse:
                             best_model_pct_rmse = pct_rmse # Update best percentage RMSE


                        # Add predicted line to the comparison chart with a distinct color
                        color = predicted_colors[i % len(predicted_colors)] # Cycle through colors
                        fig_compare.add_trace(go.Scatter(x=compared_pred_df['Date'], y=y_pred, mode='lines', name=f"{model_name_compare} Predicted", line=dict(color=color)))
                    else:
                        st.warning(f"Not enough valid data for metrics calculation for model {model_name_compare} in comparison after dropping NaNs.")
                        comparison_results.append({
                            'Model': model_name_compare,
                            'MAE': None,
                            'RMSE': None,
                            '%-MAE': None, # Added None for percentage metrics
                            '%-RMSE': None # Added None for percentage metrics
                        })
                else:
                    st.warning(f"Could not generate comparison predictions for model {model_name_compare}.")
                    comparison_results.append({
                        'Model': model_name_compare,
                        'MAE': None,
                        'RMSE': None,
                        '%-MAE': None, # Added None for percentage metrics
                        '%-RMSE': None # Added None for percentage metrics
                    })
            else:
                 st.warning(f"Could not train model {model_name_compare} for comparison.")
                 comparison_results.append({
                    'Model': model_name_compare,
                    'MAE': None,
                    'RMSE': None,
                    '%-MAE': None, # Added None for percentage metrics
                    '%-RMSE': None # Added None for percentage metrics
                })


        except Exception as e:
             st.warning(f"Could not run comparison for model {model_name_compare}: {e}")
             comparison_results.append({
                'Model': model_name_compare,
                'MAE': None,
                'RMSE': None,
                '%-MAE': None, # Added None for percentage metrics
                '%-RMSE': None # Added None for percentage metrics
            })


    fig_compare.update_layout(
        title="Model Comparison: Actual vs Predicted on Recent Data",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            range=[df_compare_full['Date'].min(), df_compare_full['Date'].max()] # Explicitly set x-axis range
        )
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Sort by RMSE, then Percentage RMSE, and drop rows with missing RMSE
    df_comparison = pd.DataFrame(comparison_results).sort_values(['RMSE', '%-RMSE']).dropna(subset=['RMSE'])
    # Format the output table to include percentage metrics
    st.dataframe(df_comparison.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "%-MAE": "{:.2f}%", "%-RMSE": "{:.2f}%"}))


# --- Output 1: Predicted price for t2 + 1 (excluding weekends/holidays) ---
# Get the features for the last day of the training period (up to end_bt)
# Use df_train_period which already has features calculated.
last_day_features_df = df_train_period.tail(1).copy()

if not last_day_features_df.empty:
    # Determine the date for the next trading day
    last_train_date = pd.to_datetime(end_bt)
    next_calendar_day = last_train_date + timedelta(days=1)
    next_trading_day = next_calendar_day
    while next_trading_day.weekday() >= 5: # Monday is 0, Sunday is 6
        next_trading_day += timedelta(days=1)

    st.subheader(f"🔮 Predicted Values for {next_trading_day.strftime('%Y-%m-%d')} (Next Trading Day):")

    # List to store predictions for the next day table
    next_day_predictions = []

    # Get the last actual closing price from the full data DataFrame (most recent day)
    last_actual_close_full_data = data['Close'].iloc[-1] if not data.empty else None

    # --- Predict for the Main Model (Close, Open, Volatility) ---
    # Use the already trained models and scalers from models_scalers_features_main
    try:
        next_day_pred_dfs_main = generate_predictions(last_day_features_df.copy(), models_scalers_features_main)

        predicted_price_main_close = next_day_pred_dfs_main.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_pred_dfs_main.get('Close', {}).empty else None
        predicted_price_main_open = next_day_pred_dfs_main.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_pred_dfs_main.get('Open', {}).empty else None
        predicted_volatility_main = next_day_pred_dfs_main.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_pred_dfs_main.get('Volatility', {}).empty else None


        next_day_predictions.append({
             'Model': model_choice,
             'Predicted Close': predicted_price_main_close,
             'Predicted Open': predicted_price_main_open,
             'Predicted Volatility': predicted_volatility_main
        })

        # Save the next trading day prediction for the main model
        if predicted_price_main_close is not None:
            # Actual close for next_trading_day is unknown at this point, so pass np.nan
            save_prediction(
                ticker,
                next_trading_day,
                predicted_price_main_close,
                np.nan,
                model_choice,
                datetime.now()
            )

        # Calculate the difference between the last actual close (today) and the predicted close (tomorrow) for the MAIN model
        if predicted_price_main_close is not None and last_actual_close_full_data is not None:
             actual_today_minus_predicted_tomorrow_main = last_actual_close_full_data - predicted_price_main_close
        else:
             actual_today_minus_predicted_tomorrow_main = None

        # Debugging: Print values before calculating difference
        st.write(f"Debug: Last Actual Close from full data: {last_actual_close_full_data}")
        st.write(f"Debug: Predicted Next Day Close (Main Model: {model_choice}): {predicted_price_main_close}")


    except Exception as e:
        st.warning(f"Error predicting next day values for main model ({model_choice}): {e}")
        predicted_price_main_close = None
        predicted_price_main_open = None
        predicted_volatility_main = None
        actual_today_minus_predicted_tomorrow_main = None


    # --- Predict for the Comparison Models (Close, Open, Volatility) ---
    # Iterate through the selected comparison models and get their next day predictions
    for compare_model_name in compare_models:
        # Skip the main model if it's also in the comparison list to avoid duplication
        if compare_model_name == model_choice:
            continue

        st.info(f"Generating next day predictions for comparison model: {compare_model_name}")
        try:
            # Train this comparison model on the training period data (df_train_period)
            # We need to train all three targets (Close, Open, Volatility) for the table
            models_scalers_features_compare = train_models(df_train_period.copy(), compare_model_name, perform_tuning=perform_tuning, messages_list=all_training_messages) # Pass messages_list

            if models_scalers_features_compare and models_scalers_features_compare.get('Close', {}).get('model'):
                 # Generate predictions for the next day using this comparison model
                 next_day_pred_dfs_compare = generate_predictions(last_day_features_df.copy(), models_scalers_features_compare)

                 predicted_price_compare_close = next_day_pred_dfs_compare.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_pred_dfs_compare.get('Close', {}).empty else None
                 predicted_price_compare_open = next_day_pred_dfs_compare.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_pred_dfs_compare.get('Open', {}).empty else None
                 predicted_volatility_compare = next_day_pred_dfs_compare.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_pred_dfs_compare.get('Volatility', {}).empty else None

                 next_day_predictions.append({
                      'Model': compare_model_name,
                      'Predicted Close': predicted_price_compare_close,
                      'Predicted Open': predicted_price_compare_open,
                      'Predicted Volatility': predicted_volatility_compare
                 })

                 # Save the next trading day prediction for the comparison model
                 if predicted_price_compare_close is not None:
                     save_prediction(
                         ticker,
                         next_trading_day,
                         predicted_price_compare_close,
                         np.nan, # Actual close is unknown for future date
                         compare_model_name,
                         datetime.now()
                     )

            else:
                 st.warning(f"Could not train or generate predictions for comparison model {compare_model_name}. Skipping.")


        except Exception as e:
            st.warning(f"Error generating next day predictions for comparison model ({compare_model_name}): {e}")


    # --- Display the table ---
    # Create a DataFrame from the list of predictions
    df_next_day_predictions = pd.DataFrame(next_day_predictions)

    if not df_next_day_predictions.empty:
        # Add the date column
        df_next_day_predictions['Date'] = next_trading_day
        # Reorder columns
        df_next_day_predictions = df_next_day_predictions[['Date', 'Model', 'Predicted Close', 'Predicted Open', 'Predicted Volatility']]
        # Sort by model name for consistent display
        df_next_day_predictions = df_next_day_predictions.sort_values('Model').reset_index(drop=True)

        st.dataframe(df_next_day_predictions.style.format({
             "Predicted Close": "{:.2f}",
             "Predicted Open": "{:.2f}",
             "Predicted Volatility": "{:.4f}" # Volatility might be a smaller number
         }))

        # Display the difference for the MAIN model below the table
        if actual_today_minus_predicted_tomorrow_main is not None:
             st.markdown(f"**Difference (Last Actual Close - Predicted Next Day Close) for Main Model ({model_choice}):** {actual_today_minus_predicted_tomorrow_main:.2f}")
        elif predicted_price_main_close is None:
             st.warning("Could not calculate difference for Main Model as next day predicted close is not available.")
        elif last_actual_close_full_data is None:
             st.warning("Could not calculate difference for Main Model as last actual close is not available.")


        # Optionally, add information about the best model from comparison below the table
        # Check if df_comparison is not empty and best_model_name is not N/A
        if not df_comparison.empty and best_model_name != "N/A":
             st.markdown(f"*(Based on recent data comparison for **Close Price**, **{best_model_name}** had the lowest RMSE ({best_model_rmse:.4f}) and %-RMSE ({best_model_pct_rmse:.2f}%) among the selected comparison models on the recent data period.)*")
        # Modify the condition here: display the "not available" message only if comparison models were selected,
        # but the df_comparison DataFrame (after dropping NaNs for RMSE) is empty.
        elif compare_models and df_comparison.empty:
             st.info("*(Model comparison results on recent data are not available. This may happen if there was an issue training or evaluating the comparison models.)*")


    else:
        st.warning("No next day predictions could be generated for any model.")


# --- Output 2: Graph of actual vs predicted closing for training period (t1 to t2) ---
# Generate predictions for the training period (Actual vs Predicted) - only need Close here
train_pred_dfs = generate_predictions(df_train_period.copy(), {'Close': models_scalers_features_main.get('Close')}) # Pass only Close info for the MAIN model


if 'Close' in train_pred_dfs and not train_pred_dfs['Close'].empty:
    train_pred_df = train_pred_dfs['Close']
    # Error metrics for training data (calculated on the data that was predicted on)
    actual_train = train_pred_df['Actual Close'].values.ravel()
    predicted_train = train_pred_df['Predicted Close'].values.ravel()
    # Only calculate metrics if there is actual data to compare against
    if len(actual_train) > 0:
        mae_train = mean_absolute_error(actual_train, predicted_train)
        rmse_train = np.sqrt(mean_squared_error(actual_train, predicted_train))

        # Calculate average actual close for the training period
        average_actual_train = actual_train.mean()
        # Calculate percentage error metrics for the training period
        pct_mae_train = (mae_train / average_actual_train) * 100 if average_actual_train > 0 else np.nan
        pct_rmse_train = (rmse_train / average_actual_train) * 100 if average_actual_train > 0 else np.nan


        st.markdown(f"Training Period ({start_bt.strftime('%Y-%m-%d')} to {end_bt.strftime('%Y-%m-%d')}) Metrics (Close Price):")
        # Display both absolute and percentage metrics
        st.markdown(f"MAE: {mae_train:.4f} ({pct_mae_train:.2f}%) | RMSE: {rmse_train:.4f} ({pct_rmse_train:.2f}%)")
    else:
        st.info("Not enough actual data in the training period to calculate metrics for Close Price.")


    # Training actual vs predicted plot
    fig_train = go.Figure()
    # Use distinct colors
    fig_train.add_trace(go.Scatter(x=train_pred_df['Date'], y=train_pred_df['Actual Close'], mode='lines', name='Actual', line=dict(color='blue')))
    fig_train.add_trace(go.Scatter(x=train_pred_df['Date'], y=train_pred_df['Predicted Close'], mode='lines', name='Predicted', line=dict(color='red')))
    fig_train.update_layout(
        title=f"Training Period: Actual vs Predicted Close Prices ({model_choice})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            range=[train_pred_df['Date'].min(), train_pred_df['Date'].max()] # Explicitly set x-axis range
        )
    )
    st.plotly_chart(fig_train, use_container_width=True)

else:
     st.warning("Could not generate training predictions for the selected period for Close Price.")


# --- Output 3: Table of difference for last 30 days of training period ---
# Generate predictions for the last 30 days of the training period - only need Close here
backtest_start_date_30_days = pd.to_datetime(end_bt) - timedelta(days=30)
df_backtest_30_days = df_features_full[(df_features_full['Date'] > backtest_start_date_30_days) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()

# Ensure the 30-day backtest period is within the trained period
df_backtest_30_days = df_backtest_30_days[df_backtest_30_days['Date'] >= pd.to_datetime(start_bt)].copy()

# Generate predictions for the backtest period (only Close needed here)
bt_dfs = generate_predictions(df_backtest_30_days.copy(), {'Close': models_scalers_features_main.get('Close')}) # Pass only Close info for the MAIN model


if 'Close' in bt_dfs and not bt_dfs['Close'].empty:
    bt_df = bt_dfs['Close']
    # Ensure columns are numeric before calculating metrics
    for col in ['Actual Close', 'Predicted Close']:
         if col in bt_df.columns:
              bt_df[col] = pd.to_numeric(bt_df[col], errors='coerce') # Corrected: bt[col] to bt_df[col]
    bt_df.dropna(subset=['Actual Close', 'Predicted Close'], inplace=True)


    if not bt_df.empty: # Check again after dropping NaNs
        # Error metrics for backtest data (last 30 days)
        actual_bt = bt_df['Actual Close'].values.ravel()
        predicted_bt = bt_df['Predicted Close'].values.ravel()
        # Only calculate metrics if there is actual data to compare against
        if len(actual_bt) > 0:
            mae_bt = mean_absolute_error(actual_bt, predicted_bt)
            rmse_bt = np.sqrt(mean_squared_error(actual_bt, predicted_bt))

            # Calculate average actual close for the backtest period
            average_actual_bt = actual_bt.mean()
            # Calculate percentage error metrics for the backtest period
            pct_mae_bt = (mae_bt / average_actual_bt) * 100 if average_actual_bt > 0 else np.nan
            pct_rmse_bt = (rmse_bt / average_actual_bt) * 100 if average_actual_bt > 0 else np.nan


            st.subheader(f"📉 Backtesting (Last 30 Days of Training Period: {bt_df['Date'].min().strftime('%Y-%m-%d')} to {bt_df['Date'].max().strftime('%Y-%m-%d')}) (Close Price)")
            # Display both absolute and percentage metrics
            st.markdown(f"Backtest MAE: {mae_bt:.4f} ({pct_mae_bt:.2f}%) | Backtest RMSE: {rmse_bt:.4f} ({pct_rmse_bt:.2f}%)")
        else:
             st.info("Not enough actual data in the last 30 days of the training period to calculate metrics for Close Price.")


        st.subheader("Backtest Data Table with Difference (Last 30 Days of Training Period) (Close Price)")
        st.dataframe(bt_df.style.format({"Actual Close": "{:.2f}", "Predicted Close": "{:.2f}", "Difference": "{:.2f}"}))

        # Save each row of the backtest DataFrame as a past prediction
        st.info(f"Saving backtest predictions for {ticker} (last 30 days of training period)...")
        for index, row in bt_df.iterrows():
            save_prediction(
                ticker,
                row['Date'], # prediction_for_date
                row['Predicted Close'],
                row['Actual Close'],
                model_choice, # Model used for this backtest is the main model
                datetime.now() # prediction_generation_date (when this backtest was run)
            )
    else:
        st.warning("Backtest DataFrame is empty after ensuring numeric types for metrics for Close Price.")
else:
    st.info("Could not generate backtest predictions for the last 30 days for Close Price. Ensure the training period includes the Backtest End Date and there are enough features.")


# --- Output 4: Future Predictions (Iterative) ---
st.subheader(f"🚀 Future {n_future} Days Predicted Close Prices (from {end_bt.strftime('%Y-%m-%d')})")

# Improved and more catchy explanation box for future predictions
st.info("""
### Unveiling the Future: How Monarch Predicts Ahead

Imagine looking into a crystal ball, but powered by data and algorithms! Monarch's future prediction isn't just a wild guess; it's a sophisticated process that learns from history and iteratively forecasts the path ahead.

1.  **Learning from the Echoes of the Past:** Our models delve deep into the stock's historical data up to your chosen "Training End Date." They analyze price movements, trading volumes, and calculate powerful technical indicators like Moving Averages, RSI, and MACD. This is where Monarch learns the unique rhythm and patterns of your selected stock.

2.  **The First Leap Forward:** Armed with this historical wisdom, the model makes its first crucial prediction: the closing price for the very next trading day after your training period ends.

3.  **The Iterative Dance:** Here's where the magic of iterative forecasting happens. To predict the *second* future day, Monarch doesn't just rely on history; it incorporates the *predicted* price from the *first* future day as if it were real data! It recalculates the technical indicators based on this new 'simulated' data point and uses everything to make the *next* prediction.

4.  **Building the Future Chain:** This powerful process repeats for each subsequent day you want to predict. Each new prediction becomes a building block, influencing the forecast for the day after, creating a seamless chain of predictions extending into the future.

**Think of it this way:** Monarch uses the latest available information, including its own previous predictions, to build a probable future scenario for the stock price, one day at a time.

**A Word of Caution:** While Monarch uses cutting-edge techniques, the stock market is influenced by countless real-world events (news, global economics, etc.) that are not part of this historical price data. These predictions are based solely on the patterns identified in the data and technical indicators. Always combine these insights with your own research and understanding. **This is not financial advice.**
""")


# Determine the maximum window size for feature calculation
# Include std_windows_list in the calculation
all_windows = lag_features_list + ma_windows_list + std_windows_list + [rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window]
# Filter out None values and ensure windows are positive integers
valid_windows = [w for w in all_windows if isinstance(w, int) and w > 0]
max_window = max(valid_windows) if valid_windows else 1 # Ensure at least 1 if no valid windows defined


# Find the index of the end_bt date in the original data
# Find the row in the original data with the date closest to end_bt but not after it
end_bt_dt = pd.to_datetime(end_bt)
closest_end_date_row = data[data['Date'] <= end_bt_dt].iloc[-1] if not data[data['Date'] <= end_bt_dt].empty else None

future_predictions_list = []

# Get the Close model info for future predictions (using the main model)
close_model_info = models_scalers_features_main.get('Close')
if close_model_info and close_model_info['model'] and close_model_info['scaler'] and close_model_info['features']:
    model_close = close_model_info['model']
    scaler_close = close_model_info['scaler']
    feature_cols_train_close = close_model_info['features']
else:
    st.warning("Close price model is not available for future predictions.")
    model_close = None # Ensure model_close is None if not available


if closest_end_date_row is not None and model_close is not None:
    end_bt_idx = closest_end_date_row.name # Get the index of the closest date

    # Get the historical context data (original OHLCV) ending at the closest_end_date_row
    # We need max_window rows *before* this date, plus this date itself, for feature calculation.
    # So, the slice needs to be max_window + 1 days long.
    start_historical_context_idx = max(0, end_bt_idx - max_window)
    df_historical_context = data.iloc[start_historical_context_idx : end_bt_idx + 1].copy()

    # Calculate features on this initial historical context
    # This will be the starting point for our iterative feature calculation
    df_iterative_features = create_features(df_historical_context.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)

    if not df_iterative_features.empty:
        # Start the iterative forecast with the last row of the feature-engineered historical context
        df_iterative = df_iterative_features.tail(1).copy()

        # Get the list of feature columns used in training the Close model (excluding Date and target)
        feature_names_train_close_only = [col for col in feature_cols_train_close if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]


        # Start the iterative forecast loop
        for i in range(n_future):
            # Determine the date for the current prediction step
            current_date = df_iterative['Date'].iloc[-1] + timedelta(days=1)
            # Skip weekends for the prediction date
            while current_date.weekday() >= 5: # Monday is 0, Sunday is 6
                current_date += timedelta(days=1)

            # Prepare the data for prediction (single row DataFrame)
            # Use the features from the previous predicted day (last row of df_iterative)
            # Ensure we only select the features that the Close model was trained on
            X_predict = df_iterative.tail(1).reindex(columns=feature_names_train_close_only, fill_value=0).copy()


            # Check if X_predict is empty after selecting columns
            if X_predict.empty:
                 st.warning(f"No features available for future prediction step {i+1} after reindexing for Close model. Stopping future forecast.")
                 break

            # Ensure the number of features matches the scaler
            if hasattr(scaler_close, 'n_features_in_') and X_predict.shape[1] != scaler_close.n_features_in_:
                 st.warning(f"Feature mismatch for future prediction step {i+1} for Close model. Expected {scaler_close.n_features_in_}, got {X_predict.shape[1]}. Stopping future forecast.")
                 break
            # Fallback check if scaler does not have n_features_in_
            elif X_predict.shape[1] != len(feature_names_train_close_only):
                 st.warning(f"Feature mismatch for future prediction step {i+1} for Close model. Expected {len(feature_names_train_close_only)}, got {X_predict.shape[1]}. Stopping future forecast.")
                 break


            X_predict_scaled = scaler_close.transform(X_predict)

            # Generate the prediction for the current day
            try:
                predicted_price = model_close.predict(X_predict_scaled)[0]
            except Exception as e:
                st.warning(f"Error during future prediction step {i+1} for Close model: {e}. Stopping future forecast.")
                break

            # Store the prediction
            future_predictions_list.append({'Date': current_date, 'Predicted Close': predicted_price})

            # --- Update data for the next iteration (Iterative Forecasting) ---
            # Create a new row for the next iteration with the predicted price
            # We need to simulate the next day's data (at least Close) to calculate features for the day after.
            # For simplicity, carry forward the last known Open, High, Low, Volume for future days.
            # We need to use the *last actual* Open, High, Low, Volume from the historical context
            last_actual_row = df_historical_context.iloc[-1]

            new_row_data = {'Date': current_date,
                            'Open': last_actual_row['Open'] if 'Open' in last_actual_row else np.nan, # Carry forward last known Actual Open
                            'High': last_actual_row['High'] if 'High' in last_actual_row else np.nan, # Carry forward last known Actual High
                            'Low': last_actual_row['Low'] if 'Low' in last_actual_row else np.nan,   # Carry forward last known Actual Low
                            'Close': predicted_price, # Use the predicted price as the Close for the next iteration
                            'Volume': last_actual_row['Volume'] if 'Volume' in last_actual_row else np.nan # Carry forward last known Actual Volume
                           }

            # Create a DataFrame for the new row
            new_row_df = pd.DataFrame([new_row_data])

            # Append the new row to the historical context data
            # Keep only the last max_window + 1 rows to maintain a sliding window for feature calculation
            df_historical_context = pd.concat([df_historical_context.tail(max_window), new_row_df], ignore_index=True)

            # Recalculate features on the updated sliding window of historical context + predicted data
            df_iterative_features = create_features(df_historical_context.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)

            # Update df_iterative to be the last row of the newly feature-engineered data for the next prediction step
            if not df_iterative_features.empty:
                 df_iterative = df_iterative_features.tail(1).copy()
            else:
                 st.warning(f"Feature calculation resulted in empty DataFrame at future prediction step {i+1}. Stopping future forecast.")
                 break


    else:
        st.warning(f"Initial feature calculation on historical context up to {end_bt.strftime('%Y-%m-%d')} resulted in an empty DataFrame. Cannot start future forecast.")


if future_predictions_list:
    future_df = pd.DataFrame(future_predictions_list)
    st.dataframe(future_df.style.format({"Predicted Close": "{:.2f}"}))

    fig_future = px.line(future_df, x='Date', y='Predicted Close', title=f"Future {n_future} Days Price Prediction ({model_choice})")
    fig_future.update_traces(line=dict(color='green'))
    st.plotly_chart(fig_future, use_container_width=True)

    # Save future predictions (actual_close will be NaN)
    st.info(f"Saving future predictions for {ticker} (next {n_future} days)...")
    for pred in future_predictions_list:
        save_prediction(
            ticker,
            pred['Date'],
            pred['Predicted Close'],
            np.nan, # Actual close is not known for future dates
            model_choice,
            datetime.now()
        )
else:
    st.warning("Could not generate future predictions. Ensure the training period includes the Backtest End Date and there are enough features, and the Close price model trained successfully.")


# --- New Section: Display Past Predictions ---
st.markdown("---")
st.header("🕰️ Historical Prediction Log")
st.markdown("""
This section displays a log of all predictions made by Monarch for the selected ticker,
including when the prediction was generated and the date it was *for*.
This allows you to see how predictions for a specific past date might have varied
depending on when they were made (i.e., based on different training data available at that time).
""")

past_predictions_df = load_past_predictions(ticker)

if not past_predictions_df.empty:
    # Sort for better readability
    past_predictions_df_display = past_predictions_df.sort_values(
        by=['prediction_for_date', 'prediction_generation_date'],
        ascending=[False, False] # Show most recent prediction_for_date first, then most recent generation date
    ).reset_index(drop=True)

    st.dataframe(past_predictions_df_display.style.format({
        "predicted_value": "{:.2f}",
        "actual_close": "{:.2f}",
        "prediction_generation_date": lambda x: x.strftime("%Y-%m-%d %H:%M"),
        "prediction_for_date": lambda x: x.strftime("%Y-%m-%d")
    }))

    # Optional: Plotting past predictions vs actuals where actuals are available
    st.subheader("Visualizing Past Predictions vs. Actuals (where actuals are known)")
    plot_df = past_predictions_df_display.dropna(subset=['actual_close']).copy()

    if not plot_df.empty:
        fig_past_predictions = go.Figure()

        # Add actual close prices
        # Group by prediction_for_date and take the first actual_close (should be consistent)
        actual_data_for_plot = plot_df.groupby('prediction_for_date')['actual_close'].first().reset_index()
        fig_past_predictions.add_trace(go.Scatter(
            x=actual_data_for_plot['prediction_for_date'],
            y=actual_data_for_plot['actual_close'],
            mode='lines+markers',
            name='Actual Close',
            line=dict(color='blue', width=3)
        ))

        # Add predicted values, differentiating by generation date and model
        colors = px.colors.qualitative.Plotly # Use a qualitative color scale
        color_idx = 0
        for model_name_log in plot_df['model_used'].unique():
            model_df = plot_df[plot_df['model_used'] == model_name_log].copy()
            for gen_date in model_df['prediction_generation_date'].unique():
                gen_date_df = model_df[model_df['prediction_generation_date'] == gen_date].copy()
                fig_past_predictions.add_trace(go.Scatter(
                    x=gen_date_df['prediction_for_date'],
                    y=gen_date_df['predicted_value'],
                    mode='lines+markers',
                    name=f'Pred ({model_name_log}) - Generated {gen_date.strftime("%Y-%m-%d")}',
                    line=dict(dash='dot', color=colors[color_idx % len(colors)]) # Dotted line for predictions
                ))
                color_idx += 1


        fig_past_predictions.update_layout(
            title=f'Historical Predicted vs. Actual Close Price for {ticker}',
            xaxis_title="Date of Prediction",
            yaxis_title="Price",
            hovermode="x unified"
        )
        st.plotly_chart(fig_past_predictions, use_container_width=True)
    else:
        st.info("No past predictions with known actual close prices to plot yet.")

else:
    st.info(f"No past predictions found for {ticker}. Run the app to generate and save predictions.")

# --- Feature Explanation Section ---
st.subheader("🧠 Feature Importance / Coefficients")
st.markdown("This section shows which features the model considered most important for making predictions during training. Note: This is shown for the **Close Price** model.")

# Get the Close model info for feature importance display (using the main model)
close_model_info = models_scalers_features_main.get('Close')

if close_model_info and close_model_info['model'] and close_model_info['features']:
    model = close_model_info['model']
    feature_cols_train = close_model_info['features']
    # Exclude original price/volume columns from feature names for display
    feature_names_display = [col for col in feature_cols_train if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
    importance_data = {}

    # Check if the model has feature_importances_ (for tree-based models)
    if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(feature_names_display):
        importance_data = {'Feature': feature_names_display, 'Importance': model.feature_importances_}
        st.markdown(f"**Feature Importances for {model_choice} (Close Price Model)**")
        df_importance = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
        st.dataframe(df_importance)
        fig_importance = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                                title=f'Feature Importances for {model_choice} (Close Price Model)')
        st.plotly_chart(fig_importance, use_container_width=True)

    # Check if the model has coef_ (for linear models)
    elif hasattr(model, 'coef_') and len(model.coef_) == len(feature_names_display):
        # Ensure the number of coefficients matches the number of features
        importance_data = {'Feature': feature_names_display, 'Coefficient': model.coef_}
        st.markdown(f"**Coefficients for {model_choice} (Close Price Model)**")
        df_coefficients = pd.DataFrame(importance_data).sort_values('Coefficient', ascending=False)
        st.dataframe(df_coefficients)
        fig_coefficients = px.bar(df_coefficients, x='Coefficient', y='Feature', orientation='h',
                                  title=f'Coefficients for {model_choice} (Close Price Model)')
        st.plotly_chart(fig_coefficients, use_container_width=True)

    else:
        st.info(f"Feature importance or coefficients are not available or do not match feature count for the selected Close Price model: {model_choice}.")
else:
    st.info("Close Price model has not been trained yet, or feature information is unavailable.")
